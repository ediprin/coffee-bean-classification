from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.at_sbn import (
    AUX_VIEWS,
    VIEWS,
    AuxiliaryTrainingModel,
    evaluate_auxiliary_view,
)
from bilinear_lmmd.engine.preprocessing_study import _evaluate_model, _write_evaluation
from bilinear_lmmd.engine.shared_multiview_sbn import auxiliary_batchnorm_batch_stats
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device


PROTOCOL = "coffee17-mvfd-sbn-v1"


def validate_mvfd_sbn_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("MVFD-SBN v1 dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("MVFD-SBN v1 harus source_only.")
    model_cfg = cfg.get("model", {})
    if model_cfg.get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if model_cfg.get("head") != "gap":
        raise ValueError("Head harus GAP.")
    if str(model_cfg.get("classifier", "linear")) != "linear":
        raise ValueError("MVFD-SBN membutuhkan classifier linear.")
    if model_cfg.get("out_indices") != [4]:
        raise ValueError("out_indices harus [4].")
    data_cfg = cfg.get("data", {})
    if data_cfg.get("augmentation_mode") != "preprocessing_study":
        raise ValueError("Augmentasi harus preprocessing_study.")
    if bool(data_cfg.get("object_crop", False)):
        raise ValueError("object_crop harus false.")
    training_cfg = cfg.get("training", {})
    if int(training_cfg.get("epochs", -1)) != 50:
        raise ValueError("MVFD-SBN v1 harus 50 epoch.")
    if training_cfg.get("classification_loss") != "cross_entropy":
        raise ValueError("Loss hard-label harus cross_entropy.")
    if float(training_cfg.get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")

    fd = cfg.get("feature_distillation", {})
    if fd.get("method") != "mvfd_sbn":
        raise ValueError("feature_distillation.method harus mvfd_sbn.")
    if tuple(fd.get("views", ())) != VIEWS:
        raise ValueError(f"views harus tepat {VIEWS}.")
    if str(fd.get("primary_view", "")).upper() != "R0":
        raise ValueError("primary_view harus R0.")
    if tuple(fd.get("teacher_views", ())) != AUX_VIEWS:
        raise ValueError(f"teacher_views harus tepat {AUX_VIEWS}.")
    if fd.get("teacher_aggregation") != "equal_mean":
        raise ValueError("teacher_aggregation v1 harus equal_mean.")
    if bool(fd.get("teacher_stop_gradient")) is not True:
        raise ValueError("Teacher prototype harus stop-gradient.")
    if bool(fd.get("selective_batch_norm")) is not True:
        raise ValueError("selective_batch_norm harus true.")
    if bool(fd.get("auxiliary_dropout")) is not False:
        raise ValueError("MVFD-SBN v1 mengunci auxiliary_dropout=false.")
    if abs(float(fd.get("aux_ce_weight_each", -1.0)) - 0.05) > 1.0e-12:
        raise ValueError("aux_ce_weight_each v1 dikunci 0.05.")
    if abs(float(fd.get("feature_distill_weight", -1.0)) - 0.007) > 1.0e-12:
        raise ValueError("feature_distill_weight v1 dikunci 0.007.")
    if fd.get("feature_distill_loss") != "squared_l2":
        raise ValueError("feature_distill_loss v1 harus squared_l2.")

    frontends = fd.get("frontends", {})
    if tuple(frontends) != VIEWS:
        raise ValueError("Frontend harus tepat R0/C0/F0/W0.")
    for arm in VIEWS:
        if str(frontends[arm].get("code", "")).upper() != arm:
            raise ValueError(f"Frontend {arm} memiliki code salah.")
    if str(cfg.get("preprocessing", {}).get("code", "")).upper() != "R0":
        raise ValueError("Evaluation/deployment preprocessing harus R0.")


def build_mvfd_runtimes(
    cfg: dict,
    device: torch.device,
) -> dict[str, PreprocessingRuntime]:
    validate_mvfd_sbn_config(cfg)
    return {
        arm: PreprocessingRuntime.from_config(
            cfg["feature_distillation"]["frontends"][arm],
            device,
        )
        for arm in VIEWS
    }


def equal_mean_teacher(embeddings: dict[str, Tensor]) -> Tensor:
    if tuple(embeddings) != AUX_VIEWS:
        raise ValueError(f"Teacher embedding keys harus tepat {AUX_VIEWS}.")
    return torch.stack([embeddings[arm] for arm in AUX_VIEWS], dim=0).mean(dim=0)


def squared_l2_feature_distillation(
    student: Tensor,
    teacher: Tensor,
) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError("Student/teacher feature shape berbeda.")
    return (student - teacher.detach()).square().sum(dim=1).mean()


def _batch_cosine(left: Tensor, right: Tensor) -> float:
    return float(
        F.cosine_similarity(left.detach(), right.detach(), dim=1).mean().item()
    )


def _batch_l2(left: Tensor, right: Tensor) -> float:
    return float(
        torch.linalg.vector_norm(
            left.detach() - right.detach(),
            ord=2,
            dim=1,
        ).mean().item()
    )


def _mean_norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach(), ord=2, dim=1).mean().item())


def train_mvfd_sbn(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_primary_initial_sha256: str,
    resume: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_mvfd_sbn_config(cfg)

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")

    model = AuxiliaryTrainingModel(cfg["model"]).to(device)
    if model.primary_initial_sha256 != expected_primary_initial_sha256:
        raise RuntimeError(
            "Primary initial model berbeda dari matched R0 reference: "
            f"{model.primary_initial_sha256} != {expected_primary_initial_sha256}"
        )

    runtimes = build_mvfd_runtimes(cfg, device)
    eval_runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)

    training_cfg = cfg["training"]
    fd_cfg = cfg["feature_distillation"]
    aux_ce_weight = float(fd_cfg["aux_ce_weight_each"])
    feature_weight = float(fd_cfg["feature_distill_weight"])

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.1))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError("Checkpoint resume berasal dari kontrak berbeda.")
        if checkpoint.get("classes") != loaders.classes:
            raise RuntimeError("Urutan kelas checkpoint berbeda.")
        required = {
            "model", "optimizer", "scheduler", "history",
            "best_f1", "rng_state",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint resume tidak lengkap: {missing}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME MVFD-SBN: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        set_study_epoch(loaders.train, epoch)
        model.train()

        totals = {
            "loss": 0.0,
            "primary_ce": 0.0,
            "c0_ce": 0.0,
            "f0_ce": 0.0,
            "w0_ce": 0.0,
            "feature_loss": 0.0,
            "weighted_feature_loss": 0.0,
            "teacher_norm": 0.0,
            "r0_norm": 0.0,
            "c0_norm": 0.0,
            "f0_norm": 0.0,
            "w0_norm": 0.0,
            "r0_teacher_cos": 0.0,
            "r0_teacher_l2": 0.0,
            "r0_c0_cos": 0.0,
            "r0_f0_cos": 0.0,
            "r0_w0_cos": 0.0,
            "r0_c0_l2": 0.0,
            "r0_f0_l2": 0.0,
            "r0_w0_l2": 0.0,
            "c0_disagreement": 0.0,
            "f0_disagreement": 0.0,
            "w0_disagreement": 0.0,
        }
        samples = 0
        progress = tqdm(loaders.train, desc=f"MVFD-SBN epoch {epoch_number}/{epochs}")

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            raw_input = runtimes["R0"](images)
            raw = model.forward_primary(raw_input, labels=labels)
            primary_ce = loss_fn(raw.logits, labels)

            aux_ce: dict[str, Tensor] = {}
            aux_embedding: dict[str, Tensor] = {}
            aux_logits: dict[str, Tensor] = {}
            with auxiliary_batchnorm_batch_stats(model.base):
                for arm in AUX_VIEWS:
                    view_input = runtimes[arm](images)
                    view = model.forward_auxiliary(arm, view_input)
                    aux_embedding[arm] = view.embedding
                    aux_logits[arm] = view.logits
                    aux_ce[arm] = loss_fn(view.logits, labels)

            teacher = equal_mean_teacher(aux_embedding)
            feature_loss = squared_l2_feature_distillation(
                raw.embedding,
                teacher,
            )
            weighted_feature_loss = feature_weight * feature_loss
            loss = (
                primary_ce
                + aux_ce_weight * sum(aux_ce.values())
                + weighted_feature_loss
            )
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            samples += batch_size
            totals["loss"] += float(loss.item())
            totals["primary_ce"] += float(primary_ce.item())
            totals["feature_loss"] += float(feature_loss.item())
            totals["weighted_feature_loss"] += float(weighted_feature_loss.item())

            teacher_detached = teacher.detach()
            totals["teacher_norm"] += _mean_norm(teacher_detached) * batch_size
            totals["r0_norm"] += _mean_norm(raw.embedding) * batch_size
            totals["r0_teacher_cos"] += _batch_cosine(raw.embedding, teacher_detached) * batch_size
            totals["r0_teacher_l2"] += _batch_l2(raw.embedding, teacher_detached) * batch_size

            raw_pred = raw.logits.detach().argmax(1)
            for arm in AUX_VIEWS:
                key = arm.lower()
                totals[f"{key}_ce"] += float(aux_ce[arm].item())
                totals[f"{key}_norm"] += _mean_norm(aux_embedding[arm]) * batch_size
                totals[f"r0_{key}_cos"] += _batch_cosine(
                    raw.embedding,
                    aux_embedding[arm],
                ) * batch_size
                totals[f"r0_{key}_l2"] += _batch_l2(
                    raw.embedding,
                    aux_embedding[arm],
                ) * batch_size
                aux_pred = aux_logits[arm].detach().argmax(1)
                totals[f"{key}_disagreement"] += float(
                    (aux_pred != raw_pred).sum().item()
                )

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                raw=f"{primary_ce.item():.4f}",
                feat=f"{feature_loss.item():.3f}",
                f0=f"{aux_ce['F0'].item():.4f}",
            )

        scheduler.step()
        metrics, _, _, _, _ = _evaluate_model(
            model.base,
            loaders.val,
            eval_runtime,
            device,
            loaders.classes,
            hard_groups,
        )
        batches = max(len(loaders.train), 1)
        sample_denom = max(samples, 1)
        record = {
            "epoch": epoch_number,
            "loss": totals["loss"] / batches,
            "primary_ce": totals["primary_ce"] / batches,
            "c0_ce": totals["c0_ce"] / batches,
            "f0_ce": totals["f0_ce"] / batches,
            "w0_ce": totals["w0_ce"] / batches,
            "feature_loss": totals["feature_loss"] / batches,
            "weighted_feature_loss": totals["weighted_feature_loss"] / batches,
            "teacher_norm": totals["teacher_norm"] / sample_denom,
            "r0_norm": totals["r0_norm"] / sample_denom,
            "c0_norm": totals["c0_norm"] / sample_denom,
            "f0_norm": totals["f0_norm"] / sample_denom,
            "w0_norm": totals["w0_norm"] / sample_denom,
            "r0_teacher_cos": totals["r0_teacher_cos"] / sample_denom,
            "r0_teacher_l2": totals["r0_teacher_l2"] / sample_denom,
            "r0_c0_cos": totals["r0_c0_cos"] / sample_denom,
            "r0_f0_cos": totals["r0_f0_cos"] / sample_denom,
            "r0_w0_cos": totals["r0_w0_cos"] / sample_denom,
            "r0_c0_l2": totals["r0_c0_l2"] / sample_denom,
            "r0_f0_l2": totals["r0_f0_l2"] / sample_denom,
            "r0_w0_l2": totals["r0_w0_l2"] / sample_denom,
            "c0_disagreement": totals["c0_disagreement"] / sample_denom,
            "f0_disagreement": totals["f0_disagreement"] / sample_denom,
            "w0_disagreement": totals["w0_disagreement"] / sample_denom,
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch_number,
                    "loss": record["loss"],
                    "primary_ce": record["primary_ce"],
                    "feature_loss": record["feature_loss"],
                    "weighted_feature_loss": record["weighted_feature_loss"],
                    "f0_ce": record["f0_ce"],
                    "r0_teacher_cos": record["r0_teacher_cos"],
                    "r0_teacher_l2": record["r0_teacher_l2"],
                    "macro_f1": metrics["macro_f1"],
                    "hard_class_f1": metrics["hard_class_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )

        is_best = metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch_number,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "primary_initial_model_state_sha256": model.primary_initial_sha256,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "epoch": epoch_number,
                    "best_f1": best_f1,
                    "weights": "raw_primary",
                    "run_contract_sha256": run_contract_sha256,
                    "primary_initial_model_state_sha256": model.primary_initial_sha256,
                },
                best_path,
            )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Training selesai tanpa best/last checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    eval_cfg = copy.deepcopy(best["config"])
    eval_cfg["model"]["pretrained"] = False
    eval_model = AuxiliaryTrainingModel(eval_cfg["model"]).to(device)
    eval_model.load_state_dict(best["model"])

    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        eval_model.base,
        loaders.val,
        eval_runtime,
        device,
        loaders.classes,
        hard_groups,
    )
    _write_evaluation(
        run_dir / "validation",
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        paths=paths,
        classes=loaders.classes,
        checkpoint=best_path,
        split="val",
    )

    auxiliary_validation = {}
    for arm in AUX_VIEWS:
        auxiliary_validation[arm] = evaluate_auxiliary_view(
            eval_model,
            loaders.val,
            runtimes[arm],
            arm,
            device,
            loaders.classes,
            hard_groups,
        )
    (run_dir / "validation" / "auxiliary_metrics.json").write_text(
        json.dumps(auxiliary_validation, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "primary_initial_model_state_sha256": model.primary_initial_sha256,
    }
