from __future__ import annotations

import copy
import json
from dataclasses import dataclass
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
from bilinear_lmmd.engine.mvfd_sbn import (
    equal_mean_teacher,
    squared_l2_feature_distillation,
)
from bilinear_lmmd.engine.preprocessing_study import _evaluate_model, _write_evaluation
from bilinear_lmmd.engine.shared_multiview_sbn import auxiliary_batchnorm_batch_stats
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device


PROTOCOL = "coffee17-mvsc-mvfd-sbn-v1"


@dataclass
class MultiLevelViewForward:
    logits: Tensor
    embedding: Tensor
    mid_feature: Tensor


def validate_mvsc_mvfd_sbn_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("MVSC-MVFD-SBN v1 dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("MVSC-MVFD-SBN v1 harus source_only.")

    model = cfg.get("model", {})
    if model.get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if model.get("head") != "gap":
        raise ValueError("Head harus GAP.")
    if str(model.get("classifier", "linear")) != "linear":
        raise ValueError("Classifier harus linear.")
    if model.get("out_indices") != [3, 4]:
        raise ValueError("MVSC-MVFD-SBN v1 harus out_indices [3, 4].")

    data = cfg.get("data", {})
    if data.get("augmentation_mode") != "preprocessing_study":
        raise ValueError("Augmentasi harus preprocessing_study.")
    if bool(data.get("object_crop", False)):
        raise ValueError("object_crop harus false.")

    training = cfg.get("training", {})
    if int(training.get("epochs", -1)) != 50:
        raise ValueError("MVSC-MVFD-SBN v1 harus 50 epoch.")
    if training.get("classification_loss") != "cross_entropy":
        raise ValueError("Loss klasifikasi harus cross_entropy.")
    if float(training.get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")

    fd = cfg.get("feature_distillation", {})
    if fd.get("method") != "mvsc_mvfd_sbn":
        raise ValueError("feature_distillation.method harus mvsc_mvfd_sbn.")
    if tuple(fd.get("views", ())) != VIEWS:
        raise ValueError(f"views harus tepat {VIEWS}.")
    if str(fd.get("primary_view", "")).upper() != "R0":
        raise ValueError("primary_view harus R0.")
    if tuple(fd.get("teacher_views", ())) != AUX_VIEWS:
        raise ValueError(f"teacher_views harus tepat {AUX_VIEWS}.")
    if fd.get("teacher_aggregation") != "equal_mean":
        raise ValueError("Teacher harus equal_mean.")
    if bool(fd.get("teacher_stop_gradient")) is not True:
        raise ValueError("Teacher harus stop-gradient.")
    if bool(fd.get("selective_batch_norm")) is not True:
        raise ValueError("Selective BatchNorm harus aktif.")
    if bool(fd.get("auxiliary_dropout")) is not False:
        raise ValueError("auxiliary_dropout harus false.")

    if abs(float(fd.get("aux_ce_weight_each", -1.0)) - 0.05) > 1.0e-12:
        raise ValueError("aux_ce_weight_each harus 0.05.")
    if abs(float(fd.get("feature_distill_weight", -1.0)) - 0.007) > 1.0e-12:
        raise ValueError("Final feature_distill_weight harus tetap 0.007.")
    if fd.get("feature_distill_loss") != "squared_l2":
        raise ValueError("Final feature loss harus squared_l2.")
    if abs(float(fd.get("mid_contrastive_weight", -1.0)) - 0.05) > 1.0e-12:
        raise ValueError("mid_contrastive_weight v1 dikunci 0.05.")
    if fd.get("mid_contrastive_loss") != "r0_anchor_multiview_supcon":
        raise ValueError("mid contrastive loss harus r0_anchor_multiview_supcon.")
    if abs(float(fd.get("mid_contrastive_temperature", -1.0)) - 0.07) > 1.0e-12:
        raise ValueError("mid_contrastive_temperature v1 harus 0.07.")
    if str(fd.get("mid_contrastive_anchor", "")).upper() != "R0":
        raise ValueError("mid_contrastive_anchor harus R0.")
    if tuple(fd.get("mid_contrastive_keys", ())) != AUX_VIEWS:
        raise ValueError(f"mid_contrastive_keys harus tepat {AUX_VIEWS}.")
    if int(fd.get("mid_feature_position", -1)) != 0:
        raise ValueError("mid_feature_position v1 harus 0.")
    if int(fd.get("mid_backbone_out_index", -1)) != 3:
        raise ValueError("mid_backbone_out_index v1 harus 3.")

    frontends = fd.get("frontends", {})
    if tuple(frontends) != VIEWS:
        raise ValueError("Frontend harus tepat R0/C0/F0/W0.")
    for arm in VIEWS:
        if str(frontends[arm].get("code", "")).upper() != arm:
            raise ValueError(f"Frontend {arm} memiliki code salah.")
    if str(cfg.get("preprocessing", {}).get("code", "")).upper() != "R0":
        raise ValueError("Evaluation/deployment preprocessing harus R0.")


def build_ml_mvfd_runtimes(
    cfg: dict,
    device: torch.device,
) -> dict[str, PreprocessingRuntime]:
    validate_mvsc_mvfd_sbn_config(cfg)
    return {
        arm: PreprocessingRuntime.from_config(
            cfg["feature_distillation"]["frontends"][arm],
            device,
        )
        for arm in VIEWS
    }


def _forward_primary_multilevel(
    model: AuxiliaryTrainingModel,
    images: Tensor,
) -> MultiLevelViewForward:
    features = model.base.encoder(images)
    if len(features) != 2:
        raise RuntimeError(f"Expected two exposed feature maps, got {len(features)}.")
    embedding = model.base.pool(features)
    logits = model.base.classifier(model.base.dropout(embedding))
    return MultiLevelViewForward(logits, embedding, features[0])


def _forward_auxiliary_multilevel(
    model: AuxiliaryTrainingModel,
    arm: str,
    images: Tensor,
) -> MultiLevelViewForward:
    if arm not in model.auxiliary_classifiers:
        raise KeyError(f"Auxiliary arm tidak dikenal: {arm}")
    features = model.base.encoder(images)
    if len(features) != 2:
        raise RuntimeError(f"Expected two exposed feature maps, got {len(features)}.")
    embedding = model.base.pool(features)
    logits = model.auxiliary_classifiers[arm](embedding)
    return MultiLevelViewForward(logits, embedding, features[0])


def r0_anchor_multiview_supcon(
    raw_mid: Tensor,
    teacher_mid: dict[str, Tensor],
    labels: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, dict[str, float]]:
    if raw_mid.ndim != 4:
        raise ValueError("R0 mid feature harus NCHW.")
    if temperature <= 0.0:
        raise ValueError("Contrastive temperature harus > 0.")
    if tuple(teacher_mid) != AUX_VIEWS:
        raise ValueError(f"Teacher mid views harus tepat {AUX_VIEWS}.")

    anchor = F.normalize(raw_mid.mean(dim=(2, 3)), dim=1)
    keys = []
    for arm in AUX_VIEWS:
        feature = teacher_mid[arm]
        if feature.shape != raw_mid.shape:
            raise ValueError(f"Shape mid feature {arm} berbeda dari R0.")
        keys.append(F.normalize(feature.detach().mean(dim=(2, 3)), dim=1))
    key_matrix = torch.cat(keys, dim=0)
    key_labels = labels.repeat(len(AUX_VIEWS))

    logits = anchor @ key_matrix.transpose(0, 1)
    logits = logits / temperature
    positive_mask = labels[:, None].eq(key_labels[None, :])
    negative_mask = ~positive_mask
    positive_count = positive_mask.sum(dim=1)
    negative_count = negative_mask.sum(dim=1)
    valid = (positive_count > 0) & (negative_count > 0)

    if bool(valid.any()):
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        per_anchor = -(
            (log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1)
            / positive_count.clamp_min(1).to(log_prob.dtype)
        )
        loss = per_anchor[valid].mean()
    else:
        loss = anchor.sum() * 0.0

    with torch.no_grad():
        cosine = anchor.detach() @ key_matrix.detach().transpose(0, 1)
        positive_cos = float(cosine[positive_mask].mean().item())
        negative_cos = (
            float(cosine[negative_mask].mean().item())
            if bool(negative_mask.any())
            else 0.0
        )
        diagnostics = {
            "positive_keys_mean": float(positive_count.float().mean().item()),
            "negative_keys_mean": float(negative_count.float().mean().item()),
            "positive_cosine": positive_cos,
            "negative_cosine": negative_cos,
            "cosine_gap": positive_cos - negative_cos,
        }
    return loss, diagnostics


def _flat_cos(left: Tensor, right: Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.detach().flatten(1),
            right.detach().flatten(1),
            dim=1,
        ).mean().item()
    )


def _flat_l2(left: Tensor, right: Tensor) -> float:
    return float(
        torch.linalg.vector_norm(
            left.detach().flatten(1) - right.detach().flatten(1),
            ord=2,
            dim=1,
        ).mean().item()
    )


def train_mvsc_mvfd_sbn(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_primary_initial_sha256: str,
    resume: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_mvsc_mvfd_sbn_config(cfg)

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

    runtimes = build_ml_mvfd_runtimes(cfg, device)
    eval_runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    train_cfg = cfg["training"]
    fd = cfg["feature_distillation"]
    aux_weight = float(fd["aux_ce_weight_each"])
    final_weight = float(fd["feature_distill_weight"])
    contrastive_weight = float(fd["mid_contrastive_weight"])
    contrastive_temperature = float(fd["mid_contrastive_temperature"])

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.1))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
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
        print(f"RESUME MVSC-MVFD-SBN: epoch {start_epoch + 1}/{epochs}", flush=True)

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
            "mid_contrastive_loss": 0.0,
            "weighted_mid_contrastive_loss": 0.0,
            "mid_positive_keys_mean": 0.0,
            "mid_negative_keys_mean": 0.0,
            "mid_positive_cosine": 0.0,
            "mid_negative_cosine": 0.0,
            "mid_cosine_gap": 0.0,
            "r0_teacher_cos": 0.0,
            "r0_teacher_l2": 0.0,
            "r0_mid_teacher_cos": 0.0,
            "r0_mid_teacher_l2": 0.0,
            "c0_disagreement": 0.0,
            "f0_disagreement": 0.0,
            "w0_disagreement": 0.0,
        }
        samples = 0
        progress = tqdm(
            loaders.train,
            desc=f"MVSC-MVFD-SBN epoch {epoch_number}/{epochs}",
        )

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            raw = _forward_primary_multilevel(model, runtimes["R0"](images))
            primary_ce = loss_fn(raw.logits, labels)

            aux_ce: dict[str, Tensor] = {}
            aux_embedding: dict[str, Tensor] = {}
            aux_mid: dict[str, Tensor] = {}
            aux_logits: dict[str, Tensor] = {}
            with auxiliary_batchnorm_batch_stats(model.base):
                for arm in AUX_VIEWS:
                    view = _forward_auxiliary_multilevel(
                        model,
                        arm,
                        runtimes[arm](images),
                    )
                    aux_ce[arm] = loss_fn(view.logits, labels)
                    aux_embedding[arm] = view.embedding
                    aux_mid[arm] = view.mid_feature
                    aux_logits[arm] = view.logits

            final_teacher = equal_mean_teacher(aux_embedding)
            mid_teacher = torch.stack(
                [aux_mid[arm] for arm in AUX_VIEWS],
                dim=0,
            ).mean(dim=0)

            final_loss = squared_l2_feature_distillation(
                raw.embedding,
                final_teacher,
            )
            contrastive_loss, contrastive_diag = r0_anchor_multiview_supcon(
                raw.mid_feature,
                aux_mid,
                labels,
                temperature=contrastive_temperature,
            )
            weighted_final = final_weight * final_loss
            weighted_mid = contrastive_weight * contrastive_loss
            loss = (
                primary_ce
                + aux_weight * sum(aux_ce.values())
                + weighted_final
                + weighted_mid
            )
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            samples += batch_size
            totals["loss"] += float(loss.item())
            totals["primary_ce"] += float(primary_ce.item())
            totals["feature_loss"] += float(final_loss.item())
            totals["weighted_feature_loss"] += float(weighted_final.item())
            totals["mid_contrastive_loss"] += float(contrastive_loss.item())
            totals["weighted_mid_contrastive_loss"] += float(weighted_mid.item())
            totals["mid_positive_keys_mean"] += contrastive_diag["positive_keys_mean"]
            totals["mid_negative_keys_mean"] += contrastive_diag["negative_keys_mean"]
            totals["mid_positive_cosine"] += contrastive_diag["positive_cosine"]
            totals["mid_negative_cosine"] += contrastive_diag["negative_cosine"]
            totals["mid_cosine_gap"] += contrastive_diag["cosine_gap"]
            totals["r0_teacher_cos"] += (
                F.cosine_similarity(
                    raw.embedding.detach(),
                    final_teacher.detach(),
                    dim=1,
                ).mean().item() * batch_size
            )
            totals["r0_teacher_l2"] += (
                torch.linalg.vector_norm(
                    raw.embedding.detach() - final_teacher.detach(),
                    ord=2,
                    dim=1,
                ).mean().item() * batch_size
            )
            totals["r0_mid_teacher_cos"] += (
                _flat_cos(raw.mid_feature, mid_teacher) * batch_size
            )
            totals["r0_mid_teacher_l2"] += (
                _flat_l2(raw.mid_feature, mid_teacher) * batch_size
            )

            raw_pred = raw.logits.detach().argmax(1)
            for arm in AUX_VIEWS:
                key = arm.lower()
                totals[f"{key}_ce"] += float(aux_ce[arm].item())
                aux_pred = aux_logits[arm].detach().argmax(1)
                totals[f"{key}_disagreement"] += float(
                    (aux_pred != raw_pred).sum().item()
                )

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                final=f"{final_loss.item():.3f}",
                mvsc=f"{contrastive_loss.item():.3f}",
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
            "mid_contrastive_loss": totals["mid_contrastive_loss"] / batches,
            "weighted_mid_contrastive_loss": (
                totals["weighted_mid_contrastive_loss"] / batches
            ),
            "mid_positive_keys_mean": totals["mid_positive_keys_mean"] / batches,
            "mid_negative_keys_mean": totals["mid_negative_keys_mean"] / batches,
            "mid_positive_cosine": totals["mid_positive_cosine"] / batches,
            "mid_negative_cosine": totals["mid_negative_cosine"] / batches,
            "mid_cosine_gap": totals["mid_cosine_gap"] / batches,
            "r0_teacher_cos": totals["r0_teacher_cos"] / sample_denom,
            "r0_teacher_l2": totals["r0_teacher_l2"] / sample_denom,
            "r0_mid_teacher_cos": totals["r0_mid_teacher_cos"] / sample_denom,
            "r0_mid_teacher_l2": totals["r0_mid_teacher_l2"] / sample_denom,
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
                    "mid_contrastive_loss": record["mid_contrastive_loss"],
                    "weighted_mid_contrastive_loss": record["weighted_mid_contrastive_loss"],
                    "mid_positive_cosine": record["mid_positive_cosine"],
                    "mid_negative_cosine": record["mid_negative_cosine"],
                    "mid_cosine_gap": record["mid_cosine_gap"],
                    "f0_ce": record["f0_ce"],
                    "r0_teacher_cos": record["r0_teacher_cos"],
                    "r0_mid_teacher_cos": record["r0_mid_teacher_cos"],
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
                    "primary_initial_model_state_sha256": (
                        model.primary_initial_sha256
                    ),
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
