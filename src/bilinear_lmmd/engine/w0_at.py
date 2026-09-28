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
from bilinear_lmmd.engine.preprocessing_study import _evaluate_model, _write_evaluation
from bilinear_lmmd.engine.shared_multiview_sbn import auxiliary_batchnorm_batch_stats
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-w0-at-v1"


def validate_w0_at_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("W0-AT V1 dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("W0-AT V1 harus source_only.")

    model = cfg.get("model", {})
    if model.get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if model.get("head") != "gap":
        raise ValueError("Head harus GAP.")
    if str(model.get("classifier", "linear")) != "linear":
        raise ValueError("Classifier harus linear.")
    if model.get("out_indices") != [3, 4]:
        raise ValueError("W0-AT V1 harus out_indices [3, 4].")

    data = cfg.get("data", {})
    if data.get("augmentation_mode") != "preprocessing_study":
        raise ValueError("Augmentasi harus preprocessing_study.")
    if bool(data.get("object_crop", False)):
        raise ValueError("object_crop harus false.")

    training = cfg.get("training", {})
    if int(training.get("epochs", -1)) != 50:
        raise ValueError("W0-AT V1 harus 50 epoch.")
    if training.get("classification_loss") != "cross_entropy":
        raise ValueError("Loss klasifikasi harus cross_entropy.")
    if float(training.get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")

    at = cfg.get("attention_transfer", {})
    expected = {
        "method": "w0_spatial_attention_transfer",
        "teacher_view": "W0",
        "teacher_stop_gradient": True,
        "selective_batch_norm": True,
        "attention_backbone_out_index": 3,
        "attention_feature_position": 0,
        "attention_power": 2,
        "channel_reduction": "mean",
        "normalization": "l2",
        "loss": "mse_mean",
        "beta": 1000.0,
        "epoch1_weighted_at_to_ce_abort_ratio": 1.0,
    }
    for key, value in expected.items():
        if at.get(key) != value:
            raise ValueError(
                f"attention_transfer.{key} harus {value!r}, observed={at.get(key)!r}"
            )

    frontend = at.get("teacher_frontend", {})
    if str(frontend.get("code", "")).upper() != "W0":
        raise ValueError("Teacher frontend harus W0.")
    if frontend.get("method") != "haar4_visushrink_soft_reconstruction":
        raise ValueError("Teacher W0 harus memakai frozen Haar4+VisuShrink.")
    if int(frontend.get("wavelet_levels", -1)) != 4:
        raise ValueError("Teacher W0 harus wavelet_levels=4.")

    if str(cfg.get("preprocessing", {}).get("code", "")).upper() != "R0":
        raise ValueError("Evaluation/deployment preprocessing harus R0.")


def activation_attention(feature: Tensor, eps: float = 1.0e-12) -> Tensor:
    if feature.ndim != 4:
        raise ValueError("Feature attention harus NCHW.")
    # Exact mechanism used in the public ICLR 2017 Attention Transfer code:
    # mean of squared activations across channels, flatten, L2 normalize.
    attention = feature.pow(2).mean(dim=1).flatten(1)
    return F.normalize(attention, p=2, dim=1, eps=eps)


def attention_transfer_loss(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError("Student/teacher feature shape berbeda.")
    q_student = activation_attention(student)
    q_teacher = activation_attention(teacher.detach())
    return (q_student - q_teacher).pow(2).mean()


def _forward_primary(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor]:
    features = model.encoder(images)
    if len(features) != 2:
        raise RuntimeError(f"Expected two exposed feature maps, got {len(features)}.")
    embedding = model.pool(features)
    logits = model.classifier(model.dropout(embedding))
    return logits, features[0]


def _forward_teacher_mid(model: nn.Module, images: Tensor) -> Tensor:
    features = model.encoder(images)
    if len(features) != 2:
        raise RuntimeError(f"Expected two exposed feature maps, got {len(features)}.")
    return features[0]


def train_w0_at(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_model_sha256: str,
    resume: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_w0_at_config(cfg)

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")

    model = build_model(cfg["model"]).to(device)
    from bilinear_lmmd.core.reproducibility import model_state_fingerprint
    initial_sha = model_state_fingerprint(model)
    if initial_sha != expected_initial_model_sha256:
        raise RuntimeError(
            "Initial model berbeda dari matched R0 reference: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )

    raw_runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    w0_runtime = PreprocessingRuntime.from_config(
        cfg["attention_transfer"]["teacher_frontend"],
        device,
    )

    training_cfg = cfg["training"]
    at_cfg = cfg["attention_transfer"]
    beta = float(at_cfg["beta"])
    abort_ratio = float(at_cfg["epoch1_weighted_at_to_ce_abort_ratio"])

    ce_fn = nn.CrossEntropyLoss(
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
            "model",
            "optimizer",
            "scheduler",
            "history",
            "best_f1",
            "rng_state",
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
        print(f"RESUME W0-AT: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        set_study_epoch(loaders.train, epoch)
        model.train()

        totals = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "at_loss": 0.0,
            "weighted_at_loss": 0.0,
            "attention_cosine": 0.0,
        }
        batches = 0
        samples = 0

        progress = tqdm(
            loaders.train,
            desc=f"W0-AT epoch {epoch_number}/{epochs}",
        )

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            raw_images = raw_runtime(images)
            logits, raw_mid = _forward_primary(model, raw_images)
            ce_loss = ce_fn(logits, labels)

            with torch.no_grad():
                with auxiliary_batchnorm_batch_stats(model):
                    w0_mid = _forward_teacher_mid(model, w0_runtime(images))

            at_loss = attention_transfer_loss(raw_mid, w0_mid)
            weighted_at = beta * at_loss
            loss = ce_loss + weighted_at
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            batches += 1
            samples += batch_size
            totals["loss"] += float(loss.item())
            totals["ce_loss"] += float(ce_loss.item())
            totals["at_loss"] += float(at_loss.item())
            totals["weighted_at_loss"] += float(weighted_at.item())
            totals["attention_cosine"] += (
                F.cosine_similarity(
                    activation_attention(raw_mid.detach()),
                    activation_attention(w0_mid.detach()),
                    dim=1,
                ).mean().item()
                * batch_size
            )

            progress.set_postfix(
                ce=f"{ce_loss.item():.4f}",
                at=f"{at_loss.item():.6f}",
                wat=f"{weighted_at.item():.4f}",
            )

        mean_ce = totals["ce_loss"] / max(batches, 1)
        mean_at = totals["at_loss"] / max(batches, 1)
        mean_weighted_at = totals["weighted_at_loss"] / max(batches, 1)
        weighted_ratio = mean_weighted_at / max(mean_ce, 1.0e-12)

        if epoch == 0 and weighted_ratio > abort_ratio:
            failure = {
                "format": "bilinear_lmmd.w0_at.loss_scale_failure.v1",
                "epoch": 1,
                "mean_ce": mean_ce,
                "mean_at": mean_at,
                "beta": beta,
                "mean_weighted_at": mean_weighted_at,
                "weighted_at_to_ce_ratio": weighted_ratio,
                "abort_ratio": abort_ratio,
                "decision": "ABORT_LOSS_CONTRACT_FAILURE",
            }
            (run_dir / "loss_scale_failure.json").write_text(
                json.dumps(failure, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "W0-AT epoch-1 weighted attention loss mendominasi CE: "
                f"ratio={weighted_ratio:.6f} > {abort_ratio:.6f}"
            )

        scheduler.step()

        metrics, _, _, _, _ = _evaluate_model(
            model,
            loaders.val,
            raw_runtime,
            device,
            loaders.classes,
            hard_groups,
        )

        record = {
            "epoch": epoch_number,
            "loss": totals["loss"] / max(batches, 1),
            "ce_loss": mean_ce,
            "at_loss": mean_at,
            "weighted_at_loss": mean_weighted_at,
            "weighted_at_to_ce_ratio": weighted_ratio,
            "attention_cosine": totals["attention_cosine"] / max(samples, 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            json.dumps(
                {
                    "epoch": epoch_number,
                    "ce_loss": record["ce_loss"],
                    "at_loss": record["at_loss"],
                    "weighted_at_loss": record["weighted_at_loss"],
                    "weighted_at_to_ce_ratio": record["weighted_at_to_ce_ratio"],
                    "attention_cosine": record["attention_cosine"],
                    "macro_f1": metrics["macro_f1"],
                    "hard_class_f1": metrics["hard_class_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )

        is_best = float(metrics["macro_f1"]) > best_f1
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
            "initial_model_state_sha256": initial_sha,
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
                    "initial_model_state_sha256": initial_sha,
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
    eval_model = build_model(eval_cfg["model"]).to(device)
    eval_model.load_state_dict(best["model"])

    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        eval_model,
        loaders.val,
        raw_runtime,
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

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": best_f1,
        "epoch1_weighted_at_to_ce_ratio": float(
            history[0]["weighted_at_to_ce_ratio"]
        ),
        "best_epoch_attention_cosine": float(
            max(history, key=lambda row: float(row["source"]["macro_f1"]))[
                "attention_cosine"
            ]
        ),
    }
