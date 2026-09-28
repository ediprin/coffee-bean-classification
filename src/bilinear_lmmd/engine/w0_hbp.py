from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    model_state_fingerprint,
    restore_rng_state,
    seed_everything,
)
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.preprocessing_study import _evaluate_model, _write_evaluation
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-w0-hbp-matched-v1"


def validate_hbp_preprocessing_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("Matched W0-HBP dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("Matched W0-HBP harus source_only.")

    model = cfg.get("model", {})
    if model.get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if model.get("head") != "hbp":
        raise ValueError("Head harus HBP.")
    if model.get("out_indices") != [1, 3, 4]:
        raise ValueError("HBP harus out_indices [1, 3, 4].")
    if int(model.get("projection_dim", -1)) != 512:
        raise ValueError("projection_dim HBP harus 512.")
    if str(model.get("classifier", "linear")) != "linear":
        raise ValueError("Classifier harus linear.")
    if int(model.get("num_classes", -1)) != 17:
        raise ValueError("Coffee17 harus 17 kelas.")

    data = cfg.get("data", {})
    if data.get("augmentation_mode") != "preprocessing_study":
        raise ValueError("Augmentasi harus preprocessing_study.")
    if bool(data.get("object_crop", False)):
        raise ValueError("object_crop harus false.")
    if int(data.get("image_size", -1)) != 224:
        raise ValueError("image_size harus 224.")
    if int(data.get("batch_size", -1)) != 32:
        raise ValueError("batch_size harus 32.")
    if list(data.get("rotation_angles", [])) != [0, 45, 90, 135, 180, 225, 270]:
        raise ValueError("rotation_angles tidak cocok dengan preprocessing study.")

    training = cfg.get("training", {})
    if int(training.get("epochs", -1)) != 50:
        raise ValueError("Matched W0-HBP harus 50 epoch.")
    if abs(float(training.get("lr", -1.0)) - 3.0e-4) > 1.0e-12:
        raise ValueError("lr harus 3e-4.")
    if abs(float(training.get("weight_decay", -1.0)) - 1.0e-4) > 1.0e-12:
        raise ValueError("weight_decay harus 1e-4.")
    if training.get("classification_loss") != "cross_entropy":
        raise ValueError("Loss harus cross_entropy.")
    if abs(float(training.get("label_smoothing", -1.0)) - 0.1) > 1.0e-12:
        raise ValueError("label_smoothing harus 0.1.")
    if training.get("scheduler") != "cosine":
        raise ValueError("scheduler harus cosine.")
    if float(training.get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")


def validate_arm_preprocessing(preprocessing: dict, arm: str) -> None:
    arm = arm.upper()
    if arm == "R0":
        if str(preprocessing.get("code", "")).upper() != "R0":
            raise ValueError("R0 arm harus memakai preprocessing code R0.")
        if preprocessing.get("method") != "raw":
            raise ValueError("R0 arm harus raw.")
        return

    if arm == "W0":
        if str(preprocessing.get("code", "")).upper() != "W0":
            raise ValueError("W0 arm harus memakai preprocessing code W0.")
        if preprocessing.get("method") != "haar4_visushrink_soft_reconstruction":
            raise ValueError("W0 harus Haar4 + VisuShrink reconstruction.")
        if int(preprocessing.get("wavelet_levels", -1)) != 4:
            raise ValueError("W0 wavelet_levels harus 4.")
        return

    raise ValueError(f"Arm tidak dikenal: {arm}")


def train_hbp_preprocessing_arm(
    cfg: dict,
    *,
    arm: str,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_model_sha256: str,
    resume: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_hbp_preprocessing_config(cfg)
    validate_arm_preprocessing(cfg["preprocessing"], arm)

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")

    model = build_model(cfg["model"]).to(device)
    initial_sha = model_state_fingerprint(model)
    if initial_sha != expected_initial_model_sha256:
        raise RuntimeError(
            f"Initial model {arm} berbeda dari matched initialization: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )

    runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    training_cfg = cfg["training"]
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
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError(f"Checkpoint {arm} berasal dari kontrak berbeda.")
        if checkpoint.get("classes") != loaders.classes:
            raise RuntimeError(f"Urutan kelas checkpoint {arm} berbeda.")
        required = {
            "model", "optimizer", "scheduler", "history",
            "best_f1", "rng_state",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint {arm} tidak lengkap: {missing}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME {arm}-HBP: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        set_study_epoch(loaders.train, epoch)
        model.train()
        running_loss = 0.0
        batches = 0
        progress = tqdm(
            loaders.train,
            desc=f"{arm}-HBP epoch {epoch_number}/{epochs}",
        )

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            model_input = runtime(images)
            output = model(model_input, labels=labels)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics, _, _, _, _ = _evaluate_model(
            model,
            loaders.val,
            runtime,
            device,
            loaders.classes,
            hard_groups,
        )
        record = {
            "epoch": epoch_number,
            "loss": running_loss / max(batches, 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "epoch": epoch_number,
                    "loss": record["loss"],
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
            "arm": arm,
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
                    "arm": arm,
                    "epoch": epoch_number,
                    "best_f1": best_f1,
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
        raise RuntimeError(f"Training {arm}-HBP selesai tanpa best/last checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    eval_cfg = copy.deepcopy(best["config"])
    eval_cfg["model"]["pretrained"] = False
    eval_model = build_model(eval_cfg["model"]).to(device)
    eval_model.load_state_dict(best["model"])

    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        eval_model,
        loaders.val,
        runtime,
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
        "arm": arm,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "initial_model_state_sha256": initial_sha,
    }
