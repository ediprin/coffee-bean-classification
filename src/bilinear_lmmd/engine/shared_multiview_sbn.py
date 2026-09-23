from __future__ import annotations

import copy
import json
from contextlib import contextmanager
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
from bilinear_lmmd.engine.shared_multiview import (
    VIEWS,
    validation_identity_label_sha256,
)
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-shared-multiview-sbn-v1"


def validate_sbn_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("SBN v1 dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("SBN v1 harus source_only.")
    if cfg.get("model", {}).get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if cfg.get("model", {}).get("head") != "gap":
        raise ValueError("Head harus GAP.")
    if cfg.get("model", {}).get("out_indices") != [4]:
        raise ValueError("out_indices harus [4].")
    if cfg.get("data", {}).get("augmentation_mode") != "preprocessing_study":
        raise ValueError("Augmentasi harus preprocessing_study.")
    if bool(cfg.get("data", {}).get("object_crop", False)):
        raise ValueError("object_crop harus false.")
    if int(cfg.get("training", {}).get("epochs", -1)) != 50:
        raise ValueError("Eksperimen utama harus 50 epoch.")
    if cfg.get("training", {}).get("classification_loss") != "cross_entropy":
        raise ValueError("Loss harus cross_entropy.")
    if float(cfg.get("training", {}).get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")

    mv = cfg.get("multiview", {})
    if mv.get("method") != "shared_mvce_sbn":
        raise ValueError("multiview.method harus shared_mvce_sbn.")
    if tuple(mv.get("views", ())) != VIEWS:
        raise ValueError(f"views harus tepat {VIEWS}.")
    if mv.get("normalization_strategy") != "selective_batch_stats":
        raise ValueError("normalization_strategy harus selective_batch_stats.")
    if str(mv.get("deployment_bn_source", "")).upper() != "R0":
        raise ValueError("deployment_bn_source harus R0.")
    if abs(
        float(mv.get("raw_weight", -1.0))
        + float(mv.get("auxiliary_weight", -1.0))
        - 1.0
    ) > 1.0e-9:
        raise ValueError("raw_weight + auxiliary_weight harus 1.")
    frontends = mv.get("frontends", {})
    if tuple(frontends) != VIEWS:
        raise ValueError("Frontend harus tepat R0/C0/F0/W0.")
    for arm in VIEWS:
        if str(frontends[arm].get("code", "")).upper() != arm:
            raise ValueError(f"Frontend {arm} memiliki code salah.")
    if str(cfg.get("preprocessing", {}).get("code", "")).upper() != "R0":
        raise ValueError("Evaluation/deployment preprocessing harus R0.")


def build_sbn_runtimes(
    cfg: dict,
    device: torch.device,
) -> dict[str, PreprocessingRuntime]:
    validate_sbn_config(cfg)
    return {
        arm: PreprocessingRuntime.from_config(
            cfg["multiview"]["frontends"][arm],
            device,
        )
        for arm in VIEWS
    }


@contextmanager
def auxiliary_batchnorm_batch_stats(model: nn.Module):
    """Use each auxiliary mini-batch's own BN statistics without persisting them.

    This is the selective-normalization contract used for transformed views:
    R0 forwards run normally and are the only forwards that update deployment
    running_mean/running_var. For C0/F0/W0, BatchNorm remains in training mode
    so normalization uses that view's current mini-batch mean/variance, while
    track_running_stats=False prevents those auxiliary statistics from changing
    the deployment buffers. Affine gamma/beta remain shared and trainable.
    """
    states: list[tuple[nn.modules.batchnorm._BatchNorm, bool, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            states.append(
                (module, bool(module.training), bool(module.track_running_stats))
            )
            module.train(True)
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, training, track_running_stats in states:
            module.track_running_stats = track_running_stats
            module.train(training)


def train_shared_multiview_sbn(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_model_sha256: str,
    resume: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_sbn_config(cfg)

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
            "Initial model berbeda dari reference: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )

    runtimes = build_sbn_runtimes(cfg, device)
    eval_runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    training_cfg = cfg["training"]
    mv_cfg = cfg["multiview"]
    raw_weight = float(mv_cfg["raw_weight"])
    auxiliary_weight = float(mv_cfg["auxiliary_weight"])

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
        print(f"RESUME MVCE-SBN: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        model.train()
        totals = {
            "loss": 0.0,
            "raw_ce": 0.0,
            "c0_ce": 0.0,
            "f0_ce": 0.0,
            "w0_ce": 0.0,
            "aux_ce": 0.0,
        }
        progress = tqdm(loaders.train, desc=f"MVCE-SBN epoch {epoch + 1}/{epochs}")

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Deployment view: standard BN training behavior and running-stat update.
            raw_input = runtimes["R0"](images)
            raw_logits = model(raw_input, labels=labels).logits
            raw_ce = loss_fn(raw_logits, labels)

            # Auxiliary views: own current-batch statistics, no running-stat update.
            auxiliary_losses = []
            auxiliary_by_arm = {}
            with auxiliary_batchnorm_batch_stats(model):
                for arm in ("C0", "F0", "W0"):
                    view_input = runtimes[arm](images)
                    logits = model(view_input, labels=labels).logits
                    ce = loss_fn(logits, labels)
                    auxiliary_losses.append(ce)
                    auxiliary_by_arm[arm] = ce

            aux_ce = torch.stack(auxiliary_losses).mean()
            loss = raw_weight * raw_ce + auxiliary_weight * aux_ce
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["raw_ce"] += float(raw_ce.item())
            totals["c0_ce"] += float(auxiliary_by_arm["C0"].item())
            totals["f0_ce"] += float(auxiliary_by_arm["F0"].item())
            totals["w0_ce"] += float(auxiliary_by_arm["W0"].item())
            totals["aux_ce"] += float(aux_ce.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                raw=f"{raw_ce.item():.4f}",
                f0=f"{auxiliary_by_arm['F0'].item():.4f}",
            )

        scheduler.step()
        metrics, _, _, _, _ = _evaluate_model(
            model,
            loaders.val,
            eval_runtime,
            device,
            loaders.classes,
            hard_groups,
        )
        batches = max(len(loaders.train), 1)
        record = {
            "epoch": epoch + 1,
            **{key: value / batches for key, value in totals.items()},
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": record["loss"],
                    "raw_ce": record["raw_ce"],
                    "c0_ce": record["c0_ce"],
                    "f0_ce": record["f0_ce"],
                    "w0_ce": record["w0_ce"],
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
            "epoch": epoch + 1,
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
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "weights": "raw",
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
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "initial_model_state_sha256": initial_sha,
    }
