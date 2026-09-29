from __future__ import annotations

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
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.gradient_boosting_ce import GradientBoostingCrossEntropy
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    classification_metrics,
    resolve_device,
)
from bilinear_lmmd.engine.wavelet_residual_hbp import (
    build_candidate,
    validate_config as validate_wr_config,
)


PROTOCOL = "coffee17-wr-hbp-gce-v1"
ARMS = ("WR_HBP_CE", "WR_HBP_GCE")


def validate_config(cfg: dict) -> None:
    validate_wr_config(cfg)

    gce = cfg.get("gce", {})
    if gce.get("implementation") != "restricted_topk_negatives":
        raise ValueError("gce.implementation harus restricted_topk_negatives.")
    if int(gce.get("top_k", -1)) != 2:
        raise ValueError("Coffee17 GCE V1 dikunci top_k=2.")
    if gce.get("negative_selection") != "per_sample_highest_logits":
        raise ValueError("GCE harus memilih hard negatives dari logit per-sample.")
    if gce.get("validation_tuning") is not False:
        raise ValueError("GCE V1 tidak mengizinkan validation tuning.")
    if abs(
        float(gce.get("label_smoothing", -1.0))
        - float(cfg["training"]["label_smoothing"])
    ) > 1.0e-12:
        raise ValueError("GCE dan CE harus memakai label smoothing yang sama.")


@torch.no_grad()
def preflight_matched_initialization(cfg: dict) -> dict:
    validate_config(cfg)

    seed_everything(int(cfg["seed"]))
    control = build_candidate(cfg)
    control_sha = model_state_fingerprint(control)

    seed_everything(int(cfg["seed"]))
    candidate = build_candidate(cfg)
    candidate_sha = model_state_fingerprint(candidate)

    if control_sha != candidate_sha:
        raise RuntimeError("Initial WR-HBP state CE/GCE berbeda.")

    left = control.state_dict()
    right = candidate.state_dict()
    if left.keys() != right.keys():
        raise RuntimeError("State keys CE/GCE berbeda.")
    mismatched = [
        key for key in left
        if not torch.equal(left[key].detach().cpu(), right[key].detach().cpu())
    ]
    if mismatched:
        raise RuntimeError(
            "Initial WR-HBP tensors CE/GCE berbeda: " + ", ".join(mismatched[:10])
        )

    control.eval()
    candidate.eval()
    raw = torch.linspace(
        0.0,
        1.0,
        steps=2 * 3 * 224 * 224,
        dtype=torch.float32,
    ).reshape(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)
    left_logits = control(normalized, raw_rgb=raw).logits
    right_logits = candidate(normalized, raw_rgb=raw).logits
    max_abs = float((left_logits - right_logits).abs().max().item())
    if max_abs > 1.0e-7:
        raise RuntimeError(f"Initial CE/GCE logits berbeda: {max_abs}")

    return {
        "initial_model_state_sha256": control_sha,
        "matched_model_tensor_equality": True,
        "initial_logit_max_abs_difference": max_abs,
    }


def _forward(model: nn.Module, raw_images: torch.Tensor, device: torch.device):
    raw = raw_images.to(device, non_blocking=True)
    normalized = imagenet_normalize(raw)
    return model(normalized, raw_rgb=raw)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
) -> tuple[dict, list[int], list[int], list[list[float]], list[str]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    for images, targets in loader:
        output = _forward(model, images, device)
        probs = output.logits.softmax(1).cpu()
        labels.extend(targets.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())

    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah path validation berbeda dari prediksi.")

    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


def _loss_for_arm(cfg: dict, arm: str) -> nn.Module:
    if arm == "WR_HBP_CE":
        return nn.CrossEntropyLoss(
            label_smoothing=float(cfg["training"]["label_smoothing"])
        )
    if arm == "WR_HBP_GCE":
        return GradientBoostingCrossEntropy(
            num_classes=int(cfg["model"]["num_classes"]),
            top_k=int(cfg["gce"]["top_k"]),
            label_smoothing=float(cfg["gce"]["label_smoothing"]),
        )
    raise ValueError(f"Arm tidak dikenal: {arm}")


def train_arm(
    cfg: dict,
    *,
    arm: str,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_model_sha256: str,
    resume: bool,
) -> dict:
    validate_config(cfg)
    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}.")

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")

    model = build_candidate(cfg)
    initial_sha = model_state_fingerprint(model)
    if initial_sha != expected_initial_model_sha256:
        raise RuntimeError(
            f"Initial model {arm} berbeda dari preflight: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = _loss_for_arm(cfg, arm)
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
        print(f"RESUME {arm}: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        set_study_epoch(loaders.train, epoch)
        model.train()
        running_loss = 0.0
        batches = 0
        progress = tqdm(loaders.train, desc=f"{arm} epoch {epoch_number}/{epochs}")

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, images, device)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics, _, _, _, _ = _evaluate(
            model,
            loaders.val,
            device,
            loaders.classes,
            hard_groups,
        )
        record = {
            "epoch": epoch_number,
            "loss": running_loss / max(batches, 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "wavelet_gate": float(model.gate_value().detach().cpu().item()),
            "wavelet_gate_parameter": float(
                model.wavelet_gate.detach().cpu().item()
            ),
        }
        history.append(record)

        print(json.dumps({
            "arm": arm,
            "epoch": epoch_number,
            "loss": record["loss"],
            "macro_f1": metrics["macro_f1"],
            "hard_class_f1": metrics["hard_class_f1"],
            "worst_class_f1": metrics["worst_class_f1"],
            "wavelet_gate": record["wavelet_gate"],
        }), flush=True)

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
            "initial_model_state_sha256": expected_initial_model_sha256,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save({
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "arm": arm,
                "epoch": epoch_number,
                "best_f1": best_f1,
                "run_contract_sha256": run_contract_sha256,
                "initial_model_state_sha256": expected_initial_model_sha256,
                "wavelet_gate": float(model.gate_value().detach().cpu().item()),
            }, best_path)

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError(f"Training {arm} selesai tanpa best/last checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    eval_model = build_candidate(cfg)
    eval_model.load_state_dict(best["model"])
    eval_model = eval_model.to(device)

    metrics, labels, predictions, probabilities, paths = _evaluate(
        eval_model,
        loaders.val,
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
        "initial_model_state_sha256": expected_initial_model_sha256,
        "wavelet_gate_at_best": float(best["wavelet_gate"]),
    }
