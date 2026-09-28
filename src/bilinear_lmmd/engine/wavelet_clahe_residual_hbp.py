from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from bilinear_lmmd.data.preprocessing.clahe import CLAHEConfig, CLAHEFrontend
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    classification_metrics,
    resolve_device,
)
from bilinear_lmmd.modeling.wavelet_clahe_residual_hbp import (
    WaveletCLAHEContrastResidualHBPModel,
    assert_shared_wr_equal,
    shared_wr_state,
)
from bilinear_lmmd.modeling.wavelet_residual_hbp import WaveletResidualHBPModel


PROTOCOL = "coffee17-wavelet-clahe-residual-hbp-v1"


def validate_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("WRC-HBP dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("Eksperimen harus source_only.")

    data = cfg.get("data", {})
    if data.get("augmentation_mode") != "preprocessing_study":
        raise ValueError("augmentation_mode harus preprocessing_study.")
    if int(data.get("image_size", -1)) != 224:
        raise ValueError("image_size harus 224.")
    if int(data.get("batch_size", -1)) != 32:
        raise ValueError("batch_size harus 32.")
    if list(data.get("rotation_angles", [])) != [0, 45, 90, 135, 180, 225, 270]:
        raise ValueError("rotation schedule tidak cocok.")
    if bool(data.get("object_crop", False)):
        raise ValueError("object_crop harus false.")

    model = cfg.get("model", {})
    if model.get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if model.get("head") != "hbp":
        raise ValueError("Shared core harus HBP.")
    if list(model.get("out_indices", [])) != [1, 3, 4]:
        raise ValueError("HBP out_indices harus [1,3,4].")
    if int(model.get("projection_dim", -1)) != 512:
        raise ValueError("HBP projection_dim harus 512.")
    if model.get("classifier", "linear") != "linear":
        raise ValueError("Classifier harus linear.")
    if int(model.get("num_classes", -1)) != 17:
        raise ValueError("Coffee17 harus 17 kelas.")

    training = cfg.get("training", {})
    if int(training.get("epochs", -1)) != 50:
        raise ValueError("Training harus 50 epoch.")
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

    wavelet = cfg.get("wavelet_residual", {})
    if wavelet.get("domain") != "luminance":
        raise ValueError("Wavelet branch harus luminance.")
    if wavelet.get("level") != "L1":
        raise ValueError("Wavelet branch harus L1.")
    if wavelet.get("bands") != ["LH", "HL", "HH"]:
        raise ValueError("Wavelet bands harus LH/HL/HH.")
    if wavelet.get("threshold") != "visushrink_soft":
        raise ValueError("Wavelet threshold harus VisuShrink soft.")
    if wavelet.get("injection") != "shallow_residual":
        raise ValueError("Wavelet injection harus shallow_residual.")
    if int(wavelet.get("hidden_channels", -1)) != 16:
        raise ValueError("Wavelet hidden_channels harus 16.")
    if wavelet.get("gate") != "tanh_zero_init":
        raise ValueError("Wavelet gate harus tanh_zero_init.")
    if abs(float(wavelet.get("eps", -1.0)) - 1.0e-8) > 1.0e-15:
        raise ValueError("Wavelet eps harus 1e-8.")

    contrast = cfg.get("contrast_residual", {})
    if contrast.get("source") != "C0_clahe_lab_luminance":
        raise ValueError("Contrast source harus C0 CLAHE-L.")
    if abs(float(contrast.get("clip_limit", -1.0)) - 2.0) > 1.0e-12:
        raise ValueError("CLAHE clip_limit harus 2.0.")
    if list(contrast.get("tile_grid_size", [])) != [8, 8]:
        raise ValueError("CLAHE tile_grid_size harus [8,8].")
    if contrast.get("representation") != "rec709_luminance_delta":
        raise ValueError("Contrast representation harus Rec.709 luminance delta.")
    if contrast.get("injection") != "shallow_residual":
        raise ValueError("Contrast injection harus shallow_residual.")
    if int(contrast.get("hidden_channels", -1)) != 8:
        raise ValueError("Contrast hidden_channels harus 8.")
    if contrast.get("gate") != "tanh_zero_init":
        raise ValueError("Contrast gate harus tanh_zero_init.")


def build_wr_control(cfg: dict) -> WaveletResidualHBPModel:
    model = cfg["model"]
    wavelet = cfg["wavelet_residual"]
    return WaveletResidualHBPModel(
        backbone=model["backbone"],
        num_classes=int(model["num_classes"]),
        out_indices=tuple(model["out_indices"]),
        projection_dim=int(model["projection_dim"]),
        dropout=float(model.get("dropout", 0.2)),
        pretrained=bool(model.get("pretrained", True)),
        branch_hidden_channels=int(wavelet["hidden_channels"]),
        wavelet_eps=float(wavelet["eps"]),
    )


def build_wrc_candidate(cfg: dict) -> WaveletCLAHEContrastResidualHBPModel:
    model = cfg["model"]
    wavelet = cfg["wavelet_residual"]
    contrast = cfg["contrast_residual"]
    return WaveletCLAHEContrastResidualHBPModel(
        backbone=model["backbone"],
        num_classes=int(model["num_classes"]),
        out_indices=tuple(model["out_indices"]),
        projection_dim=int(model["projection_dim"]),
        dropout=float(model.get("dropout", 0.2)),
        pretrained=bool(model.get("pretrained", True)),
        wavelet_hidden_channels=int(wavelet["hidden_channels"]),
        contrast_hidden_channels=int(contrast["hidden_channels"]),
        wavelet_eps=float(wavelet["eps"]),
    )


def shared_wr_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(shared_wr_state(model).items()):
        tensor = value.contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_clahe_frontend(cfg: dict) -> CLAHEFrontend:
    contrast = cfg["contrast_residual"]
    frontend = CLAHEFrontend(
        CLAHEConfig(
            clip_limit=float(contrast["clip_limit"]),
            tile_grid_size=tuple(int(v) for v in contrast["tile_grid_size"]),
        )
    )
    frontend.eval()
    return frontend.cpu()


@torch.no_grad()
def clahe_luminance_residual(
    raw_rgb_cpu: torch.Tensor,
    frontend: CLAHEFrontend,
) -> torch.Tensor:
    """Signed luminance difference C0(raw)-raw, returned as Bx1xHxW CPU."""

    if raw_rgb_cpu.device.type != "cpu":
        raw_rgb_cpu = raw_rgb_cpu.detach().cpu()
    enhanced = frontend(raw_rgb_cpu)
    weights = raw_rgb_cpu.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    raw_y = (raw_rgb_cpu * weights).sum(dim=1, keepdim=True)
    enhanced_y = (enhanced * weights).sum(dim=1, keepdim=True)
    return enhanced_y - raw_y


@torch.no_grad()
def preflight_matched_initialization(cfg: dict) -> dict:
    validate_config(cfg)

    seed_everything(42)
    control = build_wr_control(cfg)
    control_sha = shared_wr_fingerprint(control)

    seed_everything(42)
    candidate = build_wrc_candidate(cfg)
    candidate_sha = shared_wr_fingerprint(candidate)

    assert_shared_wr_equal(control, candidate)
    if control_sha != candidate_sha:
        raise RuntimeError("Shared WR-HBP fingerprint berbeda.")

    control.eval()
    candidate.eval()
    frontend = build_clahe_frontend(cfg)

    raw = torch.linspace(
        0.0,
        1.0,
        steps=2 * 3 * 224 * 224,
        dtype=torch.float32,
    ).reshape(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)
    contrast = clahe_luminance_residual(raw, frontend)

    control_logits = control(normalized, raw_rgb=raw).logits
    candidate_logits = candidate(
        normalized,
        raw_rgb=raw,
        contrast_residual=contrast,
    ).logits
    max_abs = float((control_logits - candidate_logits).abs().max().item())
    if max_abs > 1.0e-7:
        raise RuntimeError(
            f"Zero contrast gate tidak ekuivalen dengan WR control: max_abs={max_abs}"
        )
    if float(candidate.contrast_gate_value().abs().item()) != 0.0:
        raise RuntimeError("Contrast gate tidak mulai dari nol.")

    return {
        "shared_wr_core_sha256": control_sha,
        "contrast_gate_zero_initial_value": 0.0,
        "initial_logit_max_abs_difference": max_abs,
        "matched_shared_wr_tensor_equality": True,
    }


def _forward(
    model: nn.Module,
    raw_images_cpu: torch.Tensor,
    device: torch.device,
    *,
    arm: str,
    clahe_frontend: CLAHEFrontend,
    labels: torch.Tensor | None = None,
):
    if raw_images_cpu.device.type != "cpu":
        raw_images_cpu = raw_images_cpu.detach().cpu()

    contrast = None
    if arm == "WRC_HBP":
        contrast = clahe_luminance_residual(raw_images_cpu, clahe_frontend)

    raw = raw_images_cpu.to(device, non_blocking=True)
    normalized = imagenet_normalize(raw)

    if arm == "WR_HBP":
        return model(normalized, raw_rgb=raw, labels=labels)
    if arm == "WRC_HBP":
        return model(
            normalized,
            raw_rgb=raw,
            contrast_residual=contrast.to(device, non_blocking=True),
            labels=labels,
        )
    raise ValueError(f"Arm tidak dikenal: {arm}")


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
    *,
    arm: str,
    clahe_frontend: CLAHEFrontend,
) -> tuple[dict, list[int], list[int], list[list[float]], list[str]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    for images, targets in loader:
        output = _forward(
            model,
            images,
            device,
            arm=arm,
            clahe_frontend=clahe_frontend,
        )
        probs = output.logits.softmax(1).cpu()
        labels.extend(targets.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())

    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah path validation berbeda dari prediksi.")

    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


def train_arm(
    cfg: dict,
    *,
    arm: str,
    run_dir: Path,
    run_contract_sha256: str,
    expected_shared_wr_sha256: str,
    resume: bool,
) -> dict:
    validate_config(cfg)
    if arm not in {"WR_HBP", "WRC_HBP"}:
        raise ValueError("arm harus WR_HBP atau WRC_HBP.")

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")

    model = build_wr_control(cfg) if arm == "WR_HBP" else build_wrc_candidate(cfg)
    shared_sha = shared_wr_fingerprint(model)
    if shared_sha != expected_shared_wr_sha256:
        raise RuntimeError(
            f"Shared WR core {arm} berbeda dari preflight: "
            f"{shared_sha} != {expected_shared_wr_sha256}"
        )
    model = model.to(device)
    clahe_frontend = build_clahe_frontend(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
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
            output = _forward(
                model,
                images,
                device,
                arm=arm,
                clahe_frontend=clahe_frontend,
                labels=labels,
            )
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
            arm=arm,
            clahe_frontend=clahe_frontend,
        )

        record = {
            "epoch": epoch_number,
            "loss": running_loss / max(batches, 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "wavelet_gate": float(model.gate_value().detach().cpu().item()),
        }
        if arm == "WRC_HBP":
            record["contrast_gate"] = float(
                model.contrast_gate_value().detach().cpu().item()
            )
        history.append(record)

        message = {
            "arm": arm,
            "epoch": epoch_number,
            "loss": record["loss"],
            "macro_f1": metrics["macro_f1"],
            "hard_class_f1": metrics["hard_class_f1"],
            "worst_class_f1": metrics["worst_class_f1"],
            "wavelet_gate": record["wavelet_gate"],
        }
        if arm == "WRC_HBP":
            message["contrast_gate"] = record["contrast_gate"]
        print(json.dumps(message), flush=True)

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
            "shared_wr_initial_sha256": expected_shared_wr_sha256,
        }
        atomic_torch_save(checkpoint, last_path)

        if is_best:
            best_payload = {
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "arm": arm,
                "epoch": epoch_number,
                "best_f1": best_f1,
                "run_contract_sha256": run_contract_sha256,
                "shared_wr_initial_sha256": expected_shared_wr_sha256,
                "wavelet_gate": float(model.gate_value().detach().cpu().item()),
                "contrast_gate": (
                    float(model.contrast_gate_value().detach().cpu().item())
                    if arm == "WRC_HBP"
                    else None
                ),
            }
            atomic_torch_save(best_payload, best_path)

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError(f"Training {arm} selesai tanpa best/last checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    eval_model = build_wr_control(cfg) if arm == "WR_HBP" else build_wrc_candidate(cfg)
    eval_model.load_state_dict(best["model"])
    eval_model = eval_model.to(device)

    metrics, labels, predictions, probabilities, paths = _evaluate(
        eval_model,
        loaders.val,
        device,
        loaders.classes,
        hard_groups,
        arm=arm,
        clahe_frontend=clahe_frontend,
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
        "shared_wr_initial_sha256": expected_shared_wr_sha256,
        "wavelet_gate_at_best": float(best["wavelet_gate"]),
        "contrast_gate_at_best": best.get("contrast_gate"),
    }
