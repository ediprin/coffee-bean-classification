from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bilinear_lmmd.analysis.coffee17_physical_descriptors import (
    extract_physical_descriptors,
)
from bilinear_lmmd.core.reproducibility import seed_everything
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_evaluation_loader,
)
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.train import classification_metrics, resolve_device
from bilinear_lmmd.engine.wavelet_residual_hbp import (
    build_candidate,
    shared_core_fingerprint,
    validate_config as validate_wr_config,
)
from bilinear_lmmd.modeling.physical_logit_residual import (
    FEATURES,
    PhysicalResidualFit,
    fit_physical_logit_residual,
)


PROTOCOL = "coffee17-physical-logit-residual-wr-hbp-v1"


def configure_strict_determinism(seed: int) -> dict:
    """Enable strict deterministic PyTorch execution for the WR base run."""

    required_workspace = ":4096:8"
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, required_workspace):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG berbeda dari frozen setting: "
            f"{current!r} != {required_workspace!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_workspace

    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    return {
        "seed": int(seed),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
    }


def validate_config(cfg: dict) -> None:
    validate_wr_config(cfg)

    residual = cfg.get("physical_logit_residual", {})
    if list(residual.get("features", [])) != list(FEATURES):
        raise ValueError("Frozen physical residual feature list berubah.")
    if residual.get("fusion") != "masked_additive_logit_residual":
        raise ValueError("Fusion harus masked_additive_logit_residual.")
    if residual.get("base_model") != "WR_HBP":
        raise ValueError("Base model harus WR_HBP.")
    if residual.get("base_model_frozen_during_residual_fit") is not True:
        raise ValueError("WR-HBP harus frozen saat residual fit.")
    if abs(float(residual.get("l2", -1.0)) - 0.01) > 1.0e-12:
        raise ValueError("Residual L2 harus 0.01.")
    if abs(float(residual.get("label_smoothing", -1.0)) - 0.1) > 1.0e-12:
        raise ValueError("Residual label smoothing harus 0.1.")
    if int(residual.get("maxiter", -1)) != 500:
        raise ValueError("Residual maxiter harus 500.")
    if residual.get("fit_device") != "cpu_float64":
        raise ValueError("Residual fit harus cpu_float64.")
    if residual.get("insect_topology_enabled") is not False:
        raise ValueError("Insect topology harus tetap disabled.")


def _paths_from_loader(loader) -> list[str]:
    return [sample[0] for sample in loader.dataset.samples]


def _extract_feature_matrix(paths: list[str]) -> np.ndarray:
    rows = []
    for index, path in enumerate(paths, start=1):
        desc = extract_physical_descriptors(Path(path))
        if not bool(desc["mask_qc_pass"]):
            raise RuntimeError(
                "Physical descriptor mask QC gagal pada model experiment: "
                f"{path}. Audit harus menyelesaikan QC ini sebelum training fusion."
            )
        rows.append([float(desc[name]) for name in FEATURES])
        if index % 200 == 0:
            print(f"physical descriptors: {index}/{len(paths)}", flush=True)
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Physical descriptor matrix memiliki nilai non-finite.")
    return matrix


@torch.no_grad()
def _base_logits(
    model,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[int] = []
    for images, targets in loader:
        raw = images.to(device, non_blocking=True)
        normalized = imagenet_normalize(raw)
        output = model(normalized, raw_rgb=raw)
        logits.append(output.logits.detach().cpu().numpy())
        labels.extend(targets.tolist())

    paths = _paths_from_loader(loader)
    matrix = np.concatenate(logits, axis=0)
    targets = np.asarray(labels, dtype=np.int64)
    if len(paths) != len(targets) or len(matrix) != len(targets):
        raise RuntimeError("Jumlah base logits/labels/paths tidak cocok.")
    return matrix, targets, paths


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    hard_groups: dict,
) -> tuple[dict, list[int], list[list[float]]]:
    probs = _softmax_numpy(logits)
    predictions = probs.argmax(axis=1).astype(int).tolist()
    metrics = classification_metrics(
        labels.astype(int).tolist(),
        predictions,
        classes,
        hard_groups,
    )
    return metrics, predictions, probs.tolist()


def _serialize_fit(fit: PhysicalResidualFit) -> dict:
    matrix = fit.full_weight_matrix()
    active = []
    for value, (ci, fi) in zip(fit.weights, fit.active_pairs):
        active.append({
            "class": fit.classes[ci],
            "feature": fit.features[fi],
            "weight": float(value),
        })

    return {
        "classes": list(fit.classes),
        "features": list(fit.features),
        "active_parameter_count": len(fit.active_pairs),
        "active_weights": active,
        "full_weight_matrix": matrix.tolist(),
        "standardizer_mean": fit.standardizer.mean.tolist(),
        "standardizer_scale": fit.standardizer.scale.tolist(),
        "l2": fit.l2,
        "label_smoothing": fit.label_smoothing,
        "optimizer_success": fit.optimizer_success,
        "optimizer_status": fit.optimizer_status,
        "optimizer_message": fit.optimizer_message,
        "optimizer_iterations": fit.optimizer_iterations,
        "initial_objective": fit.initial_objective,
        "final_objective": fit.final_objective,
    }


def fit_and_evaluate_residual(
    *,
    cfg: dict,
    base_checkpoint: Path,
    output_dir: Path,
) -> dict:
    """Fit residual on source/train and evaluate paired WR vs residual on source/val."""

    validate_config(cfg)
    seed = int(cfg["seed"])
    determinism = configure_strict_determinism(seed)
    device = resolve_device(str(cfg["device"]))

    checkpoint = torch.load(
        Path(base_checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    model = build_candidate(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    train_loader, train_classes = build_preprocessing_evaluation_loader(
        cfg["data"],
        split=str(cfg["data"].get("train_split", "train")),
        seed=seed,
    )
    val_loader, val_classes = build_preprocessing_evaluation_loader(
        cfg["data"],
        split=str(cfg["data"].get("val_split", "val")),
        seed=seed,
    )
    if train_classes != val_classes:
        raise RuntimeError("Class order train/val berbeda.")
    classes = list(train_classes)
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    train_logits, train_labels, train_paths = _base_logits(model, train_loader, device)
    val_logits, val_labels, val_paths = _base_logits(model, val_loader, device)

    train_features = _extract_feature_matrix(train_paths)
    val_features = _extract_feature_matrix(val_paths)

    residual_cfg = cfg["physical_logit_residual"]
    fit = fit_physical_logit_residual(
        base_logits=train_logits,
        features=train_features,
        labels=train_labels,
        classes=classes,
        l2=float(residual_cfg["l2"]),
        label_smoothing=float(residual_cfg["label_smoothing"]),
        maxiter=int(residual_cfg["maxiter"]),
    )
    if not fit.optimizer_success:
        raise RuntimeError(
            "Physical residual L-BFGS-B tidak konvergen: "
            f"{fit.optimizer_status} {fit.optimizer_message}"
        )

    zero_adjusted = val_logits + np.zeros_like(val_logits)
    initial_max_abs = float(np.max(np.abs(zero_adjusted - val_logits)))
    if initial_max_abs != 0.0:
        raise RuntimeError("Zero-init residual tidak identik dengan WR logits.")

    candidate_val_logits = fit.adjusted_logits(val_logits, val_features)

    wr_metrics, wr_predictions, wr_probs = _metrics_from_logits(
        val_logits,
        val_labels,
        classes,
        hard_groups,
    )
    candidate_metrics, candidate_predictions, candidate_probs = _metrics_from_logits(
        candidate_val_logits,
        val_labels,
        classes,
        hard_groups,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_evaluation(
        output_dir / "WR_HBP_validation",
        metrics=wr_metrics,
        labels=val_labels.tolist(),
        predictions=wr_predictions,
        probabilities=wr_probs,
        paths=val_paths,
        classes=classes,
        checkpoint=Path(base_checkpoint),
        split="val",
    )
    _write_evaluation(
        output_dir / "WR_PDR_HBP_validation",
        metrics=candidate_metrics,
        labels=val_labels.tolist(),
        predictions=candidate_predictions,
        probabilities=candidate_probs,
        paths=val_paths,
        classes=classes,
        checkpoint=Path(base_checkpoint),
        split="val",
    )

    fit_payload = _serialize_fit(fit)
    (output_dir / "physical_residual_fit.json").write_text(
        json.dumps(fit_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    payload = {
        "format": "bilinear_lmmd.physical_logit_residual_wr_hbp.fold_evaluation.v1",
        "protocol": PROTOCOL,
        "classes": classes,
        "train_count": int(len(train_labels)),
        "val_count": int(len(val_labels)),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_checkpoint_epoch": int(checkpoint["epoch"]),
        "base_shared_core_initial_sha256": checkpoint.get(
            "shared_core_initial_sha256"
        ),
        "base_wavelet_gate": float(checkpoint.get("wavelet_gate", 0.0)),
        "zero_residual_initial_logit_max_abs_difference": initial_max_abs,
        "WR_HBP": wr_metrics,
        "WR_PDR_HBP": candidate_metrics,
        "DELTA_WR_PDR_MINUS_WR": {
            key: (
                float(candidate_metrics[key]) - float(wr_metrics[key])
                if candidate_metrics[key] is not None and wr_metrics[key] is not None
                else None
            )
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "hard_class_f1",
                "worst_class_f1",
            )
        },
        "residual_fit": fit_payload,
        "determinism": determinism,
        "base_model_frozen_during_residual_fit": True,
        "descriptor_extraction_uses_labels": False,
        "insect_topology_enabled": False,
        "outer_test_accessed": False,
    }
    (output_dir / "pair_result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
