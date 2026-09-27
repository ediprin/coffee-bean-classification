from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    current_git_commit,
    model_state_fingerprint,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.engine.at_sbn import AuxiliaryTrainingModel
from bilinear_lmmd.engine.hbp_mvfd_sbn import (
    PROTOCOL,
    train_hbp_mvfd_sbn,
    train_hbp_r0_control,
    validate_hbp_mvfd_sbn_config,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.modeling.models import build_model


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "worst_class_f1",
    "hard_class_f1",
)


def _json(path: Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return False
    import torch
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def _write_contract(path: Path, contract: dict) -> None:
    if path.is_file():
        if _json(path, "Existing contract") != contract:
            raise RuntimeError(f"Existing run contract berbeda: {path}")
        return
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")


def run_hbp_mvfd_sbn(
    *,
    config_path: Path,
    reference_path: Path,
    fold: int,
    data_root: Path,
    output_root: Path,
    required_commit: str,
    device: str = "auto",
    resume: bool = False,
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari frozen commit: {actual_commit} != {required_commit}"
        )

    cfg = copy.deepcopy(load_config(config_path))
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    validate_hbp_mvfd_sbn_config(cfg)

    reference = _json(reference_path, "Matched preprocessing-fold reference")
    if reference.get("format") != "coffee17.shared_multiview.sbn_reference.v1":
        raise RuntimeError("Reference format tidak dikenal.")
    expected = reference["folds"][f"fold_{fold}"]

    count, val_sha = validation_identity_label_sha256(data_root)
    if count != int(expected["count"]):
        raise RuntimeError(f"Validation count berbeda: {count} != {expected['count']}")
    if val_sha != expected["identity_label_sha256"]:
        raise RuntimeError("Validation identity hash berbeda dari reference.")

    # The exact HBP initialization is generated once from the locked config/seed.
    # Both the R0-only control and the MVFD wrapper must match it.
    seed_everything(42)
    probe = build_model(copy.deepcopy(cfg["model"]))
    hbp_initial = model_state_fingerprint(probe)
    del probe

    seed_everything(42)
    wrapped_probe = AuxiliaryTrainingModel(copy.deepcopy(cfg["model"]))
    if wrapped_probe.primary_initial_sha256 != hbp_initial:
        raise RuntimeError("MVFD wrapper mengubah initial HBP primary model.")
    del wrapped_probe

    output_root = Path(output_root).expanduser().resolve()
    control_dir = output_root / "HBP_R0_CONTROL" / f"fold_{fold}" / "seed42"
    mvfd_dir = output_root / "HBP_MVFD_SBN_ALL4" / f"fold_{fold}" / "seed42"
    control_dir.mkdir(parents=True, exist_ok=True)
    mvfd_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "git_commit": actual_commit,
        "reference_sha256": sha256_file(reference_path),
        "validation_identity_label_sha256": val_sha,
        "validation_count": count,
        "hbp_initial_model_state_sha256": hbp_initial,
        "model": {
            "backbone": cfg["model"]["backbone"],
            "head": cfg["model"]["head"],
            "out_indices": list(cfg["model"]["out_indices"]),
            "projection_dim": int(cfg["model"]["projection_dim"]),
            "classifier": cfg["model"]["classifier"],
            "num_classes": int(cfg["model"]["num_classes"]),
        },
        "epochs": int(cfg["training"]["epochs"]),
        "evaluation_split_during_training": "val_R0_primary_only",
        "test_images_accessed": False,
    }

    control_cfg = copy.deepcopy(cfg)
    control_cfg["training"]["output_dir"] = str(control_dir)
    control_contract = {
        "format": "bilinear_lmmd.hbp_mvfd_sbn.control_contract.v1",
        **common,
        "method": "HBP_R0_CONTROL",
        "training_views": ["R0"],
        "deployment_view": "R0",
        "objective": {"primary_ce_weight": 1.0},
        "resolved_config_sha256": canonical_json_sha256(control_cfg),
    }
    control_sha = canonical_json_sha256(control_contract)
    control_contract["run_contract_sha256"] = control_sha
    _write_contract(control_dir / "run_contract.json", control_contract)
    if not (control_dir / "run_config.yaml").is_file():
        (control_dir / "run_config.yaml").write_text(
            yaml.safe_dump(control_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    fd = cfg["feature_distillation"]
    mvfd_cfg = copy.deepcopy(cfg)
    mvfd_cfg["training"]["output_dir"] = str(mvfd_dir)
    mvfd_contract = {
        "format": "bilinear_lmmd.hbp_mvfd_sbn.run_contract.v1",
        **common,
        "method": "HBP_MVFD_SBN_ALL4",
        "training_views": ["R0", "C0", "F0", "W0"],
        "teacher_views": ["C0", "F0", "W0"],
        "deployment_view": "R0",
        "shared_feature_extractor_and_hbp_head": True,
        "training_only_auxiliary_classifiers": ["C0", "F0", "W0"],
        "selective_batch_norm": True,
        "teacher_aggregation": "equal_mean_auxiliary_hbp_embedding",
        "teacher_stop_gradient": True,
        "objective": {
            "primary_ce_weight": 1.0,
            "aux_ce_weight_each": float(fd["aux_ce_weight_each"]),
            "feature_distill_weight": float(fd["feature_distill_weight"]),
            "feature_distill_loss": "mean squared-L2 norm per sample",
            "transfer_direction": "C0_F0_W0_detached_mean_HBP_embedding_to_R0_HBP_embedding",
        },
        "extra_inference_forward_passes": 0,
        "resolved_config_sha256": canonical_json_sha256(mvfd_cfg),
    }
    mvfd_sha = canonical_json_sha256(mvfd_contract)
    mvfd_contract["run_contract_sha256"] = mvfd_sha
    _write_contract(mvfd_dir / "run_contract.json", mvfd_contract)
    if not (mvfd_dir / "run_config.yaml").is_file():
        (mvfd_dir / "run_config.yaml").write_text(
            yaml.safe_dump(mvfd_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    epochs = int(cfg["training"]["epochs"])
    with exclusive_training_lock(
        output_root,
        lock_name=f"HBP_MVFD_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        if not _run_complete(control_dir, epochs):
            train_hbp_r0_control(
                control_cfg,
                run_dir=control_dir,
                run_contract_sha256=control_sha,
                expected_primary_initial_sha256=hbp_initial,
                resume=resume or (control_dir / "last.pt").is_file(),
            )
        if not _run_complete(mvfd_dir, epochs):
            train_hbp_mvfd_sbn(
                mvfd_cfg,
                run_dir=mvfd_dir,
                run_contract_sha256=mvfd_sha,
                expected_primary_initial_sha256=hbp_initial,
                resume=resume or (mvfd_dir / "last.pt").is_file(),
            )

    if not _run_complete(control_dir, epochs) or not _run_complete(mvfd_dir, epochs):
        raise RuntimeError("Matched HBP experiment belum selesai.")

    control_metrics = _json(
        control_dir / "validation" / "metrics.json",
        "HBP control validation metrics",
    )
    mvfd_metrics = _json(
        mvfd_dir / "validation" / "metrics.json",
        "HBP-MVFD validation metrics",
    )
    history = _json(mvfd_dir / "history.json", "HBP-MVFD history")
    best_record = max(history, key=lambda row: float(row["source"]["macro_f1"]))
    frozen_gap = expected["r0_control"]

    result = {
        "format": "bilinear_lmmd.hbp_mvfd_sbn.result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "HBP_R0_CONTROL": {key: control_metrics[key] for key in METRICS},
        "HBP_MVFD_SBN_ALL4": {key: mvfd_metrics[key] for key in METRICS},
        "FROZEN_GAP_R0_CONTROL": {key: frozen_gap[key] for key in METRICS},
        "DELTA_HBP_VS_GAP": {
            key: control_metrics[key] - frozen_gap[key] for key in METRICS
        },
        "DELTA_MVFD_ON_HBP": {
            key: mvfd_metrics[key] - control_metrics[key] for key in METRICS
        },
        "DELTA_COMBINED_VS_GAP": {
            key: mvfd_metrics[key] - frozen_gap[key] for key in METRICS
        },
        "best_epoch_diagnostics": {
            key: best_record[key]
            for key in (
                "epoch",
                "primary_ce",
                "c0_ce",
                "f0_ce",
                "w0_ce",
                "feature_loss",
                "weighted_feature_loss",
                "teacher_norm",
                "r0_norm",
                "r0_teacher_cos",
                "r0_teacher_l2",
                "c0_disagreement",
                "f0_disagreement",
                "w0_disagreement",
            )
        },
        "hbp_initial_model_state_sha256": hbp_initial,
        "control_contract": control_contract,
        "mvfd_contract": mvfd_contract,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = mvfd_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_hbp_mvfd_sbn(
        config_path=args.config,
        reference_path=args.reference,
        fold=args.fold,
        data_root=args.data_root,
        output_root=args.output_root,
        required_commit=args.required_commit,
        device=args.device,
        resume=args.resume,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
