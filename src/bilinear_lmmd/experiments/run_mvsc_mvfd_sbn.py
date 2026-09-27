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
from bilinear_lmmd.engine.mvsc_mvfd_sbn import (
    PROTOCOL,
    train_mvsc_mvfd_sbn,
    validate_mvsc_mvfd_sbn_config,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.modeling.models import build_model


METHOD = "MVSC_MVFD_SBN_ALL4"
METRIC_KEYS = (
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


def run_mvsc_mvfd_sbn(
    *,
    config_path: Path,
    r0_reference_path: Path,
    mvfd_reference_path: Path,
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
    validate_mvsc_mvfd_sbn_config(cfg)

    r0_reference = _json(r0_reference_path, "Matched R0 reference")
    mvfd_reference = _json(mvfd_reference_path, "Frozen MVFD reference")
    if r0_reference.get("format") != "coffee17.shared_multiview.sbn_reference.v1":
        raise RuntimeError("R0 reference format tidak dikenal.")
    if mvfd_reference.get("format") != "coffee17.mvfd_sbn.result_reference.v1":
        raise RuntimeError("MVFD reference format tidak dikenal.")

    expected_r0 = r0_reference["folds"][f"fold_{fold}"]
    expected_mvfd = mvfd_reference["folds"][f"fold_{fold}"]

    count, val_sha = validation_identity_label_sha256(data_root)
    if count != int(expected_r0["count"]) or count != int(expected_mvfd["count"]):
        raise RuntimeError("Validation count berbeda dari frozen references.")
    if (
        val_sha != expected_r0["identity_label_sha256"]
        or val_sha != expected_mvfd["identity_label_sha256"]
    ):
        raise RuntimeError("Validation identity hash berbeda dari frozen references.")

    seed_everything(42)
    probe = build_model(copy.deepcopy(cfg["model"]))
    current_initial = model_state_fingerprint(probe)
    del probe
    expected_initial = r0_reference["expected_initial_model_state_sha256"]
    if current_initial != expected_initial:
        raise RuntimeError(
            "Mengekspos intermediate stage mengubah primary initialization: "
            f"{current_initial} != {expected_initial}. "
            "Eksperimen dihentikan karena kontrol tidak lagi matched."
        )

    seed_everything(42)
    wrapped_probe = AuxiliaryTrainingModel(copy.deepcopy(cfg["model"]))
    if wrapped_probe.primary_initial_sha256 != expected_initial:
        raise RuntimeError("ML-MVFD wrapper mengubah primary initialization.")
    del wrapped_probe

    run_dir = (
        Path(output_root).expanduser().resolve()
        / METHOD
        / f"fold_{fold}"
        / "seed42"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["output_dir"] = str(run_dir)

    fd = cfg["feature_distillation"]
    contract = {
        "format": "bilinear_lmmd.mvsc_mvfd_sbn.run_contract.v1",
        "protocol": PROTOCOL,
        "scientific_scope": (
            "post-primary exploratory extension of frozen MVFD-SBN"
        ),
        "fold": int(fold),
        "seed": 42,
        "git_commit": actual_commit,
        "r0_reference_sha256": sha256_file(r0_reference_path),
        "mvfd_reference_sha256": sha256_file(mvfd_reference_path),
        "validation_identity_label_sha256": val_sha,
        "validation_count": count,
        "primary_initial_model_state_sha256": current_initial,
        "training_views": ["R0", "C0", "F0", "W0"],
        "teacher_views": ["C0", "F0", "W0"],
        "deployment_view": "R0",
        "shared_feature_extractor": True,
        "training_only_auxiliary_classifiers": ["C0", "F0", "W0"],
        "selective_batch_norm": True,
        "teacher_aggregation": "equal_mean_at_final_level_only",
        "teacher_stop_gradient": True,
        "objective": {
            "primary_ce_weight": 1.0,
            "aux_ce_weight_each": float(fd["aux_ce_weight_each"]),
            "final_feature_distill_weight": float(fd["feature_distill_weight"]),
            "final_feature_loss": "mean squared-L2 norm per sample",
            "mid_contrastive_weight": float(fd["mid_contrastive_weight"]),
            "mid_contrastive_loss": str(fd["mid_contrastive_loss"]),
            "mid_contrastive_temperature": float(fd["mid_contrastive_temperature"]),
            "mid_contrastive_anchor": str(fd["mid_contrastive_anchor"]),
            "mid_contrastive_keys": list(fd["mid_contrastive_keys"]),
            "mid_backbone_out_index": int(fd["mid_backbone_out_index"]),
            "transfer_direction": (
                "detached_C0_F0_W0_mid_keys_to_R0_anchor_by_class_relations"
            ),
        },
        "extra_inference_parameters": 0,
        "extra_inference_forward_passes": 0,
        "epochs": int(cfg["training"]["epochs"]),
        "evaluation_split_during_training": "val_R0_primary_only",
        "test_images_accessed": False,
        "resolved_config_sha256": canonical_json_sha256(cfg),
    }
    contract_sha = canonical_json_sha256(contract)
    contract["run_contract_sha256"] = contract_sha

    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        if _json(contract_path, "Existing contract") != contract:
            raise RuntimeError("Existing MVSC-MVFD-SBN run contract berbeda.")
    else:
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    result_path = run_dir / "result.json"
    if result_path.is_file():
        old = _json(result_path, "Existing result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing result berasal dari kontrak berbeda.")
        return old

    with exclusive_training_lock(
        output_root,
        lock_name=f"{METHOD}_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        training_executed = False
        if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
            train_mvsc_mvfd_sbn(
                cfg,
                run_dir=run_dir,
                run_contract_sha256=contract_sha,
                expected_primary_initial_sha256=current_initial,
                resume=resume or (run_dir / "last.pt").is_file(),
            )
            training_executed = True

    if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
        raise RuntimeError("Run MVSC-MVFD-SBN belum selesai.")

    metrics = _json(run_dir / "validation" / "metrics.json", "Validation metrics")
    auxiliary_metrics = _json(
        run_dir / "validation" / "auxiliary_metrics.json",
        "Auxiliary validation metrics",
    )
    history = _json(run_dir / "history.json", "History")
    best_record = max(history, key=lambda row: float(row["source"]["macro_f1"]))

    baseline = expected_r0["r0_control"]
    frozen_mvfd = expected_mvfd["metrics"]
    diagnostic_keys = (
        "epoch",
        "primary_ce",
        "c0_ce",
        "f0_ce",
        "w0_ce",
        "feature_loss",
        "weighted_feature_loss",
        "mid_contrastive_loss",
        "weighted_mid_contrastive_loss",
        "mid_positive_keys_mean",
        "mid_negative_keys_mean",
        "mid_positive_cosine",
        "mid_negative_cosine",
        "mid_cosine_gap",
        "r0_teacher_cos",
        "r0_teacher_l2",
        "r0_mid_teacher_cos",
        "r0_mid_teacher_l2",
        "c0_disagreement",
        "f0_disagreement",
        "w0_disagreement",
    )

    result = {
        "format": "bilinear_lmmd.mvsc_mvfd_sbn.result.v1",
        "protocol": PROTOCOL,
        "method": METHOD,
        "fold": int(fold),
        "seed": 42,
        "metrics": {key: metrics[key] for key in METRIC_KEYS},
        "frozen_matched_r0_control": baseline,
        "frozen_mvfd_sbn": frozen_mvfd,
        "delta_vs_r0_control": {
            key: metrics[key] - baseline[key] for key in METRIC_KEYS
        },
        "delta_vs_frozen_mvfd": {
            key: metrics[key] - frozen_mvfd[key] for key in METRIC_KEYS
        },
        "best_epoch_diagnostics": {
            key: best_record[key] for key in diagnostic_keys
        },
        "auxiliary_validation_metrics": {
            arm: {
                key: auxiliary_metrics[arm][key] for key in METRIC_KEYS
            }
            for arm in ("C0", "F0", "W0")
        },
        "best_checkpoint": str(run_dir / "best.pt"),
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--r0-reference", required=True, type=Path)
    parser.add_argument("--mvfd-reference", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_mvsc_mvfd_sbn(
        config_path=args.config,
        r0_reference_path=args.r0_reference,
        mvfd_reference_path=args.mvfd_reference,
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
