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
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.shared_multiview_sbn import (
    PROTOCOL,
    train_shared_multiview_sbn,
    validate_sbn_config,
)
from bilinear_lmmd.modeling.models import build_model


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


def run_sbn(
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
    validate_sbn_config(cfg)

    reference = _json(reference_path, "SBN reference")
    if reference.get("format") != "coffee17.shared_multiview.sbn_reference.v1":
        raise RuntimeError("Reference format tidak dikenal.")
    expected = reference["folds"][f"fold_{fold}"]
    count, val_sha = validation_identity_label_sha256(data_root)
    if count != int(expected["count"]):
        raise RuntimeError(f"Validation count berbeda: {count} != {expected['count']}")
    if val_sha != expected["identity_label_sha256"]:
        raise RuntimeError("Validation identity hash berbeda dari reference.")

    seed_everything(42)
    probe = build_model(copy.deepcopy(cfg["model"]))
    current_initial = model_state_fingerprint(probe)
    del probe
    expected_initial = reference["expected_initial_model_state_sha256"]
    if current_initial != expected_initial:
        raise RuntimeError(
            "Initial model fingerprint berubah dari matched R0 reference: "
            f"{current_initial} != {expected_initial}"
        )

    run_dir = (
        Path(output_root).expanduser().resolve()
        / "MVCE_SBN_ALL4"
        / f"fold_{fold}"
        / "seed42"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["output_dir"] = str(run_dir)

    contract = {
        "format": "bilinear_lmmd.shared_multiview.sbn_contract.v1",
        "protocol": PROTOCOL,
        "scientific_scope": "post-primary exploratory targeted normalization test",
        "fold": int(fold),
        "seed": 42,
        "git_commit": actual_commit,
        "reference_sha256": sha256_file(reference_path),
        "validation_identity_label_sha256": val_sha,
        "validation_count": count,
        "initial_model_state_sha256": current_initial,
        "training_views": ["R0", "C0", "F0", "W0"],
        "deployment_view": "R0",
        "shared_non_bn_parameters": True,
        "normalization_strategy": "auxiliary own mini-batch statistics; only R0 persists running statistics",
        "bn_affine_parameters_shared": True,
        "extra_inference_parameters": 0,
        "loss": {
            "raw_weight": float(cfg["multiview"]["raw_weight"]),
            "auxiliary_weight": float(cfg["multiview"]["auxiliary_weight"]),
            "auxiliary_reduction": "mean(C0,F0,W0)",
        },
        "epochs": int(cfg["training"]["epochs"]),
        "evaluation_split_during_training": "val",
        "test_images_accessed": False,
        "resolved_config_sha256": canonical_json_sha256(cfg),
    }
    contract_sha = canonical_json_sha256(contract)
    contract["run_contract_sha256"] = contract_sha

    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        if _json(contract_path, "Existing contract") != contract:
            raise RuntimeError("Existing run contract berbeda.")
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
        lock_name=f"MVCE_SBN_ALL4_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        training_executed = False
        if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
            train_shared_multiview_sbn(
                cfg,
                run_dir=run_dir,
                run_contract_sha256=contract_sha,
                expected_initial_model_sha256=current_initial,
                resume=resume or (run_dir / "last.pt").is_file(),
            )
            training_executed = True

    if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
        raise RuntimeError("Run SBN belum selesai.")

    metrics = _json(run_dir / "validation" / "metrics.json", "Validation metrics")
    history = _json(run_dir / "history.json", "History")
    best_record = max(history, key=lambda row: float(row["source"]["macro_f1"]))
    baseline = expected["r0_control"]

    result = {
        "format": "bilinear_lmmd.shared_multiview.sbn_result.v1",
        "protocol": PROTOCOL,
        "method": "MVCE_SBN_ALL4",
        "fold": int(fold),
        "seed": 42,
        "metrics": {
            key: metrics[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "worst_class_f1",
                "hard_class_f1",
            )
        },
        "frozen_matched_r0_control": baseline,
        "delta_vs_r0_control": {
            key: metrics[key] - baseline[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "worst_class_f1",
                "hard_class_f1",
            )
        },
        "best_epoch_training_ce": {
            "epoch": int(best_record["epoch"]),
            "raw_ce": float(best_record["raw_ce"]),
            "c0_ce": float(best_record["c0_ce"]),
            "f0_ce": float(best_record["f0_ce"]),
            "w0_ce": float(best_record["w0_ce"]),
        },
        "failed_mvce_reference": expected["failed_mvce"],
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
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_sbn(
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
