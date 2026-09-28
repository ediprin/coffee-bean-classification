from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
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
from bilinear_lmmd.engine.w0_hbp import (
    PROTOCOL,
    train_hbp_preprocessing_arm,
    validate_arm_preprocessing,
    validate_hbp_preprocessing_config,
)
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
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def _write_contract(path: Path, payload: dict) -> None:
    if path.is_file():
        if _json(path, "Existing contract") != payload:
            raise RuntimeError(f"Existing contract berbeda: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _prediction_identity(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["path"], row["actual"]))
    return rows


def run_matched_pair(
    *,
    config_path: Path,
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
    validate_hbp_preprocessing_config(cfg)

    arms = cfg.get("preprocessing_arms", {})
    if tuple(arms) != ("R0", "W0"):
        raise ValueError("preprocessing_arms harus tepat R0 lalu W0.")
    validate_arm_preprocessing(arms["R0"], "R0")
    validate_arm_preprocessing(arms["W0"], "W0")

    val_count, val_sha = validation_identity_label_sha256(data_root)

    seed_everything(42)
    probe = build_model(copy.deepcopy(cfg["model"]))
    initial_sha = model_state_fingerprint(probe)
    del probe

    output_root = Path(output_root).expanduser().resolve()
    pair_root = output_root / f"fold_{fold}" / "seed42"
    pair_root.mkdir(parents=True, exist_ok=True)

    arm_dirs = {
        "R0": pair_root / "R0_HBP",
        "W0": pair_root / "W0_HBP",
    }
    arm_contracts: dict[str, dict] = {}

    for arm in ("R0", "W0"):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg["preprocessing"] = copy.deepcopy(arms[arm])
        arm_cfg["training"]["output_dir"] = str(arm_dirs[arm])
        arm_cfg.pop("preprocessing_arms", None)

        contract = {
            "format": "bilinear_lmmd.w0_hbp_matched.arm_contract.v1",
            "protocol": PROTOCOL,
            "fold": int(fold),
            "seed": 42,
            "arm": arm,
            "git_commit": actual_commit,
            "validation_identity_label_sha256": val_sha,
            "validation_count": int(val_count),
            "initial_model_state_sha256": initial_sha,
            "model": {
                "backbone": arm_cfg["model"]["backbone"],
                "head": arm_cfg["model"]["head"],
                "out_indices": list(arm_cfg["model"]["out_indices"]),
                "projection_dim": int(arm_cfg["model"]["projection_dim"]),
                "classifier": arm_cfg["model"]["classifier"],
                "num_classes": int(arm_cfg["model"]["num_classes"]),
            },
            "training": {
                "epochs": int(arm_cfg["training"]["epochs"]),
                "lr": float(arm_cfg["training"]["lr"]),
                "weight_decay": float(arm_cfg["training"]["weight_decay"]),
                "classification_loss": arm_cfg["training"]["classification_loss"],
                "label_smoothing": float(arm_cfg["training"]["label_smoothing"]),
                "scheduler": arm_cfg["training"]["scheduler"],
                "ema_decay": float(arm_cfg["training"]["ema_decay"]),
            },
            "preprocessing": arm_cfg["preprocessing"],
            "resolved_config_sha256": canonical_json_sha256(arm_cfg),
            "test_images_accessed": False,
        }
        contract_sha = canonical_json_sha256(contract)
        contract["run_contract_sha256"] = contract_sha
        arm_contracts[arm] = contract

        arm_dirs[arm].mkdir(parents=True, exist_ok=True)
        _write_contract(arm_dirs[arm] / "run_contract.json", contract)
        run_cfg_path = arm_dirs[arm] / "run_config.yaml"
        if not run_cfg_path.is_file():
            run_cfg_path.write_text(
                yaml.safe_dump(arm_cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

    epochs = int(cfg["training"]["epochs"])
    with exclusive_training_lock(
        output_root,
        lock_name=f"W0_HBP_MATCHED_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        for arm in ("R0", "W0"):
            if _run_complete(arm_dirs[arm], epochs):
                print(f"SKIP {arm}-HBP: completed.", flush=True)
                continue

            arm_cfg = copy.deepcopy(cfg)
            arm_cfg["preprocessing"] = copy.deepcopy(arms[arm])
            arm_cfg["training"]["output_dir"] = str(arm_dirs[arm])
            arm_cfg.pop("preprocessing_arms", None)

            train_hbp_preprocessing_arm(
                arm_cfg,
                arm=arm,
                run_dir=arm_dirs[arm],
                run_contract_sha256=arm_contracts[arm]["run_contract_sha256"],
                expected_initial_model_sha256=initial_sha,
                resume=resume or (arm_dirs[arm] / "last.pt").is_file(),
            )

    for arm in ("R0", "W0"):
        if not _run_complete(arm_dirs[arm], epochs):
            raise RuntimeError(f"{arm}-HBP belum selesai.")

    identities = {
        arm: _prediction_identity(arm_dirs[arm] / "validation/predictions.csv")
        for arm in ("R0", "W0")
    }
    if identities["R0"] != identities["W0"]:
        raise RuntimeError("Validation rows R0-HBP dan W0-HBP tidak identik.")

    metrics = {
        arm: _json(arm_dirs[arm] / "validation/metrics.json", f"{arm}-HBP metrics")
        for arm in ("R0", "W0")
    }
    delta = {
        key: float(metrics["W0"][key]) - float(metrics["R0"][key])
        for key in METRICS
    }

    result = {
        "format": "bilinear_lmmd.w0_hbp_matched.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "R0_HBP": {key: float(metrics["R0"][key]) for key in METRICS},
        "W0_HBP": {key: float(metrics["W0"][key]) for key in METRICS},
        "DELTA_W0_MINUS_R0": delta,
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "initial_model_state_sha256": initial_sha,
        "matched_initialization": True,
        "matched_validation_rows": True,
        "r0_control_retrained": True,
        "w0_candidate_trained": True,
        "outer_test_accessed": False,
        "arm_contract_sha256": {
            arm: arm_contracts[arm]["run_contract_sha256"]
            for arm in ("R0", "W0")
        },
        "best_checkpoint_sha256": {
            arm: sha256_file(arm_dirs[arm] / "best.pt")
            for arm in ("R0", "W0")
        },
    }
    result_path = pair_root / "pair_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()

    run_matched_pair(
        config_path=args.config,
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
