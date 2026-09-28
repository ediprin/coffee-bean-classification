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
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.wavelet_clahe_residual_hbp import (
    PROTOCOL,
    preflight_matched_initialization,
    train_arm,
    validate_config,
)


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


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _paired_outcomes(control_rows: list[dict], candidate_rows: list[dict]) -> dict:
    if len(control_rows) != len(candidate_rows):
        raise RuntimeError("Jumlah prediction rows WR/WRC berbeda.")

    rescue = damage = both_correct = both_wrong = 0
    for left, right in zip(control_rows, candidate_rows):
        if (left["path"], left["actual"]) != (right["path"], right["actual"]):
            raise RuntimeError("Validation row identity/order berbeda.")
        lc = left["correct"] == "1"
        rc = right["correct"] == "1"
        if not lc and rc:
            rescue += 1
        elif lc and not rc:
            damage += 1
        elif lc and rc:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "count": len(control_rows),
        "rescue": rescue,
        "damage": damage,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_correct": rescue - damage,
    }


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
    validate_config(cfg)

    preflight = preflight_matched_initialization(cfg)
    val_count, val_sha = validation_identity_label_sha256(data_root)

    output_root = Path(output_root).expanduser().resolve()
    pair_root = output_root / f"fold_{fold}" / "seed42"
    pair_root.mkdir(parents=True, exist_ok=True)

    arm_dirs = {
        "WR_HBP": pair_root / "WR_HBP",
        "WRC_HBP": pair_root / "WRC_HBP",
    }
    contracts: dict[str, dict] = {}

    for arm in ("WR_HBP", "WRC_HBP"):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg["training"]["output_dir"] = str(arm_dirs[arm])

        contract = {
            "format": "bilinear_lmmd.wavelet_clahe_residual_hbp.arm_contract.v1",
            "protocol": PROTOCOL,
            "fold": int(fold),
            "seed": 42,
            "arm": arm,
            "git_commit": actual_commit,
            "validation_identity_label_sha256": val_sha,
            "validation_count": int(val_count),
            "shared_wr_initial_sha256": preflight["shared_wr_core_sha256"],
            "zero_gate_preflight": preflight,
            "model": {
                "backbone": arm_cfg["model"]["backbone"],
                "head": arm_cfg["model"]["head"],
                "out_indices": list(arm_cfg["model"]["out_indices"]),
                "projection_dim": int(arm_cfg["model"]["projection_dim"]),
                "classifier": arm_cfg["model"]["classifier"],
                "num_classes": int(arm_cfg["model"]["num_classes"]),
                "dropout": float(arm_cfg["model"].get("dropout", 0.2)),
            },
            "wavelet_residual": arm_cfg["wavelet_residual"],
            "contrast_residual": (
                arm_cfg["contrast_residual"] if arm == "WRC_HBP" else None
            ),
            "training": {
                "epochs": int(arm_cfg["training"]["epochs"]),
                "lr": float(arm_cfg["training"]["lr"]),
                "weight_decay": float(arm_cfg["training"]["weight_decay"]),
                "classification_loss": arm_cfg["training"]["classification_loss"],
                "label_smoothing": float(arm_cfg["training"]["label_smoothing"]),
                "scheduler": arm_cfg["training"]["scheduler"],
                "ema_decay": float(arm_cfg["training"]["ema_decay"]),
            },
            "resolved_config_sha256": canonical_json_sha256(arm_cfg),
            "outer_test_accessed": False,
        }
        contract_sha = canonical_json_sha256(contract)
        contract["run_contract_sha256"] = contract_sha
        contracts[arm] = contract

        arm_dirs[arm].mkdir(parents=True, exist_ok=True)
        _write_contract(arm_dirs[arm] / "run_contract.json", contract)

        run_cfg = arm_dirs[arm] / "run_config.yaml"
        if not run_cfg.is_file():
            run_cfg.write_text(
                yaml.safe_dump(arm_cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

    epochs = int(cfg["training"]["epochs"])
    with exclusive_training_lock(
        output_root,
        lock_name=f"WRC_HBP_MATCHED_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        for arm in ("WR_HBP", "WRC_HBP"):
            if _run_complete(arm_dirs[arm], epochs):
                print(f"SKIP {arm}: completed.", flush=True)
                continue

            train_arm(
                cfg,
                arm=arm,
                run_dir=arm_dirs[arm],
                run_contract_sha256=contracts[arm]["run_contract_sha256"],
                expected_shared_wr_sha256=preflight["shared_wr_core_sha256"],
                resume=resume or (arm_dirs[arm] / "last.pt").is_file(),
            )

    for arm in ("WR_HBP", "WRC_HBP"):
        if not _run_complete(arm_dirs[arm], epochs):
            raise RuntimeError(f"{arm} belum selesai.")

    metrics = {
        arm: _json(arm_dirs[arm] / "validation/metrics.json", f"{arm} metrics")
        for arm in ("WR_HBP", "WRC_HBP")
    }
    rows = {
        arm: _prediction_rows(arm_dirs[arm] / "validation/predictions.csv")
        for arm in ("WR_HBP", "WRC_HBP")
    }
    outcomes = _paired_outcomes(rows["WR_HBP"], rows["WRC_HBP"])

    delta = {
        key: float(metrics["WRC_HBP"][key]) - float(metrics["WR_HBP"][key])
        for key in METRICS
    }

    control_best = torch.load(
        arm_dirs["WR_HBP"] / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    candidate_best = torch.load(
        arm_dirs["WRC_HBP"] / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    result = {
        "format": "bilinear_lmmd.wavelet_clahe_residual_hbp.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "WR_HBP": {key: float(metrics["WR_HBP"][key]) for key in METRICS},
        "WRC_HBP": {key: float(metrics["WRC_HBP"][key]) for key in METRICS},
        "DELTA_WRC_MINUS_WR": delta,
        "paired_outcomes": outcomes,
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "shared_wr_initial_sha256": preflight["shared_wr_core_sha256"],
        "initial_logit_max_abs_difference": preflight[
            "initial_logit_max_abs_difference"
        ],
        "matched_shared_wr_initialization": True,
        "matched_validation_rows": True,
        "wr_control_retrained": True,
        "wrc_candidate_trained": True,
        "wavelet_gate_at_best": {
            "WR_HBP": float(control_best.get("wavelet_gate", 0.0)),
            "WRC_HBP": float(candidate_best.get("wavelet_gate", 0.0)),
        },
        "contrast_gate_at_best": float(candidate_best.get("contrast_gate", 0.0)),
        "best_epoch": {
            "WR_HBP": int(control_best["epoch"]),
            "WRC_HBP": int(candidate_best["epoch"]),
        },
        "outer_test_accessed": False,
        "best_checkpoint_sha256": {
            arm: sha256_file(arm_dirs[arm] / "best.pt")
            for arm in ("WR_HBP", "WRC_HBP")
        },
        "arm_contract_sha256": {
            arm: contracts[arm]["run_contract_sha256"]
            for arm in ("WR_HBP", "WRC_HBP")
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
