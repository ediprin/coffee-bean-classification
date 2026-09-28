from __future__ import annotations

import argparse
import copy
import csv
import json
import os
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
from bilinear_lmmd.engine.physical_logit_residual_wr_hbp import (
    PROTOCOL,
    configure_strict_determinism,
    fit_and_evaluate_residual,
    validate_config,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.wavelet_residual_hbp import (
    preflight_matched_initialization,
    train_arm,
)


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
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
        raise RuntimeError("Jumlah prediction rows control/candidate berbeda.")

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


def run_fold(
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
        raise RuntimeError("Training base WR-HBP memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 sebelum Python menginisialisasi CUDA."
        )

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

    determinism = configure_strict_determinism(int(cfg["seed"]))
    preflight = preflight_matched_initialization(cfg)
    val_count, val_sha = validation_identity_label_sha256(data_root)

    output_root = Path(output_root).expanduser().resolve()
    fold_root = output_root / f"fold_{fold}" / "seed42"
    base_dir = fold_root / "WR_HBP_BASE"
    residual_dir = fold_root / "WR_PDR_HBP"
    fold_root.mkdir(parents=True, exist_ok=True)

    base_cfg = copy.deepcopy(cfg)
    base_cfg["training"]["output_dir"] = str(base_dir)

    base_contract = {
        "format": "bilinear_lmmd.physical_logit_residual_wr_hbp.base_contract.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "arm": "WR_HBP_BASE",
        "git_commit": actual_commit,
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "shared_core_initial_sha256": preflight["shared_core_sha256"],
        "gate_zero_preflight": preflight,
        "strict_determinism": determinism,
        "model": {
            "backbone": base_cfg["model"]["backbone"],
            "head": base_cfg["model"]["head"],
            "out_indices": list(base_cfg["model"]["out_indices"]),
            "projection_dim": int(base_cfg["model"]["projection_dim"]),
            "classifier": base_cfg["model"]["classifier"],
            "num_classes": int(base_cfg["model"]["num_classes"]),
            "dropout": float(base_cfg["model"].get("dropout", 0.2)),
        },
        "wavelet_residual": base_cfg["wavelet_residual"],
        "training": {
            "epochs": int(base_cfg["training"]["epochs"]),
            "lr": float(base_cfg["training"]["lr"]),
            "weight_decay": float(base_cfg["training"]["weight_decay"]),
            "classification_loss": base_cfg["training"]["classification_loss"],
            "label_smoothing": float(base_cfg["training"]["label_smoothing"]),
            "scheduler": base_cfg["training"]["scheduler"],
            "ema_decay": float(base_cfg["training"]["ema_decay"]),
        },
        "physical_logit_residual": base_cfg["physical_logit_residual"],
        "resolved_config_sha256": canonical_json_sha256(base_cfg),
        "outer_test_accessed": False,
    }
    contract_sha = canonical_json_sha256(base_contract)
    base_contract["run_contract_sha256"] = contract_sha

    base_dir.mkdir(parents=True, exist_ok=True)
    _write_contract(base_dir / "run_contract.json", base_contract)
    run_cfg = base_dir / "run_config.yaml"
    if not run_cfg.is_file():
        run_cfg.write_text(
            yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    epochs = int(cfg["training"]["epochs"])
    with exclusive_training_lock(
        output_root,
        lock_name=f"WR_PDR_HBP_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        if not _run_complete(base_dir, epochs):
            train_arm(
                cfg,
                arm="WR_HBP",
                run_dir=base_dir,
                run_contract_sha256=contract_sha,
                expected_shared_core_sha256=preflight["shared_core_sha256"],
                resume=resume or (base_dir / "last.pt").is_file(),
            )
        else:
            print("SKIP WR_HBP_BASE: completed.", flush=True)

        if not _run_complete(base_dir, epochs):
            raise RuntimeError("WR_HBP_BASE belum selesai.")

        pair = fit_and_evaluate_residual(
            cfg=cfg,
            base_checkpoint=base_dir / "best.pt",
            output_dir=residual_dir,
        )

    wr_rows = _prediction_rows(
        residual_dir / "WR_HBP_validation" / "predictions.csv"
    )
    candidate_rows = _prediction_rows(
        residual_dir / "WR_PDR_HBP_validation" / "predictions.csv"
    )
    outcomes = _paired_outcomes(wr_rows, candidate_rows)

    base_best = torch.load(
        base_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    result = {
        "format": "bilinear_lmmd.physical_logit_residual_wr_hbp.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "WR_HBP": {
            key: float(pair["WR_HBP"][key])
            for key in METRICS
        },
        "WR_PDR_HBP": {
            key: float(pair["WR_PDR_HBP"][key])
            for key in METRICS
        },
        "DELTA_WR_PDR_MINUS_WR": {
            key: float(pair["DELTA_WR_PDR_MINUS_WR"][key])
            for key in METRICS
        },
        "paired_outcomes": outcomes,
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "shared_core_initial_sha256": preflight["shared_core_sha256"],
        "base_best_epoch": int(base_best["epoch"]),
        "base_wavelet_gate": float(base_best.get("wavelet_gate", 0.0)),
        "base_checkpoint_sha256": sha256_file(base_dir / "best.pt"),
        "base_run_contract_sha256": contract_sha,
        "zero_residual_initial_logit_max_abs_difference": float(
            pair["zero_residual_initial_logit_max_abs_difference"]
        ),
        "active_residual_parameter_count": int(
            pair["residual_fit"]["active_parameter_count"]
        ),
        "residual_optimizer_success": bool(
            pair["residual_fit"]["optimizer_success"]
        ),
        "residual_fit_train_count": int(pair["residual_fit_train_count"]),
        "train_descriptor_qc_fail_count": int(
            pair["train_descriptor_qc_fail_count"]
        ),
        "val_descriptor_qc_fail_count": int(
            pair["val_descriptor_qc_fail_count"]
        ),
        "strict_determinism": determinism,
        "base_model_frozen_during_residual_fit": True,
        "insect_topology_enabled": False,
        "outer_test_accessed": False,
    }

    result_path = fold_root / "pair_result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
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

    run_fold(
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
