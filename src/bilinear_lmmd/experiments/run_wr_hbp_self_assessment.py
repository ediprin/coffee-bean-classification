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
    configure_strict_determinism,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.wavelet_residual_hbp import (
    preflight_matched_initialization,
    train_arm,
)
from bilinear_lmmd.engine.wr_hbp_self_assessment import (
    PROTOCOL,
    train_self_assessment,
    validate_config,
    zero_residual_preflight,
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


def _run_complete_base(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return False
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def _run_complete_head(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best_head.pt"
    last = run_dir / "last_head.pt"
    required_outputs = (
        run_dir / "validation" / "metrics.json",
        run_dir / "validation" / "predictions.csv",
        run_dir / "validation" / "reassessment_diagnostics.csv",
    )
    if (
        not best.is_file()
        or not last.is_file()
        or any(not path.is_file() for path in required_outputs)
    ):
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
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
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


def _symmetric_pair_confusions(
    rows: list[dict],
    pairs: list[list[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for left, right in pairs:
        key = f"{left} <-> {right}"
        counts[key] = sum(
            1
            for row in rows
            if (
                (row["actual"] == left and row["predicted"] == right)
                or (row["actual"] == right and row["predicted"] == left)
            )
        )
    counts["TOTAL"] = sum(counts.values())
    return counts


def _true_rank(row: dict, classes: list[str]) -> int:
    probs = [
        float(row[f"prob::{name}"])
        for name in classes
    ]
    actual = row["actual"]
    if actual not in classes:
        raise RuntimeError(f"Kelas actual tidak dikenal: {actual}")
    actual_index = classes.index(actual)
    order = sorted(range(len(classes)), key=lambda i: probs[i], reverse=True)
    return order.index(actual_index) + 1


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
        raise RuntimeError("Training memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 sebelum CUDA/PyTorch."
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
    wr_preflight = preflight_matched_initialization(cfg)
    val_count, val_sha = validation_identity_label_sha256(data_root)

    output_root = Path(output_root).expanduser().resolve()
    fold_root = output_root / f"fold_{fold}" / "seed42"
    base_dir = fold_root / "WR_HBP_BASE"
    sar_dir = fold_root / "WR_HBP_SAR"
    fold_root.mkdir(parents=True, exist_ok=True)

    base_cfg = copy.deepcopy(cfg)
    base_cfg["training"]["output_dir"] = str(base_dir)
    base_contract = {
        "format": "bilinear_lmmd.wr_hbp_self_assessment.base_contract.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "arm": "WR_HBP_BASE",
        "git_commit": actual_commit,
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "shared_core_initial_sha256": wr_preflight["shared_core_sha256"],
        "gate_zero_preflight": wr_preflight,
        "model": base_cfg["model"],
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
        "strict_determinism": determinism,
        "resolved_config_sha256": canonical_json_sha256(base_cfg),
        "outer_test_accessed": False,
    }
    base_contract_sha = canonical_json_sha256(base_contract)
    base_contract["run_contract_sha256"] = base_contract_sha

    base_dir.mkdir(parents=True, exist_ok=True)
    _write_contract(base_dir / "run_contract.json", base_contract)
    base_cfg_path = base_dir / "run_config.yaml"
    if not base_cfg_path.is_file():
        base_cfg_path.write_text(
            yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    base_epochs = int(cfg["training"]["epochs"])
    sar_epochs = int(cfg["self_assessment"]["epochs"])

    with exclusive_training_lock(
        output_root,
        lock_name=f"WR_HBP_SAR_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        if not _run_complete_base(base_dir, base_epochs):
            configure_strict_determinism(int(cfg["seed"]))
            train_arm(
                cfg,
                arm="WR_HBP",
                run_dir=base_dir,
                run_contract_sha256=base_contract_sha,
                expected_shared_core_sha256=wr_preflight["shared_core_sha256"],
                resume=resume or (base_dir / "last.pt").is_file(),
            )
        if not _run_complete_base(base_dir, base_epochs):
            raise RuntimeError("WR_HBP_BASE belum selesai.")

        configure_strict_determinism(int(cfg["seed"]))
        zero_preflight = zero_residual_preflight(
            cfg,
            base_checkpoint=base_dir / "best.pt",
        )

        sar_cfg = copy.deepcopy(cfg)
        sar_cfg["training"]["output_dir"] = str(sar_dir)
        sar_contract = {
            "format": "bilinear_lmmd.wr_hbp_self_assessment.head_contract.v1",
            "protocol": PROTOCOL,
            "fold": int(fold),
            "seed": 42,
            "arm": "WR_HBP_SAR",
            "git_commit": actual_commit,
            "validation_identity_label_sha256": val_sha,
            "validation_count": int(val_count),
            "base_checkpoint_sha256": zero_preflight["base_checkpoint_sha256"],
            "zero_residual_preflight": zero_preflight,
            "self_assessment": sar_cfg["self_assessment"],
            "label_smoothing": float(sar_cfg["training"]["label_smoothing"]),
            "strict_determinism": determinism,
            "resolved_config_sha256": canonical_json_sha256(sar_cfg),
            "outer_test_accessed": False,
        }
        sar_contract_sha = canonical_json_sha256(sar_contract)
        sar_contract["run_contract_sha256"] = sar_contract_sha

        sar_dir.mkdir(parents=True, exist_ok=True)
        _write_contract(sar_dir / "run_contract.json", sar_contract)
        sar_cfg_path = sar_dir / "run_config.yaml"
        if not sar_cfg_path.is_file():
            sar_cfg_path.write_text(
                yaml.safe_dump(sar_cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

        if not _run_complete_head(sar_dir, sar_epochs):
            configure_strict_determinism(int(cfg["seed"]))
            train_self_assessment(
                sar_cfg,
                base_checkpoint=base_dir / "best.pt",
                run_dir=sar_dir,
                run_contract_sha256=sar_contract_sha,
                resume=resume or (sar_dir / "last_head.pt").is_file(),
            )

    if not _run_complete_head(sar_dir, sar_epochs):
        raise RuntimeError("WR_HBP_SAR belum selesai.")

    base_metrics = _json(base_dir / "validation/metrics.json", "base metrics")
    sar_metrics = _json(sar_dir / "validation/metrics.json", "SAR metrics")
    base_rows = _prediction_rows(base_dir / "validation/predictions.csv")
    sar_rows = _prediction_rows(sar_dir / "validation/predictions.csv")
    if base_metrics["classes"] != sar_metrics["classes"]:
        raise RuntimeError("Urutan kelas base/SAR berbeda.")
    classes = list(base_metrics["classes"])

    outcomes = _paired_outcomes(base_rows, sar_rows)
    delta = {
        key: float(sar_metrics[key]) - float(base_metrics[key])
        for key in METRICS
    }

    pairs = [
        list(pair)
        for pair in cfg["evaluation"]["audited_persistent_confusions"]
    ]
    confusions = {
        "WR_HBP_BASE": _symmetric_pair_confusions(base_rows, pairs),
        "WR_HBP_SAR": _symmetric_pair_confusions(sar_rows, pairs),
    }
    confusion_delta = {
        key: confusions["WR_HBP_SAR"][key] - confusions["WR_HBP_BASE"][key]
        for key in confusions["WR_HBP_BASE"]
    }

    base_error_rows = [row for row in base_rows if row["correct"] == "0"]
    rank_counts = {
        str(rank): sum(_true_rank(row, classes) == rank for row in base_error_rows)
        for rank in range(1, len(classes) + 1)
    }
    rank_counts = {key: value for key, value in rank_counts.items() if value}
    top5_reachable = sum(
        _true_rank(row, classes) <= int(cfg["self_assessment"]["top_k"])
        for row in base_error_rows
    )

    result = {
        "format": "bilinear_lmmd.wr_hbp_self_assessment.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "WR_HBP_BASE": {key: float(base_metrics[key]) for key in METRICS},
        "WR_HBP_SAR": {key: float(sar_metrics[key]) for key in METRICS},
        "DELTA_SAR_MINUS_BASE": delta,
        "paired_outcomes": outcomes,
        "audited_pair_confusions": confusions,
        "audited_pair_confusion_delta": confusion_delta,
        "base_error_true_rank_counts": rank_counts,
        "base_error_top5_reachable": int(top5_reachable),
        "base_error_count": len(base_error_rows),
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "zero_residual_preflight": zero_preflight,
        "base_best_checkpoint_sha256": sha256_file(base_dir / "best.pt"),
        "sar_best_checkpoint_sha256": sha256_file(sar_dir / "best_head.pt"),
        "strict_determinism": determinism,
        "matched_validation_rows": True,
        "base_model_frozen_during_sar_training": True,
        "outer_test_accessed": False,
    }

    (fold_root / "pair_result.json").write_text(
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
