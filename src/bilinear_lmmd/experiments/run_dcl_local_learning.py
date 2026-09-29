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
from bilinear_lmmd.engine.dcl_local_learning import (
    ARMS,
    PROTOCOL,
    gpu_training_smoke_test,
    preflight_matched_initialization,
    train_arm,
    validate_config,
)
from bilinear_lmmd.engine.physical_logit_residual_wr_hbp import (
    configure_strict_determinism,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)

# These commits differ from the corrected protocol only in DCL-candidate
# implementation / notebook plumbing. The HBP_CE code path is unchanged.
# Reusing an existing control from one of them avoids wasting a completed
# 50-epoch control while keeping provenance explicit.
CONTROL_COMPATIBLE_COMMITS = {
    "993b67aaac1a8bbefa2ddb32ab4ce23056dd6172",
    "6f304629f1802f240a1110e6ed6b8c3b7f28d152",
}


def _json(path: Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    metrics = run_dir / "validation" / "metrics.json"
    predictions = run_dir / "validation" / "predictions.csv"
    if not all(path.is_file() for path in (best, last, metrics, predictions)):
        return False
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def _write_contract(path: Path, payload: dict) -> None:
    if path.is_file():
        if _json(path, "Existing contract") != payload:
            raise RuntimeError(f"Existing contract berbeda: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _control_contract_is_compatible(existing: dict, proposed: dict) -> bool:
    if existing.get("git_commit") not in CONTROL_COMPATIBLE_COMMITS:
        return False
    exact_fields = (
        "format",
        "protocol",
        "fold",
        "seed",
        "arm",
        "validation_identity_label_sha256",
        "validation_count",
        "initial_core_state_sha256",
        "model",
        "dcl",
        "training",
        "resolved_config_sha256",
        "outer_test_accessed",
    )
    return all(existing.get(key) == proposed.get(key) for key in exact_fields)


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


def _targeted_confusions(rows: list[dict], pairs: list[list[str]]) -> dict[str, int]:
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

    # Fail fast on the exact strict-deterministic CUDA backward path before
    # spending minutes training the control.
    smoke = gpu_training_smoke_test(cfg, device=device)
    if smoke.get("passed") is not True:
        raise RuntimeError("GPU training smoke test gagal.")
    print(json.dumps({"gpu_training_smoke": smoke}, indent=2), flush=True)

    val_count, val_sha = validation_identity_label_sha256(data_root)

    output_root = Path(output_root).expanduser().resolve()
    pair_root = output_root / f"fold_{fold}" / "seed42"
    pair_root.mkdir(parents=True, exist_ok=True)
    arm_dirs = {arm: pair_root / arm for arm in ARMS}
    contracts: dict[str, dict] = {}

    for arm in ARMS:
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg["training"]["output_dir"] = str(arm_dirs[arm])
        contract = {
            "format": "bilinear_lmmd.dcl_local_learning.arm_contract.v1",
            "protocol": PROTOCOL,
            "fold": int(fold),
            "seed": 42,
            "arm": arm,
            "git_commit": actual_commit,
            "validation_identity_label_sha256": val_sha,
            "validation_count": int(val_count),
            "initial_core_state_sha256": preflight["initial_core_state_sha256"],
            "matched_initialization_preflight": preflight,
            "model": arm_cfg["model"],
            "dcl": arm_cfg["dcl"] if arm == "HBP_DCL" else None,
            "training": {
                "epochs": int(arm_cfg["training"]["epochs"]),
                "lr": float(arm_cfg["training"]["lr"]),
                "weight_decay": float(arm_cfg["training"]["weight_decay"]),
                "classification_loss": arm_cfg["training"]["classification_loss"],
                "label_smoothing": float(arm_cfg["training"]["label_smoothing"]),
                "scheduler": arm_cfg["training"]["scheduler"],
                "ema_decay": float(arm_cfg["training"]["ema_decay"]),
            },
            "strict_determinism": determinism,
            "resolved_config_sha256": canonical_json_sha256(arm_cfg),
            "outer_test_accessed": False,
        }
        contract_sha = canonical_json_sha256(contract)
        contract["run_contract_sha256"] = contract_sha

        arm_dir = arm_dirs[arm]
        contract_path = arm_dir / "run_contract.json"
        if contract_path.is_file():
            existing = _json(contract_path, f"{arm} existing contract")
            if existing == contract:
                contracts[arm] = existing
            elif (
                arm == "HBP_CE"
                and _control_contract_is_compatible(existing, contract)
                and _run_complete(arm_dir, int(cfg["training"]["epochs"]))
            ):
                print(
                    "REUSE completed HBP_CE dari compatible legacy commit: "
                    f"{existing.get('git_commit')}",
                    flush=True,
                )
                contracts[arm] = existing
            else:
                # A stale DCL candidate must never be resumed across a
                # scientific-code change. Remove only that arm; keep any
                # compatible finished control.
                import shutil
                print(f"RESET stale arm: {arm}", flush=True)
                shutil.rmtree(arm_dir, ignore_errors=True)
                arm_dir.mkdir(parents=True, exist_ok=True)
                _write_contract(contract_path, contract)
                contracts[arm] = contract
        else:
            arm_dir.mkdir(parents=True, exist_ok=True)
            _write_contract(contract_path, contract)
            contracts[arm] = contract

        run_cfg = arm_dir / "run_config.yaml"
        if not run_cfg.is_file():
            run_cfg.write_text(
                yaml.safe_dump(arm_cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

    epochs = int(cfg["training"]["epochs"])
    with exclusive_training_lock(
        output_root,
        lock_name=f"DCL_LOCAL_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        # The candidate is the new/risky code path. Run it first so any
        # remaining candidate-only failure cannot waste a full control run.
        training_order = ("HBP_DCL", "HBP_CE")
        for arm in training_order:
            if _run_complete(arm_dirs[arm], epochs):
                print(f"SKIP {arm}: completed.", flush=True)
                continue
            configure_strict_determinism(int(cfg["seed"]))
            train_arm(
                cfg,
                arm=arm,
                run_dir=arm_dirs[arm],
                run_contract_sha256=contracts[arm]["run_contract_sha256"],
                expected_initial_core_sha256=preflight["initial_core_state_sha256"],
                resume=resume or (arm_dirs[arm] / "last.pt").is_file(),
            )

    metrics = {
        arm: _json(arm_dirs[arm] / "validation/metrics.json", f"{arm} metrics")
        for arm in ARMS
    }
    rows = {
        arm: _prediction_rows(arm_dirs[arm] / "validation/predictions.csv")
        for arm in ARMS
    }

    pairs = [list(pair) for pair in cfg["evaluation"]["targeted_confusion_pairs"]]
    targeted = {arm: _targeted_confusions(rows[arm], pairs) for arm in ARMS}
    targeted_delta = {
        key: targeted["HBP_DCL"][key] - targeted["HBP_CE"][key]
        for key in targeted["HBP_CE"]
    }
    delta = {
        key: float(metrics["HBP_DCL"][key]) - float(metrics["HBP_CE"][key])
        for key in METRICS
    }
    outcomes = _paired_outcomes(rows["HBP_CE"], rows["HBP_DCL"])

    best = {
        arm: torch.load(
            arm_dirs[arm] / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        for arm in ARMS
    }

    result = {
        "format": "bilinear_lmmd.dcl_local_learning.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "HBP_CE": {key: float(metrics["HBP_CE"][key]) for key in METRICS},
        "HBP_DCL": {key: float(metrics["HBP_DCL"][key]) for key in METRICS},
        "DELTA_DCL_MINUS_CE": delta,
        "paired_outcomes": outcomes,
        "targeted_confusions": targeted,
        "targeted_confusion_delta": targeted_delta,
        "validation_identity_label_sha256": val_sha,
        "validation_count": int(val_count),
        "initial_core_state_sha256": preflight["initial_core_state_sha256"],
        "initial_logit_max_abs_difference": preflight[
            "initial_logit_max_abs_difference"
        ],
        "matched_core_initialization": True,
        "matched_validation_rows": True,
        "gpu_training_smoke": smoke,
        "inference_architecture_identical": (
            preflight.get("matched_core_state_keys") is True
            and preflight.get("matched_core_tensor_equality") is True
            and preflight["control_inference_parameter_count"]
            == preflight["candidate_inference_parameter_count"]
        ),
        "inference_parameter_count": preflight["control_inference_parameter_count"],
        "candidate_training_only_parameter_count": preflight[
            "candidate_training_only_parameter_count"
        ],
        "best_epoch": {
            arm: int(best[arm]["epoch"])
            for arm in ARMS
        },
        "outer_test_accessed": False,
        "best_checkpoint_sha256": {
            arm: sha256_file(arm_dirs[arm] / "best.pt")
            for arm in ARMS
        },
        "arm_contract_sha256": {
            arm: contracts[arm]["run_contract_sha256"]
            for arm in ARMS
        },
        "arm_git_commit": {
            arm: contracts[arm]["git_commit"]
            for arm in ARMS
        },
        "legacy_control_reused": (
            contracts["HBP_CE"]["git_commit"] != actual_commit
        ),
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
