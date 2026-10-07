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
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.experiments.run_frequency_screening_v1 import (
    METRICS,
    _run_complete,
    preflight_candidate,
    train_candidate,
    validate_config,
)
from bilinear_lmmd.modeling.frequency_screening import (
    FREQUENCY_CANDIDATES,
    build_frequency_screening_model,
)

PROTOCOL = "coffee17-focused-frequency-efficiency-c1c2-v1"
HISTORICAL_PROTOCOL = "coffee17-master-representation-screening-v1"
HISTORICAL_COMMIT = "b1ac6ca548ebb335d69066377d6f1a1fbb56df36"
ALLOWED_CANDIDATES = ("C1", "C2")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _common_protocol_signature(cfg: dict) -> dict:
    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    return {
        "seed": int(cfg["seed"]),
        "data": {
            "source": data.get("source", "source"),
            "train_split": data.get("train_split", "train"),
            "val_split": data.get("val_split", "val"),
            "image_size": int(data["image_size"]),
            "batch_size": int(data["batch_size"]),
            "rotation_angles": list(data["rotation_angles"]),
            "augmentation_mode": data["augmentation_mode"],
            "object_crop": bool(data.get("object_crop", False)),
        },
        "model": {
            "backbone": model["backbone"],
            "pretrained": bool(model.get("pretrained", True)),
            "classifier": model.get("classifier", "linear"),
            "num_classes": int(model["num_classes"]),
            "projection_dim": int(model.get("projection_dim", 512)),
            "dropout": float(model.get("dropout", 0.2)),
        },
        "adaptation": {
            "method": cfg.get("adaptation", {}).get("method"),
        },
        "training": {
            "epochs": int(training["epochs"]),
            "lr": float(training["lr"]),
            "weight_decay": float(training["weight_decay"]),
            "classification_loss": training["classification_loss"],
            "label_smoothing": float(training["label_smoothing"]),
            "freeze_backbone": bool(training.get("freeze_backbone", False)),
            "scheduler": training["scheduler"],
            "ema_decay": float(training.get("ema_decay", 0.0)),
        },
    }


def _identity(row: dict[str, str]) -> tuple[str, str]:
    path = Path(row["path"])
    return (f"{path.parent.name}/{path.name}", row["actual"])


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _paired_outcomes_historical(
    anchor_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> dict:
    if len(anchor_rows) != len(candidate_rows):
        raise RuntimeError("Jumlah validation rows historical/candidate berbeda.")
    left = {_identity(row): row for row in anchor_rows}
    right = {_identity(row): row for row in candidate_rows}
    if left.keys() != right.keys():
        only_left = sorted(left.keys() - right.keys())[:5]
        only_right = sorted(right.keys() - left.keys())[:5]
        raise RuntimeError(
            "Identity validation historical/candidate berbeda. "
            f"only_historical={only_left}, only_candidate={only_right}"
        )

    rescue = damage = both_correct = both_wrong = 0
    for key in sorted(left):
        lc = left[key]["correct"] == "1"
        rc = right[key]["correct"] == "1"
        if not lc and rc:
            rescue += 1
        elif lc and not rc:
            damage += 1
        elif lc and rc:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "count": len(left),
        "rescue": rescue,
        "damage": damage,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_correct": rescue - damage,
    }


def audit_historical_b0(
    *,
    historical_root: Path,
    cfg: dict,
    fold: int,
    validation_count: int,
    validation_sha256: str,
) -> dict:
    run_dir = historical_root / "experiments" / f"fold_{fold}" / "seed42" / "B0"
    contract = _read_json(run_dir / "run_contract.json")
    old_cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    metrics = _read_json(run_dir / "validation" / "metrics.json")
    predictions = run_dir / "validation" / "predictions.csv"

    checks = {
        "protocol": contract.get("protocol") == HISTORICAL_PROTOCOL,
        "candidate": contract.get("candidate") == "B0",
        "anchor": contract.get("anchor") == "B0",
        "fold": int(contract.get("fold", -1)) == fold,
        "seed": int(contract.get("seed", -1)) == 42,
        "historical_commit": contract.get("git_commit") == HISTORICAL_COMMIT,
        "outer_test_locked": contract.get("outer_test_accessed") is False,
        "validation_count": int(contract.get("validation_count", -1)) == validation_count,
        "validation_sha256": (
            contract.get("validation_identity_label_sha256") == validation_sha256
        ),
        "common_protocol": (
            _common_protocol_signature(old_cfg) == _common_protocol_signature(cfg)
        ),
        "predictions_present": predictions.is_file(),
    }

    seed_everything(42)
    current_anchor_preflight = preflight_candidate(cfg, "B0")
    historical_core = contract.get("preflight", {}).get("shared_core_sha256")
    checks["shared_core_sha256"] = (
        historical_core is not None
        and current_anchor_preflight.get("shared_core_sha256") == historical_core
    )

    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise RuntimeError(
            f"Historical B0 fold {fold} tidak boleh dipakai; audit gagal: {failed}"
        )

    return {
        "fold": fold,
        "historical_run_dir": str(run_dir),
        "historical_commit": contract["git_commit"],
        "historical_run_contract_sha256": contract.get("run_contract_sha256"),
        "validation_count": validation_count,
        "validation_identity_label_sha256": validation_sha256,
        "shared_core_sha256": historical_core,
        "parameter_count": int(contract["parameter_count"]),
        "metrics": {key: float(metrics[key]) for key in METRICS},
        "predictions_csv": str(predictions),
        "checks": checks,
    }


def run_fold(
    *,
    config_path: Path,
    fold: int,
    data_root: Path,
    historical_root: Path,
    output_root: Path,
    candidates: list[str],
    device: str,
    resume: bool,
    authorize_training: bool,
    required_commit: str | None,
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training C1/C2 memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")
    if not candidates:
        raise ValueError("Minimal satu candidate diperlukan.")
    invalid = sorted(set(candidates).difference(ALLOWED_CANDIDATES))
    if invalid:
        raise ValueError(f"Focused C1/C2 hanya menerima {ALLOWED_CANDIDATES}; didapat {invalid}")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if required_commit and actual_commit != required_commit:
        raise RuntimeError(f"Git commit berbeda: {actual_commit} != {required_commit}")

    cfg = copy.deepcopy(load_config(config_path))
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    validate_config(cfg)

    val_count, val_sha = validation_identity_label_sha256(data_root)
    historical = audit_historical_b0(
        historical_root=historical_root,
        cfg=cfg,
        fold=fold,
        validation_count=val_count,
        validation_sha256=val_sha,
    )

    fold_root = Path(output_root).expanduser().resolve() / f"fold_{fold}" / "seed42"
    fold_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    print(
        f"HISTORICAL B0 VERIFIED fold={fold}: "
        f"count={val_count} sha={val_sha} — NO B0 TRAINING",
        flush=True,
    )

    anchor_rows = _prediction_rows(Path(historical["predictions_csv"]))

    for candidate in candidates:
        if FREQUENCY_CANDIDATES[candidate]["anchor"] != "B0":
            raise RuntimeError(f"{candidate} bukan kandidat dengan anchor B0.")

        preflight = preflight_candidate(cfg, candidate)
        if preflight.get("anchor_shared_core_sha256") != historical["shared_core_sha256"]:
            raise RuntimeError(
                f"{candidate}: shared-core init tidak identik dengan historical B0."
            )

        run_dir = fold_root / candidate
        run_dir.mkdir(parents=True, exist_ok=True)
        probe = build_frequency_screening_model(candidate, cfg)
        parameter_count = sum(p.numel() for p in probe.parameters())
        del probe

        contract = {
            "format": "bilinear_lmmd.focused_c1c2.arm_contract.v1",
            "protocol": PROTOCOL,
            "fold": fold,
            "seed": 42,
            "candidate": candidate,
            "anchor": "B0",
            "anchor_mode": "verified_historical_master",
            "git_commit": actual_commit,
            "historical_anchor_commit": HISTORICAL_COMMIT,
            "historical_anchor_run_contract_sha256": historical[
                "historical_run_contract_sha256"
            ],
            "validation_identity_label_sha256": val_sha,
            "validation_count": val_count,
            "preflight": preflight,
            "historical_anchor_audit": historical["checks"],
            "parameter_count": parameter_count,
            "outer_test_accessed": False,
            "resolved_config_sha256": canonical_json_sha256(cfg),
        }
        contract["run_contract_sha256"] = canonical_json_sha256(contract)

        contract_path = run_dir / "run_contract.json"
        if contract_path.is_file():
            existing = _read_json(contract_path)
            if existing != contract:
                raise RuntimeError(f"Run contract berbeda: {contract_path}")
        else:
            contract_path.write_text(
                json.dumps(contract, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "run_config.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

        if _run_complete(run_dir, int(cfg["training"]["epochs"])):
            print(f"SKIP {candidate}: completed.", flush=True)
        else:
            train_candidate(
                cfg,
                candidate=candidate,
                run_dir=run_dir,
                run_contract_sha256=contract["run_contract_sha256"],
                resume=resume or (run_dir / "last.pt").is_file(),
            )

        metrics = _read_json(run_dir / "validation" / "metrics.json")
        candidate_rows = _prediction_rows(run_dir / "validation" / "predictions.csv")
        delta = {
            key: float(metrics[key]) - float(historical["metrics"][key])
            for key in METRICS
        }
        results[candidate] = {
            "metrics": {key: float(metrics[key]) for key in METRICS},
            "delta_vs_historical_b0": delta,
            "paired_outcomes": _paired_outcomes_historical(
                anchor_rows, candidate_rows
            ),
            "parameter_count": parameter_count,
            "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        }

    payload = {
        "format": "bilinear_lmmd.focused_c1c2.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": fold,
        "seed": 42,
        "git_commit": actual_commit,
        "validation_identity_label_sha256": val_sha,
        "validation_count": val_count,
        "historical_b0": historical,
        "candidates": candidates,
        "results": results,
        "outer_test_accessed": False,
    }
    out = fold_root / "focused_fold_result.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train only C1/C2 against a verified historical B0 anchor."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=ALLOWED_CANDIDATES,
        default=list(ALLOWED_CANDIDATES),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--required-commit")
    args = parser.parse_args()

    run_fold(
        config_path=args.config,
        fold=args.fold,
        data_root=args.data_root,
        historical_root=args.historical_root.expanduser().resolve(),
        output_root=args.output_root,
        candidates=args.candidates,
        device=args.device,
        resume=args.resume,
        authorize_training=args.authorize_training,
        required_commit=args.required_commit,
    )


if __name__ == "__main__":
    main()
