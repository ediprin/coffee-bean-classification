from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
        "positive_folds": sum(value > 0.0 for value in values),
        "nonnegative_folds": sum(value >= 0.0 for value in values),
    }


def run_summary(*, output_root: Path, output: Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    folds = {}
    for fold in FOLDS:
        result = _read(output_root / f"fold_{fold}" / "seed42" / "pair_result.json")
        if result.get("protocol") != "coffee17-dcl-local-learning-v1":
            raise RuntimeError(f"Fold {fold}: protocol mismatch.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        if result.get("matched_core_initialization") is not True:
            raise RuntimeError(f"Fold {fold}: initialization tidak matched.")
        if result.get("matched_validation_rows") is not True:
            raise RuntimeError(f"Fold {fold}: validation rows tidak matched.")
        if result.get("inference_architecture_identical") is not True:
            raise RuntimeError(f"Fold {fold}: inference architecture berubah.")
        if result.get("gpu_training_smoke", {}).get("passed") is not True:
            raise RuntimeError(f"Fold {fold}: GPU training smoke tidak PASS.")
        folds[fold] = result

    aggregate = {}
    for arm in ("HBP_CE", "HBP_DCL"):
        aggregate[arm] = {
            metric: _stats([
                float(folds[fold][arm][metric])
                for fold in FOLDS
            ])
            for metric in METRICS
        }

    delta = {
        metric: _stats([
            float(folds[fold]["DELTA_DCL_MINUS_CE"][metric])
            for fold in FOLDS
        ])
        for metric in METRICS
    }

    targeted_keys = [
        key
        for key in folds[1]["targeted_confusions"]["HBP_CE"].keys()
        if key != "TOTAL"
    ]
    targeted = {
        arm: {
            key: sum(
                int(folds[fold]["targeted_confusions"][arm][key])
                for fold in FOLDS
            )
            for key in targeted_keys + ["TOTAL"]
        }
        for arm in ("HBP_CE", "HBP_DCL")
    }
    targeted_delta = {
        key: targeted["HBP_DCL"][key] - targeted["HBP_CE"][key]
        for key in targeted_keys + ["TOTAL"]
    }

    paired = {
        key: sum(
            int(folds[fold]["paired_outcomes"][key])
            for fold in FOLDS
        )
        for key in ("count", "rescue", "damage", "both_correct", "both_wrong", "net_correct")
    }

    gate = {
        "criteria": {
            "macro_mean_delta_gt_0": delta["macro_f1"]["mean"] > 0.0,
            "macro_positive_folds_ge_4": delta["macro_f1"]["positive_folds"] >= 4,
            "hard_mean_delta_gt_0": delta["hard_class_f1"]["mean"] > 0.0,
            "hard_positive_folds_ge_4": delta["hard_class_f1"]["positive_folds"] >= 4,
            "targeted_confusions_reduced": targeted["HBP_DCL"]["TOTAL"] < targeted["HBP_CE"]["TOTAL"],
            "worst_mean_not_degraded": delta["worst_class_f1"]["mean"] >= 0.0,
        },
        "control_targeted_confusions_total": targeted["HBP_CE"]["TOTAL"],
        "dcl_targeted_confusions_total": targeted["HBP_DCL"]["TOTAL"],
    }
    gate["decision"] = (
        "SUPPORT_LOCAL_DCL"
        if all(gate["criteria"].values())
        else "NO_CLEAR_DCL_SUPPORT"
    )

    payload = {
        "format": "bilinear_lmmd.dcl_local_learning.summary.v1",
        "protocol": "coffee17-dcl-local-learning-v1",
        "scope": (
            "five locked Coffee17 development folds; matched raw-RGB HBP control; "
            "DCL training-only RCM + binary swap + location reconstruction; "
            "identical HBP inference architecture; outer test untouched"
        ),
        "aggregate": aggregate,
        "delta_dcl_minus_ce": delta,
        "targeted_confusions": targeted,
        "targeted_confusion_delta": targeted_delta,
        "paired_prediction_outcomes": paired,
        "arm_git_commit_by_fold": {
            str(fold): folds[fold].get("arm_git_commit", {})
            for fold in FOLDS
        },
        "legacy_control_reused_folds": [
            fold for fold in FOLDS
            if folds[fold].get("legacy_control_reused") is True
        ],
        "gpu_training_smoke_passed_all_folds": True,
        "screening_gate": gate,
        "outer_test_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(output_root=args.output_root, output=args.output)


if __name__ == "__main__":
    main()
