from __future__ import annotations

import argparse
import json
from pathlib import Path

from bilinear_lmmd.core.config import load_config


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


def summarize(*, output_root: Path, config_path: Path, output: Path) -> dict:
    cfg = load_config(config_path)
    folds = {}
    for fold in range(1, 6):
        result = _read(
            Path(output_root) / f"fold_{fold}" / "seed42" / "fold_result.json"
        )
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test flag invalid.")
        if result.get("hbp_gpu_smoke", {}).get("passed") is not True:
            raise RuntimeError(f"Fold {fold}: HBP GPU smoke bukan PASS.")
        folds[fold] = result

    arm_means = {}
    for arm in ("HF20", "HBP_EMB", "HF20_HBP_EMB"):
        arm_means[arm] = {
            metric: sum(folds[f]["arms"][arm][metric] for f in folds) / 5.0
            for metric in METRICS
        }

    deltas = {}
    for metric in METRICS:
        values = [
            folds[f]["delta_fusion_minus_deep"][metric]
            for f in sorted(folds)
        ]
        deltas[metric] = {
            "values": values,
            "mean": sum(values) / 5.0,
            "positive_folds": sum(value > 0.0 for value in values),
            "nonnegative_folds": sum(value >= 0.0 for value in values),
        }

    targeted_deep = sum(
        folds[f]["targeted_confusions"]["HBP_EMB"]["TOTAL"] for f in folds
    )
    targeted_fusion = sum(
        folds[f]["targeted_confusions"]["HF20_HBP_EMB"]["TOTAL"] for f in folds
    )
    outcomes = {
        key: sum(
            folds[f]["paired_outcomes_fusion_vs_deep"][key]
            for f in folds
        )
        for key in ("count", "rescue", "damage", "both_correct", "both_wrong", "net_correct")
    }

    gate_cfg = cfg["screening_gate"]
    criteria = {
        "mean_macro_delta_positive": (
            deltas["macro_f1"]["mean"] > float(gate_cfg["mean_macro_delta_gt"])
        ),
        "macro_positive_at_least_4_of_5": (
            deltas["macro_f1"]["positive_folds"]
            >= int(gate_cfg["macro_positive_folds_at_least"])
        ),
        "mean_hard_delta_positive": (
            deltas["hard_class_f1"]["mean"] > float(gate_cfg["mean_hard_delta_gt"])
        ),
        "targeted_confusions_reduced": (
            targeted_fusion - targeted_deep
            < int(gate_cfg["targeted_confusion_delta_lt"])
        ),
        "mean_worst_delta_nonnegative": (
            deltas["worst_class_f1"]["mean"]
            >= float(gate_cfg["mean_worst_delta_ge"])
        ),
    }
    decision = (
        "SUPPORT_HF_COMPLEMENTARITY"
        if all(criteria.values())
        else "NO_CLEAR_HF_COMPLEMENTARITY"
    )

    selected_frequency = {}
    for fold in folds:
        for name in folds[fold]["selected_feature_names"]:
            selected_frequency[name] = selected_frequency.get(name, 0) + 1
    selected_frequency = dict(
        sorted(selected_frequency.items(), key=lambda item: (-item[1], item[0]))
    )

    summary = {
        "format": "bilinear_lmmd.hf_deep.summary.v1",
        "protocol": "coffee17-hf-deep-complementarity-v1",
        "folds": 5,
        "seed": 42,
        "arm_means": arm_means,
        "delta_fusion_minus_deep": deltas,
        "targeted_confusions_total": {
            "HBP_EMB": targeted_deep,
            "HF20_HBP_EMB": targeted_fusion,
            "delta": targeted_fusion - targeted_deep,
        },
        "paired_prediction_outcomes": outcomes,
        "selected_feature_frequency": selected_frequency,
        "screening_gate": {
            "criteria": criteria,
            "decision": decision,
        },
        "outer_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(
        output_root=args.output_root,
        config_path=args.config,
        output=args.output,
    )


if __name__ == "__main__":
    main()
