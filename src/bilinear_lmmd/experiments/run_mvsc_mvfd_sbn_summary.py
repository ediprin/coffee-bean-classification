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
    "worst_class_f1",
    "hard_class_f1",
)
METHOD = "MVSC_MVFD_SBN_ALL4"


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def run_summary(*, output_root: Path, output: Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    rows = {}
    for fold in FOLDS:
        result = _json(
            output_root
            / METHOD
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = {
            "R0_CONTROL": result["frozen_matched_r0_control"],
            "FROZEN_MVFD": result["frozen_mvfd_sbn"],
            "MVSC_MVFD": result["metrics"],
            "DELTA_R0": result["delta_vs_r0_control"],
            "DELTA_MVFD": result["delta_vs_frozen_mvfd"],
            "BEST": result["best_epoch_diagnostics"],
            "AUX_VAL": result["auxiliary_validation_metrics"],
        }

    aggregate = {}
    for label in ("R0_CONTROL", "FROZEN_MVFD", "MVSC_MVFD"):
        aggregate[label] = {
            metric: _stats(
                [rows[f"fold_{fold}"][label][metric] for fold in FOLDS]
            )
            for metric in METRICS
        }
    for label in ("DELTA_R0", "DELTA_MVFD"):
        aggregate[label] = {
            metric: _stats(
                [rows[f"fold_{fold}"][label][metric] for fold in FOLDS]
            )
            for metric in METRICS
        }

    for key in (
        "epoch",
        "primary_ce",
        "f0_ce",
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
    ):
        aggregate[f"BEST_{key.upper()}"] = _stats(
            [float(rows[f"fold_{fold}"]["BEST"][key]) for fold in FOLDS]
        )

    macro_delta = aggregate["DELTA_MVFD"]["macro_f1"]["mean"]
    hard_delta = aggregate["DELTA_MVFD"]["hard_class_f1"]["mean"]
    worst_delta = aggregate["DELTA_MVFD"]["worst_class_f1"]["mean"]
    positive_macro = int(
        sum(
            rows[f"fold_{fold}"]["DELTA_MVFD"]["macro_f1"] > 0
            for fold in FOLDS
        )
    )
    criteria = {
        "macro_improved": macro_delta > 0.0,
        "macro_positive_at_least_3_of_5_folds": positive_macro >= 3,
        "hard_f1_improved": hard_delta > 0.0,
        "worst_f1_not_drop_more_than_1pp": worst_delta >= -0.01,
    }

    payload = {
        "format": "bilinear_lmmd.mvsc_mvfd_sbn.summary.v1",
        "protocol": "coffee17-mvsc-mvfd-sbn-v1",
        "scope": (
            "five-fold validation-only post-primary exploratory extension of "
            "frozen MVFD-SBN; same Coffee17 development folds; "
            "no outer-test inference"
        ),
        "primary_comparison": (
            "MVSC_MVFD_SBN_ALL4 R0-only inference vs frozen MVFD-SBN"
        ),
        "folds": rows,
        "aggregate": aggregate,
        "positive_macro_folds_vs_mvfd": positive_macro,
        "positive_hard_folds_vs_mvfd": int(
            sum(
                rows[f"fold_{fold}"]["DELTA_MVFD"]["hard_class_f1"] > 0
                for fold in FOLDS
            )
        ),
        "screening_gate": {
            "criteria": criteria,
            "decision": "PASS" if all(criteria.values()) else "FAIL",
        },
        "test_images_accessed": False,
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
