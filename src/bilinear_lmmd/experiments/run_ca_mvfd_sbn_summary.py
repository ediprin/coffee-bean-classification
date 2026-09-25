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
            / "CA_MVFD_SBN_ALL4"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = {
            "R0_CONTROL": result["frozen_matched_r0_control"],
            "MVFD_SBN": result["frozen_mvfd_sbn"],
            "CA_MVFD_SBN": result["metrics"],
            "CA_MINUS_R0": result["delta_vs_r0_control"],
            "CA_MINUS_MVFD": result["delta_vs_mvfd_sbn"],
            "BEST": result["best_epoch_diagnostics"],
            "WEIGHTS_BY_CLASS": result["teacher_weights_by_class"],
        }

    aggregate = {}
    for label in (
        "R0_CONTROL",
        "MVFD_SBN",
        "CA_MVFD_SBN",
        "CA_MINUS_R0",
        "CA_MINUS_MVFD",
    ):
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
        "r0_teacher_cos",
        "r0_teacher_l2",
        "c0_teacher_weight",
        "f0_teacher_weight",
        "w0_teacher_weight",
        "teacher_weight_entropy",
        "teacher_weight_max",
        "f0_disagreement",
    ):
        aggregate[f"BEST_{key.upper()}"] = _stats(
            [float(rows[f"fold_{fold}"]["BEST"][key]) for fold in FOLDS]
        )

    class_names = list(rows["fold_1"]["WEIGHTS_BY_CLASS"])
    aggregate_class_weights = {}
    for class_name in class_names:
        aggregate_class_weights[class_name] = {}
        for arm in ("C0", "F0", "W0"):
            aggregate_class_weights[class_name][arm] = _stats(
                [
                    float(rows[f"fold_{fold}"]["WEIGHTS_BY_CLASS"][class_name][arm])
                    for fold in FOLDS
                ]
            )

    payload = {
        "format": "bilinear_lmmd.ca_mvfd_sbn.summary.v1",
        "protocol": "coffee17-ca-mvfd-sbn-v1",
        "scope": (
            "five-fold validation-only post-primary teacher-aggregation "
            "refinement; frozen R0 and equal-mean MVFD references; "
            "no outer-test inference"
        ),
        "primary_comparison": "CA_MVFD_SBN vs frozen MVFD_SBN",
        "folds": rows,
        "aggregate": aggregate,
        "aggregate_teacher_weights_by_class": aggregate_class_weights,
        "ca_beats_mvfd_macro_folds": int(
            sum(
                rows[f"fold_{fold}"]["CA_MINUS_MVFD"]["macro_f1"] > 0
                for fold in FOLDS
            )
        ),
        "ca_beats_mvfd_hard_folds": int(
            sum(
                rows[f"fold_{fold}"]["CA_MINUS_MVFD"]["hard_class_f1"] > 0
                for fold in FOLDS
            )
        ),
        "ca_positive_macro_folds_vs_r0": int(
            sum(
                rows[f"fold_{fold}"]["CA_MINUS_R0"]["macro_f1"] > 0
                for fold in FOLDS
            )
        ),
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
