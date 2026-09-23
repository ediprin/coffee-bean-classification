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
            / "MVFD_SBN_ALL4"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = {
            "R0_CONTROL": result["frozen_matched_r0_control"],
            "MVFD_SBN_ALL4": result["metrics"],
            "DELTA": result["delta_vs_r0_control"],
            "BEST": result["best_epoch_diagnostics"],
            "AUX_VAL": result["auxiliary_validation_metrics"],
        }

    aggregate = {
        "R0_CONTROL": {
            metric: _stats(
                [rows[f"fold_{fold}"]["R0_CONTROL"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "MVFD_SBN_ALL4": {
            metric: _stats(
                [rows[f"fold_{fold}"]["MVFD_SBN_ALL4"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "DELTA": {
            metric: _stats(
                [rows[f"fold_{fold}"]["DELTA"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
    }

    for key in (
        "epoch",
        "primary_ce",
        "f0_ce",
        "feature_loss",
        "weighted_feature_loss",
        "r0_teacher_cos",
        "r0_teacher_l2",
        "teacher_norm",
        "r0_norm",
        "c0_disagreement",
        "f0_disagreement",
        "w0_disagreement",
    ):
        aggregate[f"BEST_{key.upper()}"] = _stats(
            [float(rows[f"fold_{fold}"]["BEST"][key]) for fold in FOLDS]
        )

    payload = {
        "format": "bilinear_lmmd.mvfd_sbn.summary.v1",
        "protocol": "coffee17-mvfd-sbn-v1",
        "scope": (
            "five-fold validation-only post-primary exploratory explicit "
            "feature-transfer experiment; frozen matched R0 controls; "
            "no outer-test inference"
        ),
        "primary_comparison": (
            "MVFD_SBN_ALL4 R0-only inference vs frozen matched R0_CONTROL"
        ),
        "folds": rows,
        "aggregate": aggregate,
        "positive_macro_folds": int(
            sum(
                rows[f"fold_{fold}"]["DELTA"]["macro_f1"] > 0
                for fold in FOLDS
            )
        ),
        "positive_hard_folds": int(
            sum(
                rows[f"fold_{fold}"]["DELTA"]["hard_class_f1"] > 0
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
