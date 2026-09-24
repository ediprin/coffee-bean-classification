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
            / "AUXCE_SBN_CONTROL"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = {
            "R0_CONTROL": result["frozen_matched_r0_control"],
            "MVFD_SBN": result["frozen_mvfd_sbn"],
            "AUXCE_SBN_CONTROL": result["metrics"],
            "CONTROL_MINUS_R0": result["delta_vs_r0_control"],
            "CONTROL_MINUS_MVFD": result["delta_vs_mvfd_sbn"],
            "BEST": result["best_epoch_diagnostics"],
        }

    aggregate = {}
    for label in (
        "R0_CONTROL",
        "MVFD_SBN",
        "AUXCE_SBN_CONTROL",
        "CONTROL_MINUS_R0",
        "CONTROL_MINUS_MVFD",
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
        "f0_disagreement",
    ):
        aggregate[f"BEST_{key.upper()}"] = _stats(
            [float(rows[f"fold_{fold}"]["BEST"][key]) for fold in FOLDS]
        )

    payload = {
        "format": "bilinear_lmmd.mvfd_sbn.causal_ablation_summary.v1",
        "protocol": "coffee17-auxce-sbn-control-v1",
        "scope": (
            "five-fold validation-only causal ablation; lambda_feat=0 while "
            "keeping auxiliary-head CE and SBN fixed; no outer-test inference"
        ),
        "causal_comparison": "MVFD_SBN vs AUXCE_SBN_CONTROL",
        "folds": rows,
        "aggregate": aggregate,
        "control_positive_macro_folds_vs_r0": int(
            sum(
                rows[f"fold_{fold}"]["CONTROL_MINUS_R0"]["macro_f1"] > 0
                for fold in FOLDS
            )
        ),
        "mvfd_beats_control_macro_folds": int(
            sum(
                rows[f"fold_{fold}"]["CONTROL_MINUS_MVFD"]["macro_f1"] < 0
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
