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
    rows = {}
    for fold in FOLDS:
        result = _json(
            Path(output_root)
            / "MVCE_SBN_ALL4"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = {
            "R0_CONTROL": result["frozen_matched_r0_control"],
            "MVCE_SBN_ALL4": result["metrics"],
            "DELTA": result["delta_vs_r0_control"],
            "SBN_BEST_CE": result["best_epoch_training_ce"],
            "FAILED_MVCE_BEST_CE": result["failed_mvce_reference"],
        }

    aggregate = {
        "R0_CONTROL": {
            metric: _stats(
                [rows[f"fold_{fold}"]["R0_CONTROL"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "MVCE_SBN_ALL4": {
            metric: _stats(
                [rows[f"fold_{fold}"]["MVCE_SBN_ALL4"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "DELTA": {
            metric: _stats(
                [rows[f"fold_{fold}"]["DELTA"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "F0_CE_AT_BEST_EPOCH": _stats(
            [rows[f"fold_{fold}"]["SBN_BEST_CE"]["f0_ce"] for fold in FOLDS]
        ),
        "FAILED_MVCE_F0_CE_AT_BEST_EPOCH": _stats(
            [rows[f"fold_{fold}"]["FAILED_MVCE_BEST_CE"]["f0_ce"] for fold in FOLDS]
        ),
    }

    positive_macro_folds = sum(
        rows[f"fold_{fold}"]["DELTA"]["macro_f1"] > 0
        for fold in FOLDS
    )
    payload = {
        "format": "bilinear_lmmd.shared_multiview.sbn_summary.v1",
        "protocol": "coffee17-shared-multiview-sbn-v1",
        "scope": (
            "five-fold validation-only targeted normalization experiment; "
            "comparison uses the frozen matched R0 controls from the immediately "
            "preceding shared-multiview experiment; no outer-test inference"
        ),
        "primary_comparison": "MVCE_SBN_ALL4 vs frozen matched R0_CONTROL",
        "folds": rows,
        "aggregate": aggregate,
        "positive_macro_folds": int(positive_macro_folds),
        "diagnostic": {
            "question": (
                "Does per-auxiliary-mini-batch normalization remove the previous "
                "F0 CE collapse while preserving only R0 running statistics for deployment?"
            ),
            "old_f0_ce_mean": aggregate["FAILED_MVCE_F0_CE_AT_BEST_EPOCH"]["mean"],
            "new_f0_ce_mean": aggregate["F0_CE_AT_BEST_EPOCH"]["mean"],
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
