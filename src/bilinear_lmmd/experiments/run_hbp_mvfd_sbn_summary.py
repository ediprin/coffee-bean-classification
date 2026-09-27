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
            / "HBP_MVFD_SBN_ALL4"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = result

    aggregate = {}
    for label in (
        "FROZEN_GAP_R0_CONTROL",
        "HBP_R0_CONTROL",
        "HBP_MVFD_SBN_ALL4",
        "DELTA_HBP_VS_GAP",
        "DELTA_MVFD_ON_HBP",
        "DELTA_COMBINED_VS_GAP",
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
        "teacher_norm",
        "r0_norm",
        "c0_disagreement",
        "f0_disagreement",
        "w0_disagreement",
    ):
        aggregate[f"BEST_{key.upper()}"] = _stats(
            [
                float(rows[f"fold_{fold}"]["best_epoch_diagnostics"][key])
                for fold in FOLDS
            ]
        )

    mvfd_on_hbp = aggregate["DELTA_MVFD_ON_HBP"]
    gate = {
        "macro_mean_positive": mvfd_on_hbp["macro_f1"]["mean"] > 0.0,
        "macro_positive_folds_at_least_3": sum(
            rows[f"fold_{fold}"]["DELTA_MVFD_ON_HBP"]["macro_f1"] > 0.0
            for fold in FOLDS
        ) >= 3,
        "hard_mean_positive": mvfd_on_hbp["hard_class_f1"]["mean"] > 0.0,
        "worst_mean_not_below_minus_1pp": (
            mvfd_on_hbp["worst_class_f1"]["mean"] >= -0.01
        ),
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.hbp_mvfd_sbn.summary.v1",
        "protocol": "coffee17-hbp-mvfd-sbn-v1",
        "scope": (
            "five-fold validation-only post-primary exploratory factorial slice; "
            "matched HBP R0-only control versus HBP+MVFD-SBN; frozen GAP R0 "
            "included only as an already-established fold-matched reference; "
            "no outer-test inference"
        ),
        "primary_comparison": "HBP_MVFD_SBN_ALL4 vs HBP_R0_CONTROL",
        "secondary_comparisons": [
            "HBP_R0_CONTROL vs FROZEN_GAP_R0_CONTROL",
            "HBP_MVFD_SBN_ALL4 vs FROZEN_GAP_R0_CONTROL",
        ],
        "folds": rows,
        "aggregate": aggregate,
        "positive_macro_folds_mvfd_on_hbp": int(
            sum(
                rows[f"fold_{fold}"]["DELTA_MVFD_ON_HBP"]["macro_f1"] > 0
                for fold in FOLDS
            )
        ),
        "positive_hard_folds_mvfd_on_hbp": int(
            sum(
                rows[f"fold_{fold}"]["DELTA_MVFD_ON_HBP"]["hard_class_f1"] > 0
                for fold in FOLDS
            )
        ),
        "screening_gate": gate,
        "interaction_note": (
            "A full HBP x MVFD interaction requires the frozen GAP+MVFD result "
            "from the earlier MVFD-SBN experiment. This package intentionally "
            "does not reconstruct or hard-code that result."
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
