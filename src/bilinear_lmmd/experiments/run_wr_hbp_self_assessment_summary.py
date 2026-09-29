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
    rows: dict[str, dict] = {}
    validation_hashes = set()
    base_hashes = set()
    per_class_delta: dict[str, list[float]] = {}
    hard_group_delta: dict[str, list[float]] = {}
    confusion_delta_total: dict[str, int] = {}
    paired_totals = {
        "count": 0,
        "rescue": 0,
        "damage": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net_correct": 0,
    }
    reachable = 0
    base_errors = 0

    for fold in FOLDS:
        fold_root = output_root / f"fold_{fold}" / "seed42"
        result = _json(fold_root / "pair_result.json")
        if result.get("protocol") != "coffee17-wr-hbp-self-assessment-v1":
            raise RuntimeError(f"Fold {fold}: protocol tidak cocok.")
        if result.get("matched_validation_rows") is not True:
            raise RuntimeError(f"Fold {fold}: validation rows tidak matched.")
        if result.get("base_model_frozen_during_sar_training") is not True:
            raise RuntimeError(f"Fold {fold}: base tidak frozen.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        if result.get("strict_determinism", {}).get("deterministic_algorithms") is not True:
            raise RuntimeError(f"Fold {fold}: strict determinism tidak aktif.")
        if result["zero_residual_preflight"][
            "zero_residual_initial_logit_max_abs_difference"
        ] > 1.0e-7:
            raise RuntimeError(f"Fold {fold}: zero-residual preflight gagal.")

        rows[f"fold_{fold}"] = result
        validation_hashes.add(result["validation_identity_label_sha256"])
        base_hashes.add(result["base_best_checkpoint_sha256"])
        reachable += int(result["base_error_top5_reachable"])
        base_errors += int(result["base_error_count"])

        for key in paired_totals:
            paired_totals[key] += int(result["paired_outcomes"][key])
        for key, value in result["audited_pair_confusion_delta"].items():
            confusion_delta_total[key] = (
                confusion_delta_total.get(key, 0) + int(value)
            )

        base_metrics = _json(
            fold_root / "WR_HBP_BASE/validation/metrics.json"
        )
        sar_metrics = _json(
            fold_root / "WR_HBP_SAR/validation/metrics.json"
        )
        if base_metrics["classes"] != sar_metrics["classes"]:
            raise RuntimeError(f"Fold {fold}: urutan kelas berbeda.")

        for name in base_metrics["classes"]:
            delta = (
                float(sar_metrics["per_class"][name]["f1"])
                - float(base_metrics["per_class"][name]["f1"])
            )
            per_class_delta.setdefault(name, []).append(delta)

        for name in base_metrics.get("hard_groups", {}):
            delta = (
                float(sar_metrics["hard_groups"][name])
                - float(base_metrics["hard_groups"][name])
            )
            hard_group_delta.setdefault(name, []).append(delta)

    if len(validation_hashes) != 5:
        raise RuntimeError("Validation fold hash tidak unik 5/5.")
    if len(base_hashes) != 5:
        raise RuntimeError("Base checkpoint hash harus unik per fold.")

    aggregate = {}
    for label in ("WR_HBP_BASE", "WR_HBP_SAR", "DELTA_SAR_MINUS_BASE"):
        aggregate[label] = {
            metric: _stats([
                float(rows[f"fold_{fold}"][label][metric])
                for fold in FOLDS
            ])
            for metric in METRICS
        }

    positive = {
        metric: sum(
            rows[f"fold_{fold}"]["DELTA_SAR_MINUS_BASE"][metric] > 0.0
            for fold in FOLDS
        )
        for metric in METRICS
    }

    delta = aggregate["DELTA_SAR_MINUS_BASE"]
    gate = {
        "hard_mean_positive": delta["hard_class_f1"]["mean"] > 0.0,
        "hard_positive_folds_at_least_3": positive["hard_class_f1"] >= 3,
        "macro_mean_nonnegative": delta["macro_f1"]["mean"] >= 0.0,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
        "paired_rescue_at_least_damage": (
            paired_totals["rescue"] >= paired_totals["damage"]
        ),
        "audited_confusions_nonincreasing": (
            confusion_delta_total.get("TOTAL", 0) <= 0
        ),
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.wr_hbp_self_assessment.summary.v1",
        "protocol": "coffee17-wr-hbp-self-assessment-v1",
        "scope": (
            "five strict-deterministic development folds; WR-HBP base trained "
            "with the frozen recipe, then frozen; only a localized top-5 "
            "self-assessment residual is trained; outer test untouched"
        ),
        "aggregate": aggregate,
        "positive_delta_folds": positive,
        "paired_prediction_outcomes": paired_totals,
        "per_class_delta": {
            name: _stats(values)
            for name, values in per_class_delta.items()
        },
        "hard_group_delta": {
            name: _stats(values)
            for name, values in hard_group_delta.items()
        },
        "audited_pair_confusion_delta_total": confusion_delta_total,
        "base_error_top5_reachable": reachable,
        "base_error_count": base_errors,
        "base_error_top5_reachable_fraction": (
            reachable / base_errors if base_errors else None
        ),
        "screening_gate": gate,
        "outer_test_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
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
