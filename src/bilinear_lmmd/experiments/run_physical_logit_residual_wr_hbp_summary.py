from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
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

    folds = {}
    paired_totals = {
        "count": 0,
        "rescue": 0,
        "damage": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net_correct": 0,
    }
    per_class_deltas: dict[str, list[float]] = defaultdict(list)
    hard_group_deltas: dict[str, list[float]] = defaultdict(list)
    residual_weights: dict[str, list[float]] = defaultdict(list)
    qc_train = 0
    qc_val = 0
    active_counts = set()
    determinism_records = []

    for fold in FOLDS:
        fold_root = output_root / f"fold_{fold}" / "seed42"
        result = _json(fold_root / "pair_result.json")

        if result.get("protocol") != "coffee17-physical-logit-residual-wr-hbp-v1":
            raise RuntimeError(f"Fold {fold}: protocol tidak cocok.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        if result.get("base_model_frozen_during_residual_fit") is not True:
            raise RuntimeError(f"Fold {fold}: base tidak frozen saat residual fit.")
        if result.get("insect_topology_enabled") is not False:
            raise RuntimeError(f"Fold {fold}: insect topology tidak boleh aktif.")
        if result.get("residual_optimizer_success") is not True:
            raise RuntimeError(f"Fold {fold}: residual optimizer gagal.")
        if float(result["zero_residual_initial_logit_max_abs_difference"]) != 0.0:
            raise RuntimeError(f"Fold {fold}: zero residual tidak identik dengan WR.")

        folds[f"fold_{fold}"] = result
        active_counts.add(int(result["active_residual_parameter_count"]))
        qc_train += int(result["train_descriptor_qc_fail_count"])
        qc_val += int(result["val_descriptor_qc_fail_count"])
        determinism_records.append(result["strict_determinism"])

        for key in paired_totals:
            paired_totals[key] += int(result["paired_outcomes"][key])

        wr_metrics = _json(
            fold_root
            / "WR_PDR_HBP"
            / "WR_HBP_validation"
            / "metrics.json"
        )
        candidate_metrics = _json(
            fold_root
            / "WR_PDR_HBP"
            / "WR_PDR_HBP_validation"
            / "metrics.json"
        )
        if wr_metrics["classes"] != candidate_metrics["classes"]:
            raise RuntimeError(f"Fold {fold}: class order berbeda.")

        for class_name in wr_metrics["classes"]:
            per_class_deltas[class_name].append(
                float(candidate_metrics["per_class"][class_name]["f1"])
                - float(wr_metrics["per_class"][class_name]["f1"])
            )

        for group_name in wr_metrics.get("hard_groups", {}):
            hard_group_deltas[group_name].append(
                float(candidate_metrics["hard_groups"][group_name])
                - float(wr_metrics["hard_groups"][group_name])
            )

        fit = _json(
            fold_root
            / "WR_PDR_HBP"
            / "physical_residual_fit.json"
        )
        for row in fit["active_weights"]:
            key = f"{row['class']}::{row['feature']}"
            residual_weights[key].append(float(row["weight"]))

    if active_counts != {38}:
        raise RuntimeError(f"Active residual parameter count berubah: {active_counts}")

    for index, record in enumerate(determinism_records, start=1):
        if record.get("deterministic_algorithms") is not True:
            raise RuntimeError(f"Fold {index}: deterministic algorithms tidak aktif.")
        if record.get("cudnn_benchmark") is not False:
            raise RuntimeError(f"Fold {index}: cudnn benchmark harus false.")
        if record.get("cudnn_deterministic") is not True:
            raise RuntimeError(f"Fold {index}: cudnn deterministic harus true.")
        if record.get("cublas_workspace_config") != ":4096:8":
            raise RuntimeError(f"Fold {index}: CUBLAS workspace config salah.")

    aggregate = {}
    for label in ("WR_HBP", "WR_PDR_HBP", "DELTA_WR_PDR_MINUS_WR"):
        aggregate[label] = {
            metric: _stats([
                float(folds[f"fold_{fold}"][label][metric])
                for fold in FOLDS
            ])
            for metric in METRICS
        }

    positive = {
        metric: sum(
            folds[f"fold_{fold}"]["DELTA_WR_PDR_MINUS_WR"][metric] > 0.0
            for fold in FOLDS
        )
        for metric in METRICS
    }

    delta = aggregate["DELTA_WR_PDR_MINUS_WR"]
    gate = {
        "macro_mean_positive": delta["macro_f1"]["mean"] > 0.0,
        "macro_positive_folds_at_least_3": positive["macro_f1"] >= 3,
        "hard_mean_nonnegative": delta["hard_class_f1"]["mean"] >= 0.0,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.physical_logit_residual_wr_hbp.summary.v1",
        "protocol": "coffee17-physical-logit-residual-wr-hbp-v1",
        "scope": (
            "five fold-specific seed42 development comparisons; each fold trains "
            "one strict-deterministic WR-HBP base, freezes that exact checkpoint, "
            "fits a 38-parameter CPU float64 masked physical-logit residual using "
            "source/train only, and compares paired source/val predictions; "
            "outer test untouched"
        ),
        "folds": folds,
        "aggregate": aggregate,
        "positive_delta_folds": positive,
        "paired_prediction_outcomes": paired_totals,
        "per_class_delta": {
            name: _stats(values)
            for name, values in sorted(per_class_deltas.items())
        },
        "hard_group_delta": {
            name: _stats(values)
            for name, values in sorted(hard_group_deltas.items())
        },
        "residual_weight": {
            name: _stats(values)
            for name, values in sorted(residual_weights.items())
        },
        "active_residual_parameter_count": 38,
        "descriptor_qc_failures": {
            "train_occurrences_across_fold_runs": qc_train,
            "val_occurrences_across_fold_runs": qc_val,
            "policy": (
                "exclude QC-fail rows from residual fitting; preserve exact WR-HBP "
                "logits for QC-fail validation rows"
            ),
        },
        "strict_determinism_verified_all_folds": True,
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
