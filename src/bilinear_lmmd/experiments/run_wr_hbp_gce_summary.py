from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)
ARMS = ("WR_HBP_CE", "WR_HBP_GCE")
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
    initial_hashes = set()
    validation_hashes = set()
    all_classes: list[str] | None = None
    per_class_deltas: dict[str, list[float]] = {}
    hard_group_deltas: dict[str, list[float]] = {}
    targeted_totals: dict[str, int] = {}
    paired_totals = {
        "count": 0,
        "rescue": 0,
        "damage": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net_correct": 0,
    }

    for fold in FOLDS:
        pair_dir = output_root / f"fold_{fold}" / "seed42"
        result = _json(pair_dir / "pair_result.json")
        if result.get("protocol") != "coffee17-wr-hbp-gce-v1":
            raise RuntimeError(f"Fold {fold}: protocol tidak cocok.")
        if result.get("matched_model_initialization") is not True:
            raise RuntimeError(f"Fold {fold}: initialization tidak matched.")
        if result.get("matched_validation_rows") is not True:
            raise RuntimeError(f"Fold {fold}: validation rows tidak matched.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        if result.get("strict_determinism", {}).get("deterministic_algorithms") is not True:
            raise RuntimeError(f"Fold {fold}: strict determinism tidak aktif.")

        rows[f"fold_{fold}"] = result
        initial_hashes.add(result["initial_model_state_sha256"])
        validation_hashes.add(result["validation_identity_label_sha256"])

        for key in paired_totals:
            paired_totals[key] += int(result["paired_outcomes"][key])
        for key, value in result["targeted_confusion_delta"].items():
            targeted_totals[key] = targeted_totals.get(key, 0) + int(value)

        control_metrics = _json(pair_dir / "WR_HBP_CE/validation/metrics.json")
        candidate_metrics = _json(pair_dir / "WR_HBP_GCE/validation/metrics.json")
        if control_metrics["classes"] != candidate_metrics["classes"]:
            raise RuntimeError(f"Fold {fold}: urutan kelas berbeda.")
        if all_classes is None:
            all_classes = list(control_metrics["classes"])
        elif all_classes != list(control_metrics["classes"]):
            raise RuntimeError(f"Fold {fold}: urutan kelas lintas fold berbeda.")

        for name in all_classes:
            delta = (
                float(candidate_metrics["per_class"][name]["f1"])
                - float(control_metrics["per_class"][name]["f1"])
            )
            per_class_deltas.setdefault(name, []).append(delta)

        for group_name in control_metrics.get("hard_groups", {}):
            delta = (
                float(candidate_metrics["hard_groups"][group_name])
                - float(control_metrics["hard_groups"][group_name])
            )
            hard_group_deltas.setdefault(group_name, []).append(delta)

    if len(initial_hashes) != 1:
        raise RuntimeError("Initial model fingerprint berbeda antar-fold.")
    if len(validation_hashes) != 5:
        raise RuntimeError("Validation fold hash tidak unik 5/5.")

    aggregate = {}
    for label in (*ARMS, "DELTA_GCE_MINUS_CE"):
        aggregate[label] = {
            metric: _stats([
                float(rows[f"fold_{fold}"][label][metric])
                for fold in FOLDS
            ])
            for metric in METRICS
        }

    positive = {
        metric: sum(
            rows[f"fold_{fold}"]["DELTA_GCE_MINUS_CE"][metric] > 0.0
            for fold in FOLDS
        )
        for metric in METRICS
    }

    delta = aggregate["DELTA_GCE_MINUS_CE"]
    gate = {
        "hard_mean_positive": delta["hard_class_f1"]["mean"] > 0.0,
        "hard_positive_folds_at_least_3": positive["hard_class_f1"] >= 3,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
        "macro_mean_nonnegative": delta["macro_f1"]["mean"] >= 0.0,
        "targeted_confusions_nonincreasing": targeted_totals.get("TOTAL", 0) <= 0,
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.wr_hbp_gce.summary.v1",
        "protocol": "coffee17-wr-hbp-gce-v1",
        "scope": (
            "strict-deterministic five-fold seed42 matched WR-HBP comparison; "
            "architecture, initialization, data, optimizer and scheduler identical; "
            "only CE versus top-2 GCE-LS differs; outer test untouched"
        ),
        "folds": rows,
        "aggregate": aggregate,
        "positive_delta_folds": positive,
        "paired_prediction_outcomes": paired_totals,
        "per_class_delta": {
            name: _stats(values) for name, values in per_class_deltas.items()
        },
        "hard_group_delta": {
            name: _stats(values) for name, values in hard_group_deltas.items()
        },
        "targeted_confusion_delta_total": targeted_totals,
        "screening_gate": gate,
        "paired_fold_comparison": True,
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
