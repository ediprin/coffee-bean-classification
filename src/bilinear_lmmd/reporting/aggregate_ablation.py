from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SUMMARY_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "worst_class_f1",
    "hard_class_f1",
)


def _paired_summary(baseline: list[float], candidate: list[float]) -> dict:
    deltas = [new - old for old, new in zip(baseline, candidate)]
    return {
        "baseline_mean": statistics.mean(baseline),
        "candidate_mean": statistics.mean(candidate),
        "delta_mean": statistics.mean(deltas),
        "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "improved_seeds": sum(delta > 0 for delta in deltas),
        "total_seeds": len(deltas),
        "deltas": deltas,
    }


def aggregate(baseline_paths: list[Path], candidate_paths: list[Path]) -> dict:
    if len(baseline_paths) != len(candidate_paths) or not baseline_paths:
        raise ValueError("Jumlah report baseline dan candidate harus sama dan tidak kosong.")
    baseline = [json.loads(path.read_text(encoding="utf-8")) for path in baseline_paths]
    candidate = [json.loads(path.read_text(encoding="utf-8")) for path in candidate_paths]
    classes = baseline[0]["classes"]
    if any(report["classes"] != classes for report in baseline + candidate):
        raise ValueError("Urutan kelas seluruh report harus sama.")

    return {
        "summary": {
            key: _paired_summary(
                [report[key] for report in baseline],
                [report[key] for report in candidate],
            )
            for key in SUMMARY_KEYS
        },
        "hard_groups": {
            group: _paired_summary(
                [report["hard_groups"][group] for report in baseline],
                [report["hard_groups"][group] for report in candidate],
            )
            for group in baseline[0]["hard_groups"]
        },
        "per_class": {
            name: _paired_summary(
                [report["per_class"][name]["f1"] for report in baseline],
                [report["per_class"][name]["f1"] for report in candidate],
            )
            for name in classes
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agregasi ablation berpasangan lintas seed")
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
