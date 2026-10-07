from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from bilinear_lmmd.experiments.run_frequency_screening_v1 import METRICS


def _stats(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(output_root: Path, output: Path) -> dict:
    folds = []
    for fold in range(1, 6):
        path = (
            output_root
            / f"fold_{fold}"
            / "seed42"
            / "focused_fold_result.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Focused fold result hilang: {path}")
        folds.append(json.loads(path.read_text(encoding="utf-8")))

    commits = {item["git_commit"] for item in folds}
    if len(commits) != 1:
        raise RuntimeError(f"Scientific commit berbeda antar fold: {commits}")

    candidates = sorted(
        set.intersection(*(set(item["candidates"]) for item in folds))
    )
    if candidates != ["C1", "C2"]:
        raise RuntimeError(f"Expected C1/C2 lengkap pada semua fold, dapat {candidates}")

    anchor_metrics = {
        metric: _stats(
            [float(item["historical_b0"]["metrics"][metric]) for item in folds]
        )
        for metric in METRICS
    }

    aggregate = {}
    for candidate in candidates:
        metrics = {}
        deltas = {}
        positive = {}
        for metric in METRICS:
            values = [
                float(item["results"][candidate]["metrics"][metric])
                for item in folds
            ]
            delta_values = [
                float(
                    item["results"][candidate]["delta_vs_historical_b0"][metric]
                )
                for item in folds
            ]
            metrics[metric] = _stats(values)
            deltas[metric] = _stats(delta_values)
            positive[metric] = sum(v > 0.0 for v in delta_values)

        params = {
            int(item["results"][candidate]["parameter_count"])
            for item in folds
        }
        if len(params) != 1:
            raise RuntimeError(f"Parameter count {candidate} berbeda antar fold.")

        paired = [item["results"][candidate]["paired_outcomes"] for item in folds]
        aggregate[candidate] = {
            "anchor": "B0",
            "metrics": metrics,
            "delta_vs_historical_b0": deltas,
            "positive_delta_folds": positive,
            "parameter_count": params.pop(),
            "paired_outcomes": {
                key: sum(int(row[key]) for row in paired)
                for key in (
                    "count",
                    "rescue",
                    "damage",
                    "both_correct",
                    "both_wrong",
                    "net_correct",
                )
            },
        }

    historical_commits = {
        item["historical_b0"]["historical_commit"] for item in folds
    }
    if len(historical_commits) != 1:
        raise RuntimeError(
            f"Historical B0 commits berbeda antar fold: {historical_commits}"
        )

    payload = {
        "format": "bilinear_lmmd.focused_c1c2.aggregate.v1",
        "protocol": folds[0]["protocol"],
        "git_commit": next(iter(commits)),
        "historical_b0_commit": next(iter(historical_commits)),
        "seed": 42,
        "folds": [1, 2, 3, 4, 5],
        "historical_b0": {
            "metrics": anchor_metrics,
            "parameter_count": int(folds[0]["historical_b0"]["parameter_count"]),
        },
        "candidates": candidates,
        "aggregate": aggregate,
        "outer_test_accessed": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "candidate",
            "macro_f1",
            "delta_macro_f1",
            "macro_positive_folds",
            "hard_f1",
            "delta_hard_f1",
            "worst_f1",
            "delta_worst_f1",
            "accuracy",
            "balanced_accuracy",
            "parameters",
            "rescue",
            "damage",
            "net_correct",
        ])
        for candidate in candidates:
            item = aggregate[candidate]
            paired = item["paired_outcomes"]
            writer.writerow([
                candidate,
                item["metrics"]["macro_f1"]["mean"],
                item["delta_vs_historical_b0"]["macro_f1"]["mean"],
                item["positive_delta_folds"]["macro_f1"],
                item["metrics"]["hard_class_f1"]["mean"],
                item["delta_vs_historical_b0"]["hard_class_f1"]["mean"],
                item["metrics"]["worst_class_f1"]["mean"],
                item["delta_vs_historical_b0"]["worst_class_f1"]["mean"],
                item["metrics"]["accuracy"]["mean"],
                item["metrics"]["balanced_accuracy"]["mean"],
                item["parameter_count"],
                paired["rescue"],
                paired["damage"],
                paired["net_correct"],
            ])

    print("\n=== FOCUSED C1/C2 RESULT ===")
    print(
        "Historical B0 Macro-F1:",
        f"{anchor_metrics['macro_f1']['mean']:.4%}",
    )
    for candidate in candidates:
        item = aggregate[candidate]
        print(
            f"{candidate}: "
            f"Macro={item['metrics']['macro_f1']['mean']:.4%} "
            f"dMacro={item['delta_vs_historical_b0']['macro_f1']['mean']:+.4%} "
            f"positive={item['positive_delta_folds']['macro_f1']}/5 "
            f"dHard={item['delta_vs_historical_b0']['hard_class_f1']['mean']:+.4%} "
            f"dWorst={item['delta_vs_historical_b0']['worst_class_f1']['mean']:+.4%} "
            f"net_correct={item['paired_outcomes']['net_correct']:+d} "
            f"params={item['parameter_count']:,}"
        )
    print("SAVED:", output)
    print("SAVED:", csv_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate focused Coffee17 C1/C2 five-fold screening."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.output_root.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
