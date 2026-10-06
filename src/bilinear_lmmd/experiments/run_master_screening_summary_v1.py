from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from bilinear_lmmd.experiments.run_master_screening_v1 import MASTER_CANDIDATES


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _stats(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(output_root: Path, output: Path) -> dict:
    fold_payloads = []
    for fold in range(1, 6):
        path = output_root / f"fold_{fold}" / "seed42" / "fold_result.json"
        if not path.is_file():
            raise FileNotFoundError(f"Fold result tidak ditemukan: {path}")
        fold_payloads.append(json.loads(path.read_text(encoding="utf-8")))

    commits = {item["git_commit"] for item in fold_payloads}
    if len(commits) != 1:
        raise RuntimeError(f"Scientific commit berbeda antar fold: {commits}")

    candidates = sorted(
        set.intersection(*(set(item["candidates"]) for item in fold_payloads))
    )
    aggregate = {}
    for candidate in candidates:
        metric_summary = {}
        delta_summary = {}
        positive = {}
        for metric in METRICS:
            values = [
                float(item["results"][candidate]["metrics"][metric])
                for item in fold_payloads
            ]
            deltas = [
                float(item["results"][candidate]["delta_vs_anchor"][metric])
                for item in fold_payloads
            ]
            metric_summary[metric] = _stats(values)
            delta_summary[metric] = _stats(deltas)
            positive[metric] = sum(value > 0.0 for value in deltas)

        params = {
            int(item["results"][candidate]["parameter_count"])
            for item in fold_payloads
        }
        if len(params) != 1:
            raise RuntimeError(f"Parameter count {candidate} berbeda antar fold.")

        paired = [
            item["results"][candidate].get("paired_outcomes")
            for item in fold_payloads
            if item["results"][candidate].get("paired_outcomes") is not None
        ]
        result = {
            "anchor": MASTER_CANDIDATES[candidate]["anchor"],
            "family": MASTER_CANDIDATES[candidate]["family"],
            "metrics": metric_summary,
            "delta_vs_anchor": delta_summary,
            "positive_delta_folds": positive,
            "parameter_count": params.pop(),
            "paired_outcomes": (
                {
                    key: sum(int(row[key]) for row in paired)
                    for key in (
                        "count",
                        "rescue",
                        "damage",
                        "both_correct",
                        "both_wrong",
                        "net_correct",
                    )
                }
                if paired
                else None
            ),
        }
        gates = [
            item["results"][candidate].get("wavelet_gate_at_best")
            for item in fold_payloads
            if item["results"][candidate].get("wavelet_gate_at_best") is not None
        ]
        if gates:
            result["wavelet_gate_at_best"] = _stats([float(v) for v in gates])
        aggregate[candidate] = result

    payload = {
        "format": "bilinear_lmmd.master_screening.aggregate.v1",
        "protocol": fold_payloads[0]["protocol"],
        "git_commit": next(iter(commits)),
        "seed": 42,
        "folds": [1, 2, 3, 4, 5],
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
            "candidate", "family", "anchor", "macro_f1", "delta_macro_f1",
            "macro_positive_folds", "hard_f1", "delta_hard_f1",
            "worst_f1", "delta_worst_f1", "accuracy",
            "balanced_accuracy", "parameters", "rescue", "damage", "net_correct",
        ])
        for candidate in candidates:
            item = aggregate[candidate]
            paired = item["paired_outcomes"] or {}
            writer.writerow([
                candidate,
                item["family"],
                item["anchor"],
                item["metrics"]["macro_f1"]["mean"],
                item["delta_vs_anchor"]["macro_f1"]["mean"],
                item["positive_delta_folds"]["macro_f1"],
                item["metrics"]["hard_class_f1"]["mean"],
                item["delta_vs_anchor"]["hard_class_f1"]["mean"],
                item["metrics"]["worst_class_f1"]["mean"],
                item["delta_vs_anchor"]["worst_class_f1"]["mean"],
                item["metrics"]["accuracy"]["mean"],
                item["metrics"]["balanced_accuracy"]["mean"],
                item["parameter_count"],
                paired.get("rescue", 0),
                paired.get("damage", 0),
                paired.get("net_correct", 0),
            ])

    ranking = sorted(
        (
            (
                candidate,
                aggregate[candidate]["delta_vs_anchor"]["macro_f1"]["mean"],
                aggregate[candidate]["positive_delta_folds"]["macro_f1"],
            )
            for candidate in candidates
            if candidate not in {"B0", "B1"}
        ),
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )
    print("\n=== MASTER RANKING BY MEAN DELTA MACRO-F1 ===")
    for candidate, delta, positive_folds in ranking:
        item = aggregate[candidate]
        print(
            f"{candidate:>2} [{item['family']}] vs {item['anchor']}: "
            f"dMacro={delta:+.4%}, positive={positive_folds}/5, "
            f"dHard={item['delta_vs_anchor']['hard_class_f1']['mean']:+.4%}, "
            f"dWorst={item['delta_vs_anchor']['worst_class_f1']['mean']:+.4%}, "
            f"params={item['parameter_count']:,}"
        )
    print(f"\nSAVED: {output}")
    print(f"SAVED: {csv_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate unified five-fold Coffee17 screening V1"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.output_root.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
