from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from bilinear_lmmd.modeling.representation_screening import (
    REPRESENTATION_CANDIDATES,
)


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

    commit_set = {item["git_commit"] for item in fold_payloads}
    if len(commit_set) != 1:
        raise RuntimeError(f"Scientific commit berbeda antar fold: {commit_set}")

    candidates = sorted(
        set.intersection(*(set(item["candidates"]) for item in fold_payloads))
    )
    aggregate = {}
    for candidate in candidates:
        anchor = REPRESENTATION_CANDIDATES[candidate]["anchor"]
        metrics = {}
        deltas = {}
        positive = {}
        for metric in METRICS:
            values = [
                float(item["results"][candidate]["metrics"][metric])
                for item in fold_payloads
            ]
            delta_values = [
                float(item["results"][candidate]["delta_vs_anchor"][metric])
                for item in fold_payloads
            ]
            metrics[metric] = _stats(values)
            deltas[metric] = _stats(delta_values)
            positive[metric] = sum(value > 0.0 for value in delta_values)

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
        aggregate[candidate] = {
            "anchor": anchor,
            "loss": REPRESENTATION_CANDIDATES[candidate]["loss"],
            "metrics": metrics,
            "delta_vs_anchor": deltas,
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

    payload = {
        "format": "bilinear_lmmd.representation_screening.aggregate.v1",
        "protocol": fold_payloads[0]["protocol"],
        "git_commit": next(iter(commit_set)),
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
            "candidate", "anchor", "loss", "macro_f1", "delta_macro_f1",
            "macro_positive_folds", "hard_f1", "delta_hard_f1",
            "worst_f1", "delta_worst_f1", "accuracy",
            "balanced_accuracy", "parameters", "rescue", "damage",
            "net_correct",
        ])
        for candidate in candidates:
            item = aggregate[candidate]
            paired = item["paired_outcomes"] or {}
            writer.writerow([
                candidate,
                item["anchor"],
                item["loss"],
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

    print(json.dumps(payload, indent=2))
    print(f"SAVED: {output}")
    print(f"SAVED: {csv_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate five-fold Coffee17 representation screening V1"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.output_root.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
