from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline["classes"] != candidate["classes"]:
        raise ValueError("Urutan kelas kedua laporan berbeda.")

    summary_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "worst_class_f1",
        "hard_class_f1",
    )
    summary = {
        key: {
            "baseline": baseline[key],
            "candidate": candidate[key],
            "delta": candidate[key] - baseline[key],
        }
        for key in summary_keys
    }
    per_class = {
        name: {
            "baseline_f1": baseline["per_class"][name]["f1"],
            "candidate_f1": candidate["per_class"][name]["f1"],
            "delta_f1": (
                candidate["per_class"][name]["f1"]
                - baseline["per_class"][name]["f1"]
            ),
        }
        for name in baseline["classes"]
    }
    return {
        "summary": summary,
        "hard_groups": {
            name: {
                "baseline": score,
                "candidate": candidate["hard_groups"][name],
                "delta": candidate["hard_groups"][name] - score,
            }
            for name, score in baseline["hard_groups"].items()
        },
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandingkan laporan GAP dengan bilinear/HBP")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
