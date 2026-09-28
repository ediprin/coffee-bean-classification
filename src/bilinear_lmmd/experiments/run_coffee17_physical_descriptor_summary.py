from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_consistency(pairwise_by_fold: dict[int, dict]) -> dict:
    feature_rows: dict[str, list[dict]] = defaultdict(list)
    for fold, payload in pairwise_by_fold.items():
        for row in payload["results"]:
            feature_rows[row["feature"]].append({"fold": fold, **row})

    out = {}
    for feature, rows in sorted(feature_rows.items()):
        deltas = [float(r["cliffs_delta_a_minus_b"]) for r in rows]
        med_deltas = [float(r["median_delta_a_minus_b"]) for r in rows]
        q_sig = [
            bool(r["q_value_bh_within_hypothesis"] < 0.05)
            for r in rows
        ]
        medium_sig = [
            bool(r["medium_effect_and_fdr"])
            for r in rows
        ]
        pos = sum(d > 0 for d in deltas)
        neg = sum(d < 0 for d in deltas)
        zero = len(deltas) - pos - neg
        direction_consistent = max(pos, neg) >= 4
        out[feature] = {
            "folds": len(rows),
            "cliffs_delta_mean": statistics.mean(deltas),
            "cliffs_delta_median": statistics.median(deltas),
            "cliffs_delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "median_delta_mean": statistics.mean(med_deltas),
            "positive_direction_folds": pos,
            "negative_direction_folds": neg,
            "zero_direction_folds": zero,
            "q_lt_0p05_folds": sum(q_sig),
            "medium_effect_and_fdr_folds": sum(medium_sig),
            "exploratory_consistent_candidate": bool(
                direction_consistent
                and abs(statistics.median(deltas)) >= 0.33
                and sum(medium_sig) >= 3
            ),
        }
    return out


def _shape_kw_consistency(shape_by_fold: dict[int, dict]) -> dict:
    rows: dict[str, list[dict]] = defaultdict(list)
    for fold, payload in shape_by_fold.items():
        for row in payload["kruskal_wallis"]:
            rows[row["feature"]].append({"fold": fold, **row})

    out = {}
    for feature, vals in sorted(rows.items()):
        qs = [float(v["q_value_bh_shape_family"]) for v in vals]
        hs = [float(v["h_statistic"]) for v in vals]
        out[feature] = {
            "q_lt_0p05_folds": sum(q < 0.05 for q in qs),
            "h_statistic_mean": statistics.mean(hs),
            "h_statistic_median": statistics.median(hs),
            "exploratory_consistent_group_difference": sum(q < 0.05 for q in qs) >= 4,
        }
    return out


def run_summary(*, audit_root: Path, output: Path) -> dict:
    audit_root = Path(audit_root).expanduser().resolve()

    folds = {}
    sour = {}
    black = {}
    insect = {}
    shape = {}
    qc_total = 0

    for fold in FOLDS:
        payload = _load(audit_root / f"fold_{fold}" / "fold_audit.json")
        if payload.get("outer_test_accessed") is not False:
            raise RuntimeError(f"fold {fold}: outer test flag invalid")
        if payload.get("training_executed") is not False:
            raise RuntimeError(f"fold {fold}: training flag invalid")
        if payload.get("descriptor_extraction_uses_labels") is not False:
            raise RuntimeError(f"fold {fold}: descriptor extraction must be label-free")

        folds[f"fold_{fold}"] = {
            "sample_count": payload["sample_count"],
            "class_count": payload["class_count"],
            "mask_qc_fail_count": payload["mask_qc_fail_count"],
        }
        qc_total += int(payload["mask_qc_fail_count"])

        hypotheses = payload["hypotheses"]
        sour[fold] = hypotheses["full_vs_partial_sour"]
        black[fold] = hypotheses["full_vs_partial_black"]
        insect[fold] = hypotheses["severe_vs_slight_insect"]
        shape[fold] = hypotheses["shape_family"]

    payload = {
        "format": "bilinear_lmmd.coffee17.physical_descriptor_audit.summary.v1",
        "scope": (
            "five fold-specific development-only descriptor audits; each fold uses "
            "only its train+val materialization; no fold's locked test directory is "
            "materialized; no neural training"
        ),
        "folds": folds,
        "mask_qc_fail_count_across_fold_runs": qc_total,
        "full_vs_partial_sour": {
            "feature_consistency": _feature_consistency(sour),
            "interpretation_boundary": (
                "Warm/Lab features are color-distribution proxies. This audit does "
                "not claim a validated sour-region segmentation or direct SCAA "
                "coverage measurement."
            ),
        },
        "full_vs_partial_black": {
            "feature_consistency": _feature_consistency(black),
            "interpretation_boundary": (
                "L* coverage curves are fixed luminance proxies, not a calibrated "
                "black-defect segmentation mask."
            ),
        },
        "severe_vs_slight_insect": {
            "feature_consistency": _feature_consistency(insect),
            "interpretation_boundary": (
                "Dark connected-component counts are topology proxies, not exact "
                "insect perforation counts; a single Coffee17 view can miss holes "
                "on the opposite bean side."
            ),
        },
        "shape_family": {
            "kruskal_wallis_consistency": _shape_kw_consistency(shape),
            "pairwise_consistency": {},
        },
        "outer_test_accessed": False,
        "training_executed": False,
    }

    pair_keys = sorted(next(iter(shape.values()))["pairwise"].keys())
    for key in pair_keys:
        payload["shape_family"]["pairwise_consistency"][key] = _feature_consistency({
            fold: shape[fold]["pairwise"][key]
            for fold in FOLDS
        })

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(audit_root=args.audit_root, output=args.output)


if __name__ == "__main__":
    main()
