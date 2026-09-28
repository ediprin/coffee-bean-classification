from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

from bilinear_lmmd.analysis.coffee17_physical_descriptors import (
    ALL_FEATURES,
    DARK_COMPONENT_FEATURES,
    LAB_FEATURES,
    L_COVERAGE_FEATURES,
    SHAPE_FEATURES,
    WARM_COVERAGE_FEATURES,
    extract_physical_descriptors,
)


SOUR_FEATURES = LAB_FEATURES + WARM_COVERAGE_FEATURES
BLACK_FEATURES = LAB_FEATURES + L_COVERAGE_FEATURES
INSECT_FEATURES = DARK_COMPONENT_FEATURES
SHAPE_CLASSES = ("Broken", "Cut", "Immature", "Shell", "Withered")


def _bh_qvalues(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    q = [1.0] * n
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        value = min(1.0, pvalues[idx] * n / rank)
        running = min(running, value)
        q[idx] = running
    return q


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Positive means x tends to be larger than y.
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float((np.count_nonzero(diff > 0) - np.count_nonzero(diff < 0)) / diff.size)


def _pairwise(
    rows: list[dict],
    *,
    class_a: str,
    class_b: str,
    features: tuple[str, ...],
) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    results = []
    pvalues = []
    for feature in features:
        a = np.asarray([float(r[feature]) for r in by_class[class_a]], dtype=float)
        b = np.asarray([float(r[feature]) for r in by_class[class_b]], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        stat = mannwhitneyu(a, b, alternative="two-sided")
        pvalues.append(float(stat.pvalue))
        results.append({
            "feature": feature,
            "class_a": class_a,
            "class_b": class_b,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "median_delta_a_minus_b": float(np.median(a) - np.median(b)),
            "cliffs_delta_a_minus_b": _cliffs_delta(a, b),
            "p_value": float(stat.pvalue),
        })

    for row, q in zip(results, _bh_qvalues(pvalues)):
        row["q_value_bh_within_hypothesis"] = q
        row["medium_effect_and_fdr"] = bool(
            q < 0.05 and abs(row["cliffs_delta_a_minus_b"]) >= 0.33
        )
    return {
        "class_a": class_a,
        "class_b": class_b,
        "feature_count": len(features),
        "results": results,
    }


def _shape_group(rows: list[dict]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["class"] in SHAPE_CLASSES:
            by_class[row["class"]].append(row)

    kw_rows = []
    pvalues = []
    for feature in SHAPE_FEATURES:
        groups = [
            np.asarray([float(r[feature]) for r in by_class[c]], dtype=float)
            for c in SHAPE_CLASSES
        ]
        stat = kruskal(*groups)
        pvalues.append(float(stat.pvalue))
        kw_rows.append({
            "feature": feature,
            "h_statistic": float(stat.statistic),
            "p_value": float(stat.pvalue),
            "class_medians": {
                c: float(np.median([
                    float(r[feature]) for r in by_class[c]
                ]))
                for c in SHAPE_CLASSES
            },
        })

    for row, q in zip(kw_rows, _bh_qvalues(pvalues)):
        row["q_value_bh_shape_family"] = q

    pairwise = {}
    for a, b in combinations(SHAPE_CLASSES, 2):
        key = f"{a}__vs__{b}"
        pairwise[key] = _pairwise(
            rows,
            class_a=a,
            class_b=b,
            features=SHAPE_FEATURES,
        )
    return {
        "classes": list(SHAPE_CLASSES),
        "kruskal_wallis": kw_rows,
        "pairwise": pairwise,
    }


def _class_descriptives(rows: list[dict]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    out = {}
    for class_name, items in sorted(by_class.items()):
        per_feature = {}
        for feature in ALL_FEATURES:
            values = np.asarray([float(r[feature]) for r in items], dtype=float)
            values = values[np.isfinite(values)]
            per_feature[feature] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        out[class_name] = per_feature
    return out


def _discover_development_images(dev_root: Path) -> list[tuple[str, str, Path]]:
    dev_root = Path(dev_root).expanduser().resolve()
    if (dev_root / "source/test").exists():
        raise RuntimeError("Audit development tidak boleh memiliki source/test.")
    contract = dev_root / "development_contract.json"
    if not contract.is_file():
        raise RuntimeError("development_contract.json tidak ditemukan.")
    payload = json.loads(contract.read_text(encoding="utf-8"))
    if payload.get("test_images_extracted") is not False:
        raise RuntimeError("Development contract menunjukkan test pernah diekstrak.")

    found = []
    for split in ("train", "val"):
        root = dev_root / "source" / split
        if not root.is_dir():
            raise FileNotFoundError(root)
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                found.append((split, class_dir.name, path))
    return found


def run_fold_audit(*, dev_root: Path, output_dir: Path, fold: int) -> dict:
    samples = _discover_development_images(dev_root)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for index, (split, class_name, path) in enumerate(samples, start=1):
        try:
            desc = extract_physical_descriptors(path)
        except Exception as exc:
            failures.append({
                "path": str(path),
                "class": class_name,
                "split": split,
                "error": repr(exc),
            })
            continue
        row = {
            "fold": fold,
            "split": split,
            "class": class_name,
            "filename": path.name,
            "identity": f"{class_name}/{path.name}",
            **desc,
        }
        rows.append(row)
        if index % 100 == 0:
            print(f"fold {fold}: {index}/{len(samples)}", flush=True)

    if failures:
        (output_dir / "extraction_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Descriptor extraction gagal untuk {len(failures)} image; lihat extraction_failures.json"
        )

    qc_fail = [r for r in rows if not bool(r["mask_qc_pass"])]
    fieldnames = list(rows[0].keys())
    with (output_dir / "descriptors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hypotheses = {
        "full_vs_partial_sour": _pairwise(
            rows,
            class_a="Full Sour",
            class_b="Partial Sour",
            features=SOUR_FEATURES,
        ),
        "full_vs_partial_black": _pairwise(
            rows,
            class_a="Full Black",
            class_b="Partial Black",
            features=BLACK_FEATURES,
        ),
        "severe_vs_slight_insect": _pairwise(
            rows,
            class_a="Severe Insect Damage",
            class_b="Slight Insect Damage",
            features=INSECT_FEATURES,
        ),
        "shape_family": _shape_group(rows),
    }

    payload = {
        "format": "bilinear_lmmd.coffee17.physical_descriptor_audit.fold.v1",
        "fold": int(fold),
        "scope": "Coffee17 fold-specific development train+val only; no test materialization; no neural training",
        "sample_count": len(rows),
        "class_count": len({r["class"] for r in rows}),
        "mask_qc_fail_count": len(qc_fail),
        "mask_qc_fail_identities": [r["identity"] for r in qc_fail],
        "descriptor_extraction_uses_labels": False,
        "labels_used_only_for_statistical_grouping": True,
        "direct_sour_or_black_segmentation_claimed": False,
        "dark_component_counts_are_exact_insect_holes": False,
        "hypotheses": hypotheses,
        "class_descriptives": _class_descriptives(rows),
        "outer_test_accessed": False,
        "training_executed": False,
    }
    (output_dir / "fold_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    args = parser.parse_args()
    run_fold_audit(
        dev_root=args.dev_root,
        output_dir=args.output_dir,
        fold=args.fold,
    )


if __name__ == "__main__":
    main()
