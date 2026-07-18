from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bilinear_lmmd.data.preparation.prepare_coarse_coffee17 import (
    COARSE_GROUPS,
    EXPECTED_FINE_CLASSES,
    FINE_TO_COARSE,
)


FINE_CLASSES = tuple(EXPECTED_FINE_CLASSES)
COARSE_CLASSES = tuple(sorted(COARSE_GROUPS))


@dataclass(frozen=True)
class PredictionRow:
    identity: str
    fine_class: str
    actual: str
    predicted: str


@dataclass(frozen=True)
class SeedArrays:
    fine_actual: np.ndarray
    fine_gap: np.ndarray
    fine_hbp: np.ndarray
    coarse_actual: np.ndarray
    coarse_gap: np.ndarray
    coarse_hbp: np.ndarray


def _read_predictions(path: Path, task: str) -> dict[str, PredictionRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions belum ditemukan: {path}")
    if task not in {"fine", "coarse"}:
        raise ValueError(f"Task tidak dikenal: {task}")

    rows: dict[str, PredictionRow] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = Path(row["path"]).name
            if task == "fine":
                fine_class = row["actual"]
                original_name = filename
                if fine_class not in FINE_TO_COARSE:
                    raise ValueError(f"Kelas fine tidak dikenal pada {path}: {fine_class}")
            else:
                if "__" not in filename:
                    raise ValueError(
                        f"Nama coarse tidak menyimpan identitas fine: {filename}"
                    )
                fine_class, original_name = filename.split("__", 1)
                if fine_class not in FINE_TO_COARSE:
                    raise ValueError(f"Prefix fine tidak dikenal: {filename}")
                expected_actual = FINE_TO_COARSE[fine_class]
                if row["actual"] != expected_actual:
                    raise ValueError(
                        f"Label coarse {row['actual']} tidak cocok dengan "
                        f"{fine_class}->{expected_actual}"
                    )

            identity = f"{fine_class}/{original_name}"
            if identity in rows:
                raise ValueError(f"Identity duplikat pada {path}: {identity}")
            rows[identity] = PredictionRow(
                identity=identity,
                fine_class=fine_class,
                actual=row["actual"],
                predicted=row["predicted"],
            )
    if not rows:
        raise ValueError(f"Predictions kosong: {path}")
    return rows


def _encoded(values: list[str], classes: tuple[str, ...], source: Path) -> np.ndarray:
    index = {name: position for position, name in enumerate(classes)}
    unknown = sorted(set(values).difference(index))
    if unknown:
        raise ValueError(f"Kelas tidak dikenal pada {source}: {unknown}")
    return np.asarray([index[value] for value in values], dtype=np.int64)


def _load_seed(report_root: Path, seed: int) -> tuple[list[str], SeedArrays]:
    paths = {
        "fine_gap": report_root / f"GF0_seed{seed}" / "predictions.csv",
        "fine_hbp": report_root / f"GF1_seed{seed}" / "predictions.csv",
        "coarse_gap": report_root / f"GC0_seed{seed}" / "predictions.csv",
        "coarse_hbp": report_root / f"GC1_seed{seed}" / "predictions.csv",
    }
    tables = {
        name: _read_predictions(path, "fine" if name.startswith("fine") else "coarse")
        for name, path in paths.items()
    }
    identity_sets = {name: set(table) for name, table in tables.items()}
    reference = identity_sets["fine_gap"]
    mismatches = {
        name: {
            "missing": sorted(reference.difference(identities))[:5],
            "extra": sorted(identities.difference(reference))[:5],
        }
        for name, identities in identity_sets.items()
        if identities != reference
    }
    if mismatches:
        raise ValueError(f"Identity prediction tidak sejajar untuk seed {seed}: {mismatches}")

    identities = sorted(reference)
    fine_actual_names = [tables["fine_gap"][identity].actual for identity in identities]
    for identity, actual in zip(identities, fine_actual_names):
        if tables["fine_hbp"][identity].actual != actual:
            raise ValueError(f"Ground truth fine berbeda untuk {identity}")
        expected_coarse = FINE_TO_COARSE[actual]
        if any(
            tables[name][identity].actual != expected_coarse
            for name in ("coarse_gap", "coarse_hbp")
        ):
            raise ValueError(f"Ground truth coarse berbeda untuk {identity}")

    return identities, SeedArrays(
        fine_actual=_encoded(fine_actual_names, FINE_CLASSES, paths["fine_gap"]),
        fine_gap=_encoded(
            [tables["fine_gap"][identity].predicted for identity in identities],
            FINE_CLASSES,
            paths["fine_gap"],
        ),
        fine_hbp=_encoded(
            [tables["fine_hbp"][identity].predicted for identity in identities],
            FINE_CLASSES,
            paths["fine_hbp"],
        ),
        coarse_actual=_encoded(
            [tables["coarse_gap"][identity].actual for identity in identities],
            COARSE_CLASSES,
            paths["coarse_gap"],
        ),
        coarse_gap=_encoded(
            [tables["coarse_gap"][identity].predicted for identity in identities],
            COARSE_CLASSES,
            paths["coarse_gap"],
        ),
        coarse_hbp=_encoded(
            [tables["coarse_hbp"][identity].predicted for identity in identities],
            COARSE_CLASSES,
            paths["coarse_hbp"],
        ),
    )


def _macro_f1(
    actual: np.ndarray,
    predicted: np.ndarray,
    indices: np.ndarray,
    classes: int,
) -> float:
    matrix = np.bincount(
        actual[indices] * classes + predicted[indices],
        minlength=classes * classes,
    ).reshape(classes, classes)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(scores.mean())


def _effect(arrays: SeedArrays, indices: np.ndarray) -> tuple[float, float, float]:
    fine_gain = _macro_f1(
        arrays.fine_actual, arrays.fine_hbp, indices, len(FINE_CLASSES)
    ) - _macro_f1(arrays.fine_actual, arrays.fine_gap, indices, len(FINE_CLASSES))
    coarse_gain = _macro_f1(
        arrays.coarse_actual, arrays.coarse_hbp, indices, len(COARSE_CLASSES)
    ) - _macro_f1(
        arrays.coarse_actual, arrays.coarse_gap, indices, len(COARSE_CLASSES)
    )
    return fine_gain, coarse_gain, fine_gain - coarse_gain


def _interval(values: np.ndarray, confidence: float) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "probability_positive": float((values > 0).mean()),
    }


def run_granularity_bootstrap(
    report_root: Path,
    seeds: list[int],
    output: Path,
    iterations: int = 10_000,
    random_seed: int = 20260717,
    confidence: float = 0.95,
) -> dict:
    if not seeds:
        raise ValueError("Minimal satu seed diperlukan.")
    if iterations < 100:
        raise ValueError("Bootstrap memerlukan minimal 100 iterasi.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence harus berada di antara 0 dan 1.")

    loaded = [_load_seed(report_root, seed) for seed in seeds]
    identities = loaded[0][0]
    if any(current_identities != identities for current_identities, _ in loaded[1:]):
        raise ValueError("Identity test berbeda lintas seed.")
    arrays = [item[1] for item in loaded]
    reference_actual = arrays[0].fine_actual
    if any(not np.array_equal(item.fine_actual, reference_actual) for item in arrays[1:]):
        raise ValueError("Ground truth fine berubah lintas seed.")

    all_indices = np.arange(len(identities), dtype=np.int64)
    groups = [
        all_indices[reference_actual == class_index]
        for class_index in range(len(FINE_CLASSES))
    ]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Setiap fine class harus hadir agar stratified bootstrap valid.")

    per_seed = []
    for seed, item in zip(seeds, arrays):
        fine_gain, coarse_gain, effect = _effect(item, all_indices)
        per_seed.append(
            {
                "seed": seed,
                "fine_hbp_gain": fine_gain,
                "coarse_hbp_gain": coarse_gain,
                "granularity_effect": effect,
            }
        )

    fixed = np.empty((iterations, 3), dtype=np.float64)
    hierarchical = np.empty((iterations, 3), dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    for iteration in tqdm(range(iterations), desc="paired stratified bootstrap"):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        seed_effects = np.asarray([_effect(item, sampled) for item in arrays])
        fixed[iteration] = seed_effects.mean(axis=0)
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        hierarchical[iteration] = seed_effects[sampled_seeds].mean(axis=0)

    metric_names = ("fine_hbp_gain", "coarse_hbp_gain", "granularity_effect")
    point = {
        name: float(np.mean([row[name] for row in per_seed]))
        for name in metric_names
    }
    result = {
        "report_root": str(report_root),
        "seeds": seeds,
        "samples": len(identities),
        "iterations": iterations,
        "random_seed": random_seed,
        "confidence": confidence,
        "estimand": "(HBP_fine-GAP_fine)-(HBP_coarse-GAP_coarse)",
        "stratification": "resample_with_replacement_within_each_fine_class",
        "per_seed": per_seed,
        "point_estimate": point,
        "bootstrap_fixed_trained_seeds": {
            name: _interval(fixed[:, index], confidence)
            for index, name in enumerate(metric_names)
        },
        "bootstrap_hierarchical_seed_and_sample": {
            name: _interval(hierarchical[:, index], confidence)
            for index, name in enumerate(metric_names)
        },
        "interpretation": {
            "fixed_seed_ci_above_zero": bool(
                np.quantile(fixed[:, 2], (1.0 - confidence) / 2.0) > 0
            ),
            "hierarchical_ci_above_zero": bool(
                np.quantile(hierarchical[:, 2], (1.0 - confidence) / 2.0) > 0
            ),
            "scope": (
                "Fixed-seed bootstrap quantifies test-sample uncertainty conditional "
                "on these trained seeds; hierarchical bootstrap additionally resamples "
                "only three seeds and must remain exploratory."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== PAIRED STRATIFIED BOOTSTRAP ===")
    print(f"Samples/seed : {len(identities)}")
    print(f"Seeds        : {seeds}")
    print(f"Iterations   : {iterations:,}")
    for label, key in (
        ("Fine HBP gain", "fine_hbp_gain"),
        ("Coarse HBP gain", "coarse_hbp_gain"),
        ("Granularity effect", "granularity_effect"),
    ):
        row = result["bootstrap_fixed_trained_seeds"][key]
        print(
            f"{label:19s}: point={point[key]:+.2%} "
            f"CI={row['lower']:+.2%}..{row['upper']:+.2%} "
            f"P(>0)={row['probability_positive']:.3f}"
        )
    hierarchical_row = result["bootstrap_hierarchical_seed_and_sample"][
        "granularity_effect"
    ]
    print(
        "Hierarchical effect: "
        f"CI={hierarchical_row['lower']:+.2%}..{hierarchical_row['upper']:+.2%} "
        f"P(>0)={hierarchical_row['probability_positive']:.3f}"
    )
    print(f"SAVED: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired stratified bootstrap untuk efek granularity GAP-HBP"
    )
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260717)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    run_granularity_bootstrap(
        report_root=args.report_root,
        seeds=args.seeds,
        output=args.output,
        iterations=args.iterations,
        random_seed=args.random_seed,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
