from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bilinear_lmmd.analysis.confusion_pairs import sample_identity
from bilinear_lmmd.core.config import DEFAULTS


MODEL_CODES = ("BE2G", "BE2H", "MSF0", "MSF1", "MSFC")
COMPARISONS = (
    ("BE2G", "MSFC"),
    ("BE2H", "MSFC"),
    ("MSF0", "MSFC"),
    ("MSFC", "MSF1"),
)
METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "hard_class_f1",
    "bottom3_class_f1",
    "worst_class_f1",
)


def _read_predictions(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions tidak ditemukan: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Predictions kosong: {path}")
    required = {"path", "actual", "predicted"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Kolom predictions kurang: {path}: {sorted(missing)}")

    table = {}
    for row in rows:
        identity = sample_identity(row["path"])
        if identity in table:
            raise ValueError(f"Identity duplikat pada {path}: {identity}")
        table[identity] = row
    return table


def _load_aligned_predictions(
    prediction_paths: dict[str, Path],
) -> tuple[list[str], tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    missing_models = set(MODEL_CODES).difference(prediction_paths)
    if missing_models:
        raise ValueError(f"Path model belum lengkap: {sorted(missing_models)}")
    tables = {
        code: _read_predictions(prediction_paths[code]) for code in MODEL_CODES
    }
    reference = set(tables[MODEL_CODES[0]])
    for code, table in tables.items():
        if set(table) != reference:
            raise ValueError(
                f"Identity {code} tidak sejajar dengan {MODEL_CODES[0]}: "
                f"missing={len(reference.difference(table))}, "
                f"extra={len(set(table).difference(reference))}"
            )

    identities = sorted(reference)
    actual_names = [tables[MODEL_CODES[0]][identity]["actual"] for identity in identities]
    for code, table in tables.items():
        mismatched = [
            identity
            for identity, actual in zip(identities, actual_names)
            if table[identity]["actual"] != actual
        ]
        if mismatched:
            raise ValueError(
                f"Ground truth {code} tidak sejajar: {mismatched[:3]}"
            )

    classes = tuple(sorted(set(actual_names)))
    class_index = {name: index for index, name in enumerate(classes)}
    unknown_predictions = {
        code: sorted(
            {
                table[identity]["predicted"]
                for identity in identities
                if table[identity]["predicted"] not in class_index
            }
        )
        for code, table in tables.items()
    }
    unknown_predictions = {
        code: names for code, names in unknown_predictions.items() if names
    }
    if unknown_predictions:
        raise ValueError(f"Prediksi kelas tidak dikenal: {unknown_predictions}")

    actual = np.asarray([class_index[name] for name in actual_names], dtype=np.int64)
    predictions = {
        code: np.asarray(
            [class_index[tables[code][identity]["predicted"]] for identity in identities],
            dtype=np.int64,
        )
        for code in MODEL_CODES
    }
    return identities, classes, actual, predictions


def _hard_indices(classes: tuple[str, ...]) -> np.ndarray:
    hard_groups = DEFAULTS["evaluation"]["hard_groups"]
    hard_names = list(
        dict.fromkeys(name for members in hard_groups.values() for name in members)
    )
    unknown = sorted(set(hard_names).difference(classes))
    if unknown:
        raise ValueError(f"Hard class tidak ditemukan pada predictions: {unknown}")
    index = {name: position for position, name in enumerate(classes)}
    return np.asarray([index[name] for name in hard_names], dtype=np.int64)


def _metric_bundle(
    actual: np.ndarray,
    predicted: np.ndarray,
    indices: np.ndarray,
    class_count: int,
    hard_indices: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    selected_actual = actual[indices]
    selected_predicted = predicted[indices]
    matrix = np.bincount(
        selected_actual * class_count + selected_predicted,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    sorted_f1 = np.sort(per_class_f1)
    metrics = {
        "accuracy": float((selected_actual == selected_predicted).mean()),
        "macro_f1": float(per_class_f1.mean()),
        "hard_class_f1": float(per_class_f1[hard_indices].mean()),
        "bottom3_class_f1": float(sorted_f1[:3].mean()),
        "worst_class_f1": float(sorted_f1[0]),
    }
    return metrics, per_class_f1


def _interval(values: np.ndarray, confidence: float) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "probability_positive": float((values > 0.0).mean()),
    }


def _outcome(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
) -> dict[str, int]:
    baseline_correct = baseline[indices] == actual[indices]
    candidate_correct = candidate[indices] == actual[indices]
    return {
        "both_correct": int((baseline_correct & candidate_correct).sum()),
        "rescued_by_candidate": int((~baseline_correct & candidate_correct).sum()),
        "harmed_by_candidate": int((baseline_correct & ~candidate_correct).sum()),
        "both_wrong": int((~baseline_correct & ~candidate_correct).sum()),
    }


def analyze_multistage_capacity(
    prediction_paths: dict[str, Path],
    output_dir: Path,
    *,
    iterations: int = 10_000,
    random_seed: int = 20260723,
    confidence: float = 0.95,
) -> dict:
    if iterations < 100:
        raise ValueError("Bootstrap memerlukan minimal 100 iterasi.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence harus berada di antara 0 dan 1.")

    identities, classes, actual, predictions = _load_aligned_predictions(
        prediction_paths
    )
    hard_indices = _hard_indices(classes)
    all_indices = np.arange(len(actual), dtype=np.int64)
    class_groups = [
        all_indices[actual == class_index] for class_index in range(len(classes))
    ]
    if any(len(group) == 0 for group in class_groups):
        raise ValueError("Setiap kelas harus hadir pada validation.")

    point_metrics = {}
    point_per_class = {}
    for code in MODEL_CODES:
        metrics, per_class_f1 = _metric_bundle(
            actual,
            predictions[code],
            all_indices,
            len(classes),
            hard_indices,
        )
        point_metrics[code] = metrics
        point_per_class[code] = per_class_f1

    comparisons = {}
    per_class_rows = []
    sample_rows = []
    for baseline, candidate in COMPARISONS:
        key = f"{baseline}_vs_{candidate}"
        outcome = _outcome(
            actual,
            predictions[baseline],
            predictions[candidate],
            all_indices,
        )
        comparisons[key] = {
            "baseline": baseline,
            "candidate": candidate,
            "point_delta": {
                name: point_metrics[candidate][name] - point_metrics[baseline][name]
                for name in METRIC_NAMES
            },
            "outcome": outcome,
        }
        for class_index, class_name in enumerate(classes):
            indices = class_groups[class_index]
            class_outcome = _outcome(
                actual,
                predictions[baseline],
                predictions[candidate],
                indices,
            )
            per_class_rows.append(
                {
                    "comparison": key,
                    "class": class_name,
                    "support": int(len(indices)),
                    "baseline_f1": float(point_per_class[baseline][class_index]),
                    "candidate_f1": float(point_per_class[candidate][class_index]),
                    "delta_f1": float(
                        point_per_class[candidate][class_index]
                        - point_per_class[baseline][class_index]
                    ),
                    **class_outcome,
                }
            )
        for position, identity in enumerate(identities):
            baseline_correct = predictions[baseline][position] == actual[position]
            candidate_correct = predictions[candidate][position] == actual[position]
            if baseline_correct and candidate_correct:
                outcome_name = "both_correct"
            elif not baseline_correct and candidate_correct:
                outcome_name = "rescued_by_candidate"
            elif baseline_correct and not candidate_correct:
                outcome_name = "harmed_by_candidate"
            else:
                outcome_name = "both_wrong"
            sample_rows.append(
                {
                    "comparison": key,
                    "identity": identity,
                    "actual": classes[actual[position]],
                    "baseline_prediction": classes[predictions[baseline][position]],
                    "candidate_prediction": classes[predictions[candidate][position]],
                    "outcome": outcome_name,
                }
            )

    bootstrap = {
        f"{baseline}_vs_{candidate}": {
            metric: np.empty(iterations, dtype=np.float64)
            for metric in METRIC_NAMES
        }
        for baseline, candidate in COMPARISONS
    }
    rng = np.random.default_rng(random_seed)
    progress_every = max(1, iterations // 20)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_groups]
        )
        sampled_metrics = {
            code: _metric_bundle(
                actual,
                predictions[code],
                sampled,
                len(classes),
                hard_indices,
            )[0]
            for code in MODEL_CODES
        }
        for baseline, candidate in COMPARISONS:
            key = f"{baseline}_vs_{candidate}"
            for metric in METRIC_NAMES:
                bootstrap[key][metric][iteration] = (
                    sampled_metrics[candidate][metric]
                    - sampled_metrics[baseline][metric]
                )
        completed = iteration + 1
        if completed % progress_every == 0 or completed == iterations:
            print(
                f"[BOOTSTRAP {completed}/{iterations}] "
                f"{completed / iterations:.0%}",
                flush=True,
            )

    bootstrap_rows = []
    for key, metric_values in bootstrap.items():
        comparisons[key]["bootstrap"] = {}
        for metric, values in metric_values.items():
            interval = _interval(values, confidence)
            comparisons[key]["bootstrap"][metric] = interval
            bootstrap_rows.append(
                {"comparison": key, "metric": metric, **interval}
            )

    class_support = Counter(classes[index] for index in actual.tolist())
    report = {
        "schema_version": 1,
        "analysis": "Coffee17 multistage capacity audit",
        "selection_split": "val",
        "test_opened": False,
        "diagnostic_only": True,
        "sample_count": len(actual),
        "classes": list(classes),
        "class_support": dict(sorted(class_support.items())),
        "minimum_class_support": min(class_support.values()),
        "models": {
            code: {
                "predictions": str(prediction_paths[code]),
                "metrics": point_metrics[code],
            }
            for code in MODEL_CODES
        },
        "comparisons": comparisons,
        "bootstrap": {
            "iterations": iterations,
            "random_seed": random_seed,
            "confidence": confidence,
            "sampling": "paired stratified by actual class",
        },
        "limitations": [
            "Hanya validation seed 42; bukan bukti konfirmasi multi-seed.",
            "Dukungan per kelas kecil membuat Worst-F1 sangat diskrit dan tidak stabil.",
            "Bootstrap mengukur ketidakpastian sampel pada split ini, bukan variasi training seed.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multistage_capacity_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_class_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    with (output_dir / "sample_outcomes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    with (output_dir / "bootstrap_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    print("\n=== AUDIT MULTISTAGE CAPACITY (VALIDATION SEED 42) ===")
    for baseline, candidate in COMPARISONS:
        key = f"{baseline}_vs_{candidate}"
        row = comparisons[key]
        print(f"\n{key}")
        for metric in (
            "macro_f1",
            "hard_class_f1",
            "bottom3_class_f1",
            "worst_class_f1",
        ):
            interval = row["bootstrap"][metric]
            print(
                f"  {metric:22s} delta={row['point_delta'][metric]:+.2%} "
                f"CI={interval['lower']:+.2%}..{interval['upper']:+.2%} "
                f"P(>0)={interval['probability_positive']:.3f}"
            )
        print(
            "  outcome: "
            f"rescue={row['outcome']['rescued_by_candidate']} "
            f"harm={row['outcome']['harmed_by_candidate']} "
            f"both_wrong={row['outcome']['both_wrong']}"
        )
    print("\nTest dibuka: False")
    print("SAVED:", output_dir / "multistage_capacity_audit.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit tanpa training untuk Coffee17 MSF capacity control."
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260723)
    args = parser.parse_args()

    prediction_paths = {
        "BE2G": (
            args.baseline_root
            / "val_reports"
            / f"BE2G_seed{args.seed}"
            / "predictions.csv"
        ),
        "BE2H": (
            args.baseline_root
            / "val_reports"
            / f"BE2H_seed{args.seed}"
            / "predictions.csv"
        ),
        **{
            code: (
                args.output_root
                / "val_reports"
                / f"{code}_seed{args.seed}"
                / "predictions.csv"
            )
            for code in ("MSF0", "MSF1", "MSFC")
        },
    }
    analyze_multistage_capacity(
        prediction_paths,
        args.output_root / "val_reports" / f"capacity_audit_seed{args.seed}",
        iterations=args.iterations,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
