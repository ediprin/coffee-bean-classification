from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class PredictionRun:
    model: str
    seed: int
    path: Path


def sample_identity(path: str) -> str:
    """Return an execution-root-independent identity for an ImageFolder sample."""

    normalized = path.replace("\\", "/")
    for marker in ("/source/train/", "/source/val/", "/source/test/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    parts = PurePosixPath(normalized).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else normalized


def _read_run(run: PredictionRun) -> list[dict[str, str]]:
    if not run.path.is_file():
        raise FileNotFoundError(f"Predictions tidak ditemukan: {run.path}")
    with run.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "actual", "predicted", "correct"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Kolom predictions kurang pada {run.path}: {sorted(missing)}")
    if not rows:
        raise ValueError(f"Predictions kosong: {run.path}")
    return rows


def build_confusion_audit(runs: list[PredictionRun]) -> dict:
    """Aggregate persistent class-pair confusion and elusive samples.

    This is diagnostic only. It deliberately consumes validation predictions
    and must never be used directly as a training sampler.
    """

    if len(runs) < 2:
        raise ValueError("Audit membutuhkan minimal dua prediction run.")
    run_keys = [(run.model, int(run.seed)) for run in runs]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("Kombinasi model/seed prediction harus unik.")

    sample_rows: dict[str, dict] = {}
    directed_errors: Counter[tuple[str, str]] = Counter()
    directed_occurrences: defaultdict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    actual_opportunities: Counter[str] = Counter()
    classes: set[str] = set()

    for run in runs:
        seen_in_run: set[str] = set()
        for row in _read_run(run):
            identity = sample_identity(row["path"])
            if identity in seen_in_run:
                raise ValueError(f"Sampel duplikat dalam run {run.model}/{run.seed}: {identity}")
            seen_in_run.add(identity)
            actual = row["actual"]
            predicted = row["predicted"]
            classes.update((actual, predicted))
            actual_opportunities[actual] += 1
            record = sample_rows.setdefault(
                identity,
                {"identity": identity, "actual": actual, "runs": []},
            )
            if record["actual"] != actual:
                raise ValueError(
                    f"Label aktual tidak konsisten untuk {identity}: "
                    f"{record['actual']} vs {actual}"
                )
            correct = int(actual == predicted)
            record["runs"].append(
                {
                    "model": run.model,
                    "seed": int(run.seed),
                    "predicted": predicted,
                    "correct": correct,
                }
            )
            if not correct:
                edge = (actual, predicted)
                directed_errors[edge] += 1
                directed_occurrences[edge].add((run.model, int(run.seed)))

    pair_rows = []
    undirected: dict[tuple[str, str], dict] = {}
    for (actual, predicted), errors in directed_errors.items():
        occurrences = directed_occurrences[(actual, predicted)]
        pair_rows.append(
            {
                "actual": actual,
                "predicted": predicted,
                "errors": errors,
                "opportunities": actual_opportunities[actual],
                "error_rate": errors / actual_opportunities[actual],
                "distinct_models": len({model for model, _ in occurrences}),
                "distinct_seeds": len({seed for _, seed in occurrences}),
            }
        )
        pair = tuple(sorted((actual, predicted)))
        aggregate = undirected.setdefault(
            pair,
            {
                "class_a": pair[0],
                "class_b": pair[1],
                "errors": 0,
                "runs": set(),
            },
        )
        aggregate["errors"] += errors
        aggregate["runs"].update(occurrences)

    pair_rows.sort(key=lambda row: (-row["errors"], row["actual"], row["predicted"]))
    symmetric_pairs = []
    for aggregate in undirected.values():
        occurrences = aggregate.pop("runs")
        aggregate["distinct_models"] = len({model for model, _ in occurrences})
        aggregate["distinct_seeds"] = len({seed for _, seed in occurrences})
        aggregate["stable"] = (
            aggregate["errors"] >= 2
            and aggregate["distinct_models"] >= 2
            and aggregate["distinct_seeds"] >= 2
        )
        symmetric_pairs.append(aggregate)
    symmetric_pairs.sort(
        key=lambda row: (-row["errors"], row["class_a"], row["class_b"])
    )

    samples = []
    expected_runs = len(runs)
    for record in sample_rows.values():
        correct_count = sum(item["correct"] for item in record["runs"])
        prediction_counts = Counter(item["predicted"] for item in record["runs"])
        samples.append(
            {
                "identity": record["identity"],
                "actual": record["actual"],
                "observed_runs": len(record["runs"]),
                "correct_count": correct_count,
                "correct_fraction": correct_count / len(record["runs"]),
                "unanimous_wrong": correct_count == 0 and len(record["runs"]) == expected_runs,
                "predictions": dict(sorted(prediction_counts.items())),
            }
        )
    samples.sort(key=lambda row: (row["correct_fraction"], row["identity"]))

    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "training_sampler_allowed": False,
        "runs": [
            {"model": run.model, "seed": int(run.seed), "path": str(run.path)}
            for run in runs
        ],
        "classes": sorted(classes),
        "sample_count": len(samples),
        "complete_sample_count": sum(
            row["observed_runs"] == expected_runs for row in samples
        ),
        "unanimous_wrong_count": sum(row["unanimous_wrong"] for row in samples),
        "directed_pairs": pair_rows,
        "symmetric_pairs": symmetric_pairs,
        "stable_pairs": [row for row in symmetric_pairs if row["stable"]],
        "samples": samples,
    }


def write_confusion_audit(audit: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confusion_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    with (output_dir / "pairwise_confusion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "actual",
            "predicted",
            "errors",
            "opportunities",
            "error_rate",
            "distinct_models",
            "distinct_seeds",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit["directed_pairs"])
    with (output_dir / "sample_consensus.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "identity",
            "actual",
            "observed_runs",
            "correct_count",
            "correct_fraction",
            "unanimous_wrong",
            "predictions",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audit["samples"]:
            writer.writerow({**row, "predictions": json.dumps(row["predictions"])})
