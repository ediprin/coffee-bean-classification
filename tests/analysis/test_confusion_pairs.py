from __future__ import annotations

import csv
from pathlib import Path

from bilinear_lmmd.analysis.confusion_pairs import (
    PredictionRun,
    build_confusion_audit,
    sample_identity,
    write_confusion_audit,
)


def _predictions(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "actual", "predicted", "correct"])
        for sample, actual, predicted in rows:
            writer.writerow([sample, actual, predicted, int(actual == predicted)])


def test_sample_identity_ignores_execution_root():
    assert sample_identity("/content/a/source/val/Cut/x.jpg") == "Cut/x.jpg"
    assert sample_identity("C:\\tmp\\source\\val\\Cut\\x.jpg") == "Cut/x.jpg"


def test_confusion_audit_finds_stable_pair_and_unanimous_wrong(tmp_path: Path):
    paths = []
    for model, seed in (("GAP", 42), ("HBP", 42), ("GAP", 123), ("HBP", 123)):
        path = tmp_path / f"{model}_{seed}.csv"
        _predictions(
            path,
            [
                ("/run/source/val/A/a.jpg", "A", "B"),
                ("/run/source/val/B/b.jpg", "B", "B"),
            ],
        )
        paths.append(PredictionRun(model, seed, path))

    audit = build_confusion_audit(paths)
    assert audit["unanimous_wrong_count"] == 1
    assert audit["stable_pairs"] == [
        {
            "class_a": "A",
            "class_b": "B",
            "errors": 4,
            "distinct_models": 2,
            "distinct_seeds": 2,
            "stable": True,
        }
    ]
    write_confusion_audit(audit, tmp_path / "out")
    assert (tmp_path / "out" / "confusion_audit.json").is_file()
    assert (tmp_path / "out" / "pairwise_confusion.csv").is_file()
    assert (tmp_path / "out" / "sample_consensus.csv").is_file()
