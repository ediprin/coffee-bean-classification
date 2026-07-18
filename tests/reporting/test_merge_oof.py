import csv
import json

import bilinear_lmmd.reporting.merge_oof as merge_module


def _write_report(report_dir, rows):
    report_dir.mkdir()
    (report_dir / "metrics.json").write_text(
        json.dumps({"classes": ["A", "B"]}), encoding="utf-8"
    )
    with (report_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "actual", "predicted", "correct")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_merge_oof_combines_unique_predictions(tmp_path, monkeypatch):
    monkeypatch.setitem(
        merge_module.DEFAULTS["evaluation"], "hard_groups", {"hard": ["A", "B"]}
    )
    report_1 = tmp_path / "fold_1"
    report_2 = tmp_path / "fold_2"
    _write_report(
        report_1,
        [
            {"path": "/fold_1/test/A/a.jpg", "actual": "A", "predicted": "A", "correct": "1"},
            {"path": "/fold_1/test/B/b.jpg", "actual": "B", "predicted": "A", "correct": "0"},
        ],
    )
    _write_report(
        report_2,
        [
            {"path": "/fold_2/test/A/c.jpg", "actual": "A", "predicted": "A", "correct": "1"},
            {"path": "/fold_2/test/B/d.jpg", "actual": "B", "predicted": "B", "correct": "1"},
        ],
    )

    output = tmp_path / "oof"
    metrics = merge_module.merge_oof(
        [report_1, report_2], output, expected_count=4
    )
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.75
    assert len(list(csv.DictReader((output / "predictions.csv").open()))) == 4

