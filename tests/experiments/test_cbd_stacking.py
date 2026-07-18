import csv
from pathlib import Path

import pytest

from bilinear_lmmd.experiments.run_cbd_stacking_confirmation import (
    aggregate_seed_results,
    load_prediction_table,
    stack_seed,
)


CLASSES = ["Black", "Good", "Sour"]


def _predictions(path: Path, rows: list[tuple[str, str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "actual", "predicted", "correct", *[f"prob::{c}" for c in CLASSES]]
        )
        for sample_path, actual, probabilities in rows:
            predicted = CLASSES[max(range(len(CLASSES)), key=probabilities.__getitem__)]
            writer.writerow(
                [sample_path, actual, predicted, int(actual == predicted), *probabilities]
            )


def test_load_prediction_table_sorts_paths(tmp_path):
    path = tmp_path / "predictions.csv"
    _predictions(
        path,
        [
            ("b.png", "Good", [0.1, 0.8, 0.1]),
            ("a.png", "Black", [0.8, 0.1, 0.1]),
        ],
    )
    table = load_prediction_table(path)
    assert table.paths == ["a.png", "b.png"]
    assert table.labels.tolist() == [0, 1]


def test_stack_seed_produces_all_controls(tmp_path):
    val_rows = []
    test_rows = []
    for index in range(18):
        actual = CLASSES[index % 3]
        label = index % 3
        gap = [0.1, 0.1, 0.1]
        hbp = [0.1, 0.1, 0.1]
        gap[label] = 0.8
        hbp[label] = 0.75
        val_rows.append((f"val-{index}.png", actual, gap, hbp))
        test_rows.append((f"test-{index}.png", actual, gap, hbp))

    paths = {}
    for split, rows in (("val", val_rows), ("test", test_rows)):
        for code, probability_index in (("gap", 2), ("hbp", 3)):
            path = tmp_path / split / f"{code}.csv"
            _predictions(
                path,
                [(sample, actual, row[probability_index]) for row in rows for sample, actual in [(row[0], row[1])]],
            )
            paths[(split, code)] = path

    result, prediction_rows = stack_seed(
        paths[("val", "gap")],
        paths[("val", "hbp")],
        paths[("test", "gap")],
        paths[("test", "hbp")],
    )
    assert set(result["models"]) == {
        "GAP_RAW", "HBP_RAW", "GAP_CAL", "HBP_CAL", "STACKING"
    }
    assert result["models"]["STACKING"]["accuracy"] == pytest.approx(1.0)
    assert len(prediction_rows) == 18


def test_aggregate_requires_consistent_fusion_gain():
    seed_results = {}
    for seed in (42, 123, 2026):
        seed_results[seed] = {
            "models": {
                name: {
                    "accuracy": value,
                    "macro_f1": value,
                    "defect_f1": value,
                    "worst_f1": value,
                    "worst_class": "Black",
                }
                for name, value in {
                    "GAP_RAW": 0.80,
                    "HBP_RAW": 0.81,
                    "GAP_CAL": 0.84,
                    "HBP_CAL": 0.83,
                    "STACKING": 0.85,
                }.items()
            }
        }
    aggregate = aggregate_seed_results(seed_results)
    assert aggregate["pre_registered_decision"]["status"] == "PASS"
    assert aggregate["stacking_deltas"]["GAP_CAL"]["macro_f1"]["improved_seeds"] == 3


def test_one_seed_stacking_is_screening_not_failure():
    result = {
        "models": {
            name: {
                "accuracy": value,
                "macro_f1": value,
                "defect_f1": value,
                "worst_f1": value,
                "worst_class": "Black",
            }
            for name, value in {
                "GAP_RAW": 0.80,
                "HBP_RAW": 0.81,
                "GAP_CAL": 0.84,
                "HBP_CAL": 0.83,
                "STACKING": 0.85,
            }.items()
        }
    }
    aggregate = aggregate_seed_results({42: result})
    assert aggregate["pre_registered_decision"]["status"] == "SCREEN_ONLY"
