from __future__ import annotations

import csv
from pathlib import Path

from bilinear_lmmd.analysis.multistage_capacity import (
    MODEL_CODES,
    analyze_multistage_capacity,
)


CLASSES = (
    "Cut",
    "Full Sour",
    "Immature",
    "Partial Black",
    "Partial Sour",
    "Severe Insect Damage",
    "Slight Insect Damage",
    "Withered",
)


def _write_predictions(
    path: Path,
    predictions: dict[str, list[str]],
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "actual", "predicted", "correct"])
        for actual in CLASSES:
            for sample_index, predicted in enumerate(predictions[model][actual]):
                writer.writerow(
                    [
                        f"/run/source/val/{actual}/{sample_index}.jpg",
                        actual,
                        predicted,
                        int(actual == predicted),
                    ]
                )


def test_multistage_capacity_audit_aligns_and_bootstraps(tmp_path: Path) -> None:
    values = {
        model: {class_name: [class_name, class_name] for class_name in CLASSES}
        for model in MODEL_CODES
    }
    values["BE2G"]["Withered"][0] = "Immature"
    values["BE2H"]["Withered"][0] = "Immature"
    values["MSF0"]["Withered"][0] = "Immature"
    values["MSF1"]["Cut"][0] = "Withered"
    paths = {}
    for model in MODEL_CODES:
        path = tmp_path / model / "predictions.csv"
        _write_predictions(path, values, model)
        paths[model] = path

    output = tmp_path / "audit"
    report = analyze_multistage_capacity(
        paths,
        output,
        iterations=100,
        random_seed=7,
    )

    assert report["selection_split"] == "val"
    assert report["test_opened"] is False
    assert report["sample_count"] == 16
    comparison = report["comparisons"]["BE2G_vs_MSFC"]
    assert comparison["outcome"]["rescued_by_candidate"] == 1
    assert comparison["point_delta"]["macro_f1"] > 0
    assert set(comparison["bootstrap"]) == {
        "accuracy",
        "macro_f1",
        "hard_class_f1",
        "bottom3_class_f1",
        "worst_class_f1",
    }
    assert (output / "multistage_capacity_audit.json").is_file()
    assert (output / "per_class_deltas.csv").is_file()
    assert (output / "sample_outcomes.csv").is_file()
    assert (output / "bootstrap_summary.csv").is_file()
