import json
import csv
from pathlib import Path

from PIL import Image

from bilinear_lmmd.config import load_config
from bilinear_lmmd.models import build_model
from bilinear_lmmd.prepare_coarse_coffee17 import (
    COARSE_GROUPS,
    EXPECTED_FINE_CLASSES,
    prepare_coarse_coffee17,
)
from bilinear_lmmd.run_granularity_experiment import _paired_granularity_effect
from bilinear_lmmd.run_granularity_bootstrap import run_granularity_bootstrap


def test_prepare_coarse_preserves_splits_and_all_images(tmp_path):
    fine_root = tmp_path / "fine"
    for split in ("train", "val", "test"):
        for index, class_name in enumerate(EXPECTED_FINE_CLASSES):
            destination = fine_root / "source" / split / class_name / f"{index}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(index, 10, 20)).save(destination)

    coarse_root = tmp_path / "coarse"
    audit = prepare_coarse_coffee17(fine_root, coarse_root)

    assert audit["status"] == "complete"
    assert audit["split_totals"] == {"train": 17, "val": 17, "test": 17}
    assert set(audit["coarse_classes"]) == set(COARSE_GROUPS)
    for split in ("train", "val", "test"):
        classes = {path.name for path in (coarse_root / "source" / split).iterdir()}
        assert classes == set(COARSE_GROUPS)
        assert len(list((coarse_root / "source" / split).glob("*/*"))) == 17


def test_coarse_configs_match_fine_head_capacity_except_class_count():
    pairs = (
        ("configs/M0_mobilenetv3_gap_source.yaml", "configs/GC0_mobilenetv3_gap_coarse9_source.yaml"),
        ("configs/M0b_mobilenetv3_bilinear_source.yaml", "configs/GC0b_mobilenetv3_bilinear_coarse9_source.yaml"),
        ("configs/M1_mobilenetv3_hbp_source.yaml", "configs/GC1_mobilenetv3_hbp_coarse9_source.yaml"),
    )
    for fine_path, coarse_path in pairs:
        fine = load_config(fine_path)
        coarse = load_config(coarse_path)
        assert fine["model"]["head"] == coarse["model"]["head"]
        assert fine["model"]["num_classes"] == 17
        assert coarse["model"]["num_classes"] == 9
        for key in ("backbone", "out_indices", "projection_dim"):
            assert fine["model"][key] == coarse["model"][key]
        assert fine["data"] == coarse["data"]
        assert fine["adaptation"] == coarse["adaptation"]
        fine_training = dict(fine["training"])
        coarse_training = dict(coarse["training"])
        fine_training.pop("output_dir")
        coarse_training.pop("output_dir")
        assert fine_training == coarse_training


def test_granularity_effect_is_difference_of_within_task_gains(tmp_path):
    values = {
        "GF0": 0.70,
        "GF0b": 0.76,
        "GF1": 0.80,
        "GC0": 0.85,
        "GC0b": 0.86,
        "GC1": 0.87,
    }
    for code, value in values.items():
        path = tmp_path / f"{code}_seed123" / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "accuracy": value,
                    "balanced_accuracy": value,
                    "macro_f1": value,
                    "worst_class_f1": value,
                }
            ),
            encoding="utf-8",
        )

    result = _paired_granularity_effect(tmp_path, [123])
    summary = result["metrics"]["macro_f1"]["summary"]
    assert abs(summary["fine_hbp_gain"]["mean"] - 0.10) < 1e-9
    assert abs(summary["coarse_hbp_gain"]["mean"] - 0.02) < 1e-9
    assert abs(summary["hbp_granularity_effect"]["mean"] - 0.08) < 1e-9


def test_granularity_models_build_with_expected_output_dimensions():
    expected = {
        "configs/GC0_mobilenetv3_gap_coarse9_source.yaml": 960,
        "configs/GC0b_mobilenetv3_bilinear_coarse9_source.yaml": 1536,
        "configs/GC1_mobilenetv3_hbp_coarse9_source.yaml": 1536,
    }
    for path, embedding_dim in expected.items():
        cfg = load_config(path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        assert model.classifier.out_features == 9
        assert model.pool.output_dim == embedding_dim


def _write_prediction_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "actual", "predicted", "correct"))
        for image_path, actual, predicted in rows:
            writer.writerow((image_path, actual, predicted, int(actual == predicted)))


def test_granularity_bootstrap_pairs_identical_samples(tmp_path):
    fine_classes = list(EXPECTED_FINE_CLASSES)
    fine_gap_rows = []
    fine_hbp_rows = []
    coarse_rows = []
    for index, fine_class in enumerate(fine_classes):
        wrong_class = fine_classes[(index + 1) % len(fine_classes)]
        coarse_class = next(
            coarse for coarse, members in COARSE_GROUPS.items() if fine_class in members
        )
        for sample in range(2):
            filename = f"sample_{index}_{sample}.png"
            fine_gap_prediction = fine_class if sample == 0 else wrong_class
            fine_gap_rows.append((f"/fine/{fine_class}/{filename}", fine_class, fine_gap_prediction))
            fine_hbp_rows.append((f"/fine/{fine_class}/{filename}", fine_class, fine_class))
            coarse_rows.append(
                (
                    f"/coarse/{coarse_class}/{fine_class}__{filename}",
                    coarse_class,
                    coarse_class,
                )
            )

    report_root = tmp_path / "reports"
    for seed in (42, 123):
        _write_prediction_csv(report_root / f"GF0_seed{seed}/predictions.csv", fine_gap_rows)
        _write_prediction_csv(report_root / f"GF1_seed{seed}/predictions.csv", fine_hbp_rows)
        _write_prediction_csv(report_root / f"GC0_seed{seed}/predictions.csv", coarse_rows)
        _write_prediction_csv(report_root / f"GC1_seed{seed}/predictions.csv", coarse_rows)

    output = tmp_path / "bootstrap.json"
    result = run_granularity_bootstrap(
        report_root=report_root,
        seeds=[42, 123],
        output=output,
        iterations=100,
        random_seed=7,
    )

    assert output.is_file()
    assert result["samples"] == 34
    assert result["point_estimate"]["fine_hbp_gain"] > 0
    assert result["point_estimate"]["coarse_hbp_gain"] == 0
    assert result["point_estimate"]["granularity_effect"] > 0
