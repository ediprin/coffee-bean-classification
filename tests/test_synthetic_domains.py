import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from bilinear_lmmd.config import load_config
from bilinear_lmmd.prepare_synthetic_domains import prepare_synthetic_domains
from bilinear_lmmd.run_synthetic_benchmark import (
    METRICS,
    _collect_summary,
    _parse_source_checkpoint_specs,
    _validate_source_checkpoint,
)


def _bean_image(path, bean_color):
    image = Image.new("RGB", (80, 64), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((18, 14, 62, 50), fill=bean_color, outline=(45, 65, 35), width=2)
    draw.line((40, 17, 39, 47), fill=(75, 85, 50), width=2)
    image.save(path)


def _source_tree(root):
    for split in ("train", "val", "test"):
        for class_name, color in (("A", (110, 145, 65)), ("B", (75, 110, 50))):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            _bean_image(class_dir / f"{split}_{class_name}.jpg", color)


def test_prepare_synthetic_domains_preserves_splits_and_writes_audit(tmp_path):
    source = tmp_path / "source"
    _source_tree(source)
    output = tmp_path / "synthetic"

    metadata = prepare_synthetic_domains(
        source,
        output,
        ["illumination", "sensor", "background", "combined"],
        seed=17,
    )

    assert metadata["status"] == "complete"
    assert metadata["source_samples"] == 6
    assert metadata["mask_failure_count"] == 0
    for domain in metadata["domains"]:
        assert metadata["counts"][domain] == {"train": 2, "val": 2, "test": 2}
        for split in ("train", "val", "test"):
            source_files = sorted((output / domain / "source" / split).glob("*/*"))
            target_files = sorted((output / domain / "target" / split).glob("*/*"))
            assert len(source_files) == len(target_files) == 2

    original = np.asarray(Image.open(source / "test" / "A" / "test_A.jpg"))
    shifted = np.asarray(
        Image.open(
            output
            / "illumination"
            / "target"
            / "test"
            / "A"
            / "test_A__illumination.jpg"
        )
    )
    assert np.mean(np.abs(original.astype(float) - shifted.astype(float))) > 2.0

    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 24
    assert all(row["source"].split("/", 1)[0] == row["split"] for row in rows)


def test_prepare_synthetic_domains_is_resumable_only_for_same_request(tmp_path):
    source = tmp_path / "source"
    _source_tree(source)
    output = tmp_path / "synthetic"

    first = prepare_synthetic_domains(source, output, ["sensor"], seed=42)
    second = prepare_synthetic_domains(source, output, ["sensor"], seed=42)
    assert first == second

    with pytest.raises(FileExistsError):
        prepare_synthetic_domains(source, output, ["sensor"], seed=123)


def test_synthetic_summary_reports_lmmd_vs_mmd_pairwise_delta(tmp_path):
    output = tmp_path / "results"
    scores = {"M0": 0.50, "M2": 0.60, "M3": 0.65}
    for model, score in scores.items():
        report_root = output / "reports" / "combined" / f"{model}_seed42"
        for domain, value in (("source", 0.80), ("target", score)):
            destination = report_root / domain
            destination.mkdir(parents=True)
            (destination / "metrics.json").write_text(
                json.dumps({metric: value for metric in METRICS}),
                encoding="utf-8",
            )

    summary = _collect_summary(
        output,
        domains=["combined"],
        models=["M0", "M2", "M3"],
        seeds=[42],
    )
    comparison = summary["pairwise"]["combined"]["M2_vs_M3"]
    assert comparison["delta"]["macro_f1"]["mean"] == pytest.approx(0.05)
    assert summary["aggregates"]["combined"]["M3"]["source"]["macro_f1"][
        "mean"
    ] == pytest.approx(0.80)


def test_source_checkpoint_spec_and_validation_are_seed_aware(tmp_path):
    checkpoint_path = tmp_path / "M0_seed123" / "best.pt"
    checkpoint_path.parent.mkdir()
    config = load_config("configs/M0_mobilenetv3_gap_source.yaml")
    config["seed"] = 123
    torch.save({"config": config}, checkpoint_path)

    parsed = _parse_source_checkpoint_specs(
        [f"M0:123={checkpoint_path}"]
    )
    assert parsed == {("M0", 123): checkpoint_path}
    _validate_source_checkpoint("M0", 123, checkpoint_path)

    with pytest.raises(ValueError, match="Seed checkpoint"):
        _validate_source_checkpoint("M0", 42, checkpoint_path)
    with pytest.raises(ValueError, match="source-only"):
        _parse_source_checkpoint_specs([f"M3:123={checkpoint_path}"])
    with pytest.raises(ValueError, match="MODEL:SEED=PATH"):
        _parse_source_checkpoint_specs(["M0=missing-seed"])
