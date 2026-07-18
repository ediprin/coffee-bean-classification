from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.loaders import _transforms
from bilinear_lmmd.data.preparation.prepare_paper_protocol import prepare_paper_protocol
from bilinear_lmmd.experiments.run_paper_reproduction import _model_counts


def _raw_dataset(root: Path) -> dict[str, int]:
    counts = {"Class A": 10, "Class B": 10}
    for class_index, (class_name, count) in enumerate(counts.items()):
        destination = root / class_name
        destination.mkdir(parents=True)
        for index in range(count):
            Image.new(
                "RGB",
                (24, 24),
                (30 + class_index * 80, 40 + index, 50),
            ).save(destination / f"sample_{index:02d}.jpg")
    return counts


def test_prepare_paper_protocol_matches_70_20_10_after_rotation(tmp_path: Path):
    raw = tmp_path / "raw"
    expected = _raw_dataset(raw)
    output = tmp_path / "paper"

    audit = prepare_paper_protocol(
        raw,
        output,
        seed=42,
        expected_counts=expected,
    )

    assert audit["original_count"] == 20
    assert audit["augmented_count"] == 140
    assert audit["split_counts"] == {"train": 98, "val": 28, "test": 14}
    assert audit["identity_overlap"]["originals_crossing_splits"] > 0
    assert len(list((output / "source").glob("*/*/*"))) == 140


def test_paper_transform_has_no_random_online_augmentation():
    image = Image.new("RGB", (31, 25), (80, 100, 120))
    transform = _transforms(
        224,
        train=True,
        rotation_angles=[0],
        augmentation_mode="paper",
    )

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)


def test_paper_configs_freeze_backbone_and_differ_only_in_pooling_capacity():
    gap = load_config("configs/paper/P0_paper_mobilenetv3_gap_ce.yaml")
    hbp = load_config("configs/paper/P1_paper_mobilenetv3_hbp_ce.yaml")
    gap_total, gap_trainable = _model_counts(
        Path("configs/paper/P0_paper_mobilenetv3_gap_ce.yaml")
    )
    hbp_total, hbp_trainable = _model_counts(
        Path("configs/paper/P1_paper_mobilenetv3_hbp_ce.yaml")
    )

    assert gap["training"]["freeze_backbone"] is True
    assert hbp["training"]["freeze_backbone"] is True
    assert gap["training"]["epochs"] == hbp["training"]["epochs"] == 3
    assert gap["training"]["lr"] == hbp["training"]["lr"] == 0.01
    assert gap_trainable < gap_total
    assert hbp_trainable < hbp_total
    assert hbp_trainable > gap_trainable
