from pathlib import Path

from PIL import Image

from bilinear_lmmd.data.preprocessing.study_data import (
    EpochDeterministicSampler,
    PreprocessingStudyImageFolder,
    deterministic_rotation_angle,
)


def _dataset(root: Path):
    for class_index, name in enumerate(("A", "B")):
        for index in range(4):
            destination = root / name
            destination.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (20, 18), (20 + class_index * 100, index * 10, 120)
            ).save(destination / f"{index}.png")


def test_rotation_is_stateless_by_seed_epoch_identity():
    angles = [0, 45, 90, 135, 180, 225, 270]
    a = deterministic_rotation_angle(
        seed=42, epoch=7, identity="A/x.png", angles=angles
    )
    b = deterministic_rotation_angle(
        seed=42, epoch=7, identity="A/x.png", angles=angles
    )
    assert a == b
    assert a in angles


def test_dataset_and_sampler_are_reproducible(tmp_path):
    root = tmp_path / "train"
    _dataset(root)
    first = PreprocessingStudyImageFolder(
        root, image_size=32, train=True, seed=42,
        rotation_angles=[0,45,90,135,180,225,270],
    )
    second = PreprocessingStudyImageFolder(
        root, image_size=32, train=True, seed=42,
        rotation_angles=[0,45,90,135,180,225,270],
    )
    first.set_epoch(3)
    second.set_epoch(3)
    assert [first.rotation_for_index(i) for i in range(len(first))] == [
        second.rotation_for_index(i) for i in range(len(second))
    ]

    sampler_a = EpochDeterministicSampler(first, 42)
    sampler_b = EpochDeterministicSampler(second, 42)
    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)
    assert list(sampler_a) == list(sampler_b)
