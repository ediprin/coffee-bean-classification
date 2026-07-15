import json
from pathlib import Path

from PIL import Image

from bilinear_lmmd.prepare_usk_coffee import (
    CLASSES,
    discover_samples,
    prepare_usk_coffee,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_usk_preparer_discovers_classes_deduplicates_and_splits(tmp_path):
    raw = tmp_path / "raw"
    for class_index, class_name in enumerate(CLASSES):
        for index in range(8):
            _image(
                raw / "nested" / class_name.lower() / f"{index}.png",
                (class_index * 40 + index, index, 200 - class_index),
            )
    duplicate = raw / "nested" / "defect" / "duplicate.png"
    duplicate.write_bytes((raw / "nested" / "defect" / "0.png").read_bytes())
    _image(raw / "nested" / "peaberry" / "bean_front.png", (211, 1, 1))
    _image(raw / "nested" / "peaberry" / "bean_back.png", (212, 1, 1))

    output = tmp_path / "prepared"
    audit = prepare_usk_coffee(raw, output, seed=17)

    assert audit["input_images"] == 35
    assert audit["clean_images"] == 34
    assert audit["split_strategy"] == "generated_60_20_20"
    assert len(audit["same_class_exact_duplicates"]) == 1
    assert audit["detected_view_pairs"] == 1
    assert audit["cross_split_view_pairs"] == {}
    assert set(audit["class_counts"]) == set(CLASSES)
    assert sum(
        len(list((output / "source" / split).glob("*/*")))
        for split in ("train", "val", "test")
    ) == 34

    front = next((output / "source").rglob("*bean_front.png"))
    back = next((output / "source").rglob("*bean_back.png"))
    assert front.parents[1].name == back.parents[1].name


def test_usk_preparer_preserves_archive_splits_and_reports_view_leakage(tmp_path):
    raw = tmp_path / "raw"
    value = 1
    for split in ("train", "valid", "test"):
        for class_name in CLASSES:
            _image(raw / split / class_name / f"{split}.png", (value, value, value))
            value += 1
    _image(raw / "train" / "Peaberry" / "paired_front.png", (100, 1, 1))
    _image(raw / "test" / "Peaberry" / "paired_back.png", (101, 1, 1))

    audit = prepare_usk_coffee(raw, tmp_path / "prepared")

    assert audit["split_strategy"] == "preserved_from_archive"
    assert len(audit["cross_split_view_pairs"]) == 1
    assert sum(audit["split_counts"]["val"].values()) == 4


def test_usk_discovery_ignores_unlabelled_images(tmp_path):
    raw = tmp_path / "raw"
    for index, class_name in enumerate(CLASSES):
        _image(raw / class_name / "one.png", (index, 10, 20))
    _image(raw / "misc" / "preview.png", (99, 99, 99))

    samples, ignored = discover_samples(raw)

    assert len(samples) == 4
    assert ignored == ["misc/preview.png"]


def test_usk_audit_is_idempotent(tmp_path):
    raw = tmp_path / "raw"
    for index, class_name in enumerate(CLASSES):
        _image(raw / class_name / "one.png", (index, 20, 30))
    output = tmp_path / "prepared"

    first = prepare_usk_coffee(raw, output)
    second = prepare_usk_coffee(raw, output)

    assert first == second
    assert json.loads((output / "audit.json").read_text()) == first
