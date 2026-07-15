from pathlib import Path

from PIL import Image

from bilinear_lmmd.prepare_roast_coffee import (
    CLASSES,
    discover_roast_samples,
    prepare_roast_coffee,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_roast_preparer_preserves_test_and_carves_validation_from_train(tmp_path):
    raw = tmp_path / "raw"
    color = 1
    for class_name in CLASSES:
        for index in range(5):
            _image(raw / "train" / class_name / f"train-{index}.png", (color, 1, 1))
            color += 1
        for index in range(2):
            _image(raw / "test" / class_name / f"test-{index}.png", (color, 2, 2))
            color += 1

    output = tmp_path / "prepared"
    audit = prepare_roast_coffee(raw, output, seed=19, validation_ratio=0.2)

    assert audit["input_images"] == 28
    assert audit["clean_images"] == 28
    assert audit["split_strategy"] == "preserved_test_train_to_val_0.20"
    assert audit["split_counts"]["train"] == {name: 4 for name in CLASSES}
    assert audit["split_counts"]["val"] == {name: 1 for name in CLASSES}
    assert audit["split_counts"]["test"] == {name: 2 for name in CLASSES}
    assert len(list((output / "source" / "test").glob("*/*"))) == 8


def test_roast_preparer_preserves_existing_validation(tmp_path):
    raw = tmp_path / "raw"
    color = 1
    for split in ("train", "valid", "test"):
        for class_name in CLASSES:
            _image(raw / split / class_name / "one.png", (color, 3, 4))
            color += 1

    audit = prepare_roast_coffee(raw, tmp_path / "prepared")

    assert audit["split_strategy"] == "preserved_archive_train_val_test"
    assert all(sum(audit["split_counts"][split].values()) == 4 for split in ("train", "val", "test"))


def test_roast_preparer_reports_cross_split_exact_duplicate(tmp_path):
    raw = tmp_path / "raw"
    color = 1
    for split in ("train", "test"):
        for class_name in CLASSES:
            _image(raw / split / class_name / "one.png", (color, 5, 6))
            color += 1
    duplicate = raw / "test" / "Dark" / "duplicate.png"
    duplicate.write_bytes((raw / "train" / "Dark" / "one.png").read_bytes())

    audit = prepare_roast_coffee(raw, tmp_path / "prepared")

    assert len(audit["cross_split_exact_duplicates"]) == 1
    record = audit["cross_split_exact_duplicates"][0]
    assert record["kept_split"] == "test"
    assert audit["deduplication_policy"] == "keep_test_then_val_then_train"
    assert not list((tmp_path / "prepared" / "source" / "train" / "Dark").glob("*"))
    assert len(list((tmp_path / "prepared" / "source" / "test" / "Dark").glob("*"))) == 2


def test_roast_discovery_ignores_images_outside_known_split(tmp_path):
    raw = tmp_path / "raw"
    for index, class_name in enumerate(CLASSES):
        _image(raw / "train" / class_name / "one.png", (index, 7, 8))
        _image(raw / "test" / class_name / "one.png", (index + 10, 7, 8))
    _image(raw / "preview" / "Dark" / "preview.png", (99, 9, 9))

    samples, ignored = discover_roast_samples(raw)

    assert len(samples) == 8
    assert ignored == ["preview/Dark/preview.png"]
