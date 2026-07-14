import json
from pathlib import Path

from PIL import Image

from bilinear_lmmd.prepare_clean_grouped_folds import (
    analyze_exact_duplicates,
    prepare_clean_grouped_folds,
)


def _image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def test_duplicate_analysis_keeps_same_class_and_quarantines_conflict(tmp_path):
    original = tmp_path / "A" / "one.png"
    same_class = tmp_path / "A" / "two.png"
    conflict = tmp_path / "B" / "three.png"
    unique = tmp_path / "B" / "unique.png"
    _image(original, (10, 20, 30))
    same_class.parent.mkdir(parents=True, exist_ok=True)
    same_class.write_bytes(original.read_bytes())
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(original.read_bytes())
    _image(unique, (50, 60, 70))
    images = {
        "A/one.png": original,
        "A/two.png": same_class,
        "B/three.png": conflict,
        "B/unique.png": unique,
    }

    kept, duplicates, conflicts = analyze_exact_duplicates(images)

    assert kept == {"B/unique.png": unique}
    assert duplicates == []
    assert conflicts[0]["quarantined"] == [
        "A/one.png",
        "A/two.png",
        "B/three.png",
    ]


def test_cleaner_builds_content_clean_outer_folds(tmp_path):
    source = tmp_path / "source"
    identities = []
    for class_index, class_name in enumerate(("A", "B")):
        for index in range(7):
            split = ("train", "val", "test")[index % 3]
            path = source / split / class_name / f"{index}.png"
            _image(path, (class_index * 100 + index, index, 200 - index))
            identities.append(path)
    duplicate = source / "test" / "A" / "duplicate.png"
    duplicate.write_bytes((source / "train" / "A" / "0.png").read_bytes())
    output = tmp_path / "clean"

    audit = prepare_clean_grouped_folds(
        source,
        output,
        expected_count=15,
        folds=3,
        validation_ratio=0.2,
    )

    assert audit["input_count"] == 15
    assert audit["clean_count"] == 14
    assert audit["removed_same_class_count"] == 1
    assert audit["quarantined_conflict_count"] == 0
    metadata = json.loads(
        (output / "folds" / "metadata.json").read_text(encoding="utf-8")
    )
    test_identities = [
        identity
        for fold in metadata["assignments"].values()
        for identity in fold["test"]
    ]
    assert len(test_identities) == len(set(test_identities)) == 14
    assert not (output / "clean_source" / "A" / "duplicate.png").exists()
