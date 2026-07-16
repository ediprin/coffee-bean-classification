from pathlib import Path

from PIL import Image

from bilinear_lmmd.prepare_cbd_multiclassify import (
    EXPECTED_CLASSES,
    discover_samples,
    prepare_cbd_multiclassify,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_discovery_reads_roboflow_split_and_excludes_unlabeled(tmp_path):
    raw = tmp_path / "raw"
    for index, class_name in enumerate(EXPECTED_CLASSES):
        _image(raw / "train" / class_name / "sample.png", (index, 1, 2))
    _image(raw / "valid" / "Unlabeled" / "unknown.png", (90, 1, 2))

    samples, unlabeled, unknown = discover_samples(raw)

    assert {sample.class_name for sample in samples} == set(EXPECTED_CLASSES)
    assert {sample.archive_split for sample in samples} == {"train"}
    assert unlabeled == ["valid/Unlabeled/unknown.png"]
    assert unknown == []


def test_preparer_groups_roboflow_variants_and_deduplicates(tmp_path):
    raw = tmp_path / "raw"
    color = 1
    for class_name in EXPECTED_CLASSES:
        for identity in range(5):
            _image(
                raw / "train" / class_name / f"bean-{identity}.rf.aaa{identity}.png",
                (color, identity, 3),
            )
            color += 1
        _image(
            raw / "test" / class_name / "bean-0.rf.bbb0.png",
            (color, 10, 4),
        )
        color += 1
    duplicate = raw / "valid" / "Black" / "duplicate.png"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes(
        (raw / "train" / "Black" / "bean-1.rf.aaa1.png").read_bytes()
    )
    _image(raw / "train" / "Unlabeled" / "skip.png", (250, 1, 1))

    output = tmp_path / "prepared"
    audit = prepare_cbd_multiclassify(raw, output, seed=9)

    assert audit["excluded_unlabeled_count"] == 1
    assert len(audit["same_class_exact_duplicates"]) == 1
    assert audit["archive_cross_split_identity_count"] == len(EXPECTED_CLASSES)
    assert audit["generated_cross_split_identity_count"] == 0
    assert audit["split_strategy"] == "generated_identity_grouped_60_20_20"
    assert audit["clean_labeled_images"] == len(EXPECTED_CLASSES) * 6
    assert all(
        set(audit["split_counts"][split]) == set(EXPECTED_CLASSES)
        for split in ("train", "val", "test")
    )

    for class_name in EXPECTED_CLASSES:
        matching_splits = []
        for split in ("train", "val", "test"):
            names = [path.name for path in (output / "source" / split / class_name).glob("*")]
            if any("bean-0.rf." in name for name in names):
                matching_splits.append(split)
        assert len(matching_splits) == 1
