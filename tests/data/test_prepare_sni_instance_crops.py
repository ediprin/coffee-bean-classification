from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from bilinear_lmmd.data.preparation.prepare_sni_instance_crops import (
    CANONICAL_CLASSES,
    ImageRecord,
    InstanceRecord,
    allocate_groups,
    canonical_class,
    ensure_imagefolder_directories,
    orient_to_coco_size,
    source_identity,
    square_crop,
    valid_polygon_segmentation,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Biji tanpa cacat", "biji_normal"),
        ("biji_kulit_tanduk", "biji_berkulit_tanduk"),
        ("Biji berlubang lebih dari satu", "biji_berlubang_lebih_satu"),
        ("Batu berukuran besar", "tanah_batu_ranting_besar"),
        ("Ranting berukuran kecil", "tanah_batu_ranting_kecil"),
        ("tanah_batu_ranting_sedang", "tanah_batu_ranting_sedang"),
    ],
)
def test_canonical_class_maps_both_public_taxonomies(label, expected):
    assert canonical_class(label) == expected


def test_canonical_class_rejects_unknown_label():
    with pytest.raises(ValueError, match="tidak memiliki pemetaan SNI"):
        canonical_class("kelas rekaan")


def test_source_identity_removes_roboflow_export_hash():
    first = source_identity("bean_17_png.rf.0123ABC-def.jpg")
    second = source_identity("bean_17_png.rf.9876fed-cba.jpg")
    assert first == second == "bean17png"


def test_square_crop_is_square_and_pads_at_image_edge():
    image = Image.new("RGB", (20, 10), (10, 20, 30))
    crop = square_crop(image, (-0.01, 0.0, 5.0, 4.0), margin_fraction=0.1)
    assert crop.size[0] == crop.size[1]
    assert crop.size == (6, 6)
    assert crop.getpixel((0, 0)) == (10, 20, 30)


@pytest.mark.parametrize(
    ("segmentation", "expected"),
    [
        ([[0, 0, 2, 0, 2, 2]], True),
        ([], False),
        ([[0, 0, 2, 0]], False),
        ([[0, 0, 2, 0, float("nan"), 2]], False),
        ({"counts": [], "size": [2, 2]}, False),
    ],
)
def test_polygon_segmentation_validation(segmentation, expected):
    assert valid_polygon_segmentation(segmentation) is expected


def test_exif_orientation_is_applied_before_coco_crop(tmp_path):
    path = tmp_path / "portrait.jpg"
    image = Image.new("RGB", (8, 4), (10, 20, 30))
    exif = image.getexif()
    exif[274] = 6  # stored landscape, displayed portrait
    image.save(path, exif=exif)

    with Image.open(path) as opened:
        oriented, changed = orient_to_coco_size(opened, (4, 8))
    assert changed is True
    assert oriented.size == (4, 8)


def test_coco_dimension_mismatch_is_rejected():
    image = Image.new("RGB", (8, 4))
    with pytest.raises(ValueError, match="tidak cocok dengan metadata COCO"):
        orient_to_coco_size(image, (7, 4))


def test_imagefolder_directory_creation_is_resume_safe(tmp_path):
    output = tmp_path / "partial"
    ensure_imagefolder_directories(output)
    ensure_imagefolder_directories(output)
    for split in ("train", "val", "test"):
        for class_name in CANONICAL_CLASSES:
            assert (output / "source" / split / class_name).is_dir()


def _synthetic_records():
    images = []
    instances = []
    group_ids = {}
    image_index = 0
    # Four independent photographs per class are sufficient to exercise all
    # split paths while keeping the test small. Two instances in each photo
    # verify that an image group is never divided.
    for class_name in CANONICAL_CLASSES:
        for repetition in range(4):
            uid = f"image-{image_index}"
            group = f"group-{image_index}"
            images.append(
                ImageRecord(
                    uid=uid,
                    dataset="synthetic",
                    archive_split="train",
                    image_id=image_index,
                    file_name=f"{uid}.jpg",
                    path=Path(f"/{uid}.jpg"),
                    width=32,
                    height=32,
                    source_identity=uid,
                    sha256=uid,
                )
            )
            group_ids[uid] = group
            for annotation in range(2):
                instances.append(
                    InstanceRecord(
                        uid=f"{uid}:{annotation}",
                        image_uid=uid,
                        annotation_id=annotation,
                        original_class=class_name,
                        canonical_class=class_name,
                        bbox=(2.0, 2.0, 20.0, 20.0),
                    )
                )
            image_index += 1
    return images, instances, group_ids


def test_grouped_allocator_is_deterministic_and_keeps_images_together():
    images, instances, group_ids = _synthetic_records()
    first = allocate_groups(images, instances, group_ids, seed=42)
    second = allocate_groups(images, instances, group_ids, seed=42)
    assert first == second
    assert set(first.values()) == {"train", "val", "test"}

    counts = {split: Counter() for split in ("train", "val", "test")}
    for instance in instances:
        split = first[group_ids[instance.image_uid]]
        counts[split][instance.canonical_class] += 1
    # With only four source groups per class an exact 70/15/15 split is not
    # possible, but each held-out split must still contain every class.
    for split_counts in counts.values():
        assert set(split_counts) == set(CANONICAL_CLASSES)
