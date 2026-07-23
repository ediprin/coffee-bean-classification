from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from bilinear_lmmd.data.preparation.prepare_sni_classification_v2 import (
    ATTRIBUTE_NAMES,
    VISUAL_CLASSES,
    partial_attributes,
    prepare_sni_classification_v2,
    visual_label,
)
from bilinear_lmmd.data.sni_ontology import SNI_CLASSES


def _source_dataset(tmp_path: Path, *, leak_group: bool = False) -> Path:
    root = tmp_path / "sni-v1"
    rows = []
    crop_index = 0
    for split in ("train", "val", "test"):
        for class_name in SNI_CLASSES:
            for dataset in ("adrian_detection", "faruq_segmentation"):
                relative = (
                    Path("source") / split / class_name / f"{crop_index}.jpg"
                )
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16), (crop_index % 255, 10, 20)).save(path)
                rows.append(
                    {
                        "dataset": dataset,
                        "archive_split": split,
                        "generated_split": split,
                        "group_id": (
                            "leaking-group"
                            if leak_group and class_name == SNI_CLASSES[0]
                            else f"group-{crop_index}"
                        ),
                        "source_identity": f"source-{crop_index}",
                        "source_file": f"/raw/{crop_index}.jpg",
                        "source_sha256": f"source-hash-{crop_index}",
                        "image_id": crop_index,
                        "annotation_id": crop_index,
                        "original_class": class_name,
                        "canonical_class": class_name,
                        "bbox_x": 1,
                        "bbox_y": 1,
                        "bbox_width": 10,
                        "bbox_height": 12,
                        "crop_width": 16,
                        "crop_height": 16,
                        "crop_sha256": f"crop-hash-{crop_index}",
                        "crop_path": relative.as_posix(),
                    }
                )
                crop_index += 1
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "SNI_instance_crop_v1",
                "classes": list(SNI_CLASSES),
                "output_crops": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_visual_taxonomy_collapses_unobservable_size() -> None:
    assert visual_label("biji_hitam") == "biji_hitam"
    assert visual_label("kulit_kopi_ukuran_besar") == "kulit_kopi"
    assert visual_label("kulit_kopi_ukuran_kecil") == "kulit_kopi"
    assert visual_label("tanah_batu_ranting_sedang") == "benda_asing"
    assert len(VISUAL_CLASSES) == 15


def test_partial_attributes_do_not_invent_negative_labels() -> None:
    compound = partial_attributes("biji_hitam_pecah")
    assert compound["black"] == 1
    assert compound["broken"] == 1
    assert compound["normal"] == 0
    assert compound["spotted"] == -1

    normal = partial_attributes("biji_normal")
    assert normal["normal"] == 1
    assert all(normal[name] == 0 for name in ATTRIBUTE_NAMES if name != "normal")


def test_prepare_v2_writes_weighted_and_cross_domain_manifests(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path)
    output = tmp_path / "v2"
    audit = prepare_sni_classification_v2(
        source,
        output,
        min_eval_samples_per_class=1,
        min_eval_groups_per_class=1,
    )

    assert audit["status"] == "complete"
    assert audit["label_design"]["visual_v2_num_classes"] == 15
    assert audit["integrity"]["cross_split_groups"] == 0
    assert audit["training_balance"]["files_duplicated_or_deleted"] is False
    assert audit["statistical_readiness"]["strong_per_class_claim_ready"] is True
    assert audit["training_authorized"] is False
    assert (output / "ontology.json").is_file()
    assert (output / "manifests/train_weighted.csv").is_file()
    assert (
        output
        / "cross_domain/adrian_detection_to_faruq_segmentation/external_test.csv"
    ).is_file()

    with (output / "manifests/all.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    size_row = next(
        row for row in rows if row["flat_label_v1"] == "kulit_kopi_ukuran_besar"
    )
    assert size_row["visual_label"] == "kulit_kopi"
    assert size_row["size_label_metadata"] == "besar"
    assert size_row["size_visual_supervision_allowed"] == "0"


def test_prepare_v2_rejects_source_group_leakage(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path, leak_group=True)
    with pytest.raises(ValueError, match="cross_split_groups"):
        prepare_sni_classification_v2(source, tmp_path / "v2")
