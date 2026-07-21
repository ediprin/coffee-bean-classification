from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat


SPLITS = ("train", "val", "test")
ARCHIVE_SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
CANONICAL_CLASSES = (
    "biji_berkulit_tanduk",
    "biji_berlubang_lebih_satu",
    "biji_berlubang_satu",
    "biji_bertutul_tutul",
    "biji_coklat",
    "biji_hitam",
    "biji_hitam_pecah",
    "biji_hitam_sebagian",
    "biji_muda",
    "biji_normal",
    "biji_pecah",
    "kopi_gelondong",
    "kulit_kopi_ukuran_besar",
    "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang",
    "kulit_tanduk_ukuran_besar",
    "kulit_tanduk_ukuran_kecil",
    "kulit_tanduk_ukuran_sedang",
    "tanah_batu_ranting_besar",
    "tanah_batu_ranting_kecil",
    "tanah_batu_ranting_sedang",
)
SNI_DEFECT_WEIGHTS = {
    "biji_hitam": 1.0,
    "biji_hitam_sebagian": 0.5,
    "biji_hitam_pecah": 0.5,
    "kopi_gelondong": 1.0,
    "biji_coklat": 0.25,
    "kulit_kopi_ukuran_besar": 1.0,
    "kulit_kopi_ukuran_sedang": 0.5,
    "kulit_kopi_ukuran_kecil": 0.2,
    "biji_berkulit_tanduk": 0.5,
    "kulit_tanduk_ukuran_besar": 0.5,
    "kulit_tanduk_ukuran_sedang": 0.2,
    "kulit_tanduk_ukuran_kecil": 0.1,
    "biji_pecah": 0.2,
    "biji_muda": 0.2,
    "biji_berlubang_satu": 0.1,
    "biji_berlubang_lebih_satu": 0.2,
    "biji_bertutul_tutul": 0.1,
    "tanah_batu_ranting_besar": 5.0,
    "tanah_batu_ranting_sedang": 2.0,
    "tanah_batu_ranting_kecil": 1.0,
    "biji_normal": 0.0,
}
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[a-z0-9_-]+$", re.IGNORECASE)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _aliases() -> dict[str, str]:
    aliases = {_normalized(name): name for name in CANONICAL_CLASSES}
    aliases.update(
        {
            "bijitanpacacat": "biji_normal",
            "bijinormal": "biji_normal",
            "bijicokelat": "biji_coklat",
            "bijicoklat": "biji_coklat",
            "bijigelondong": "kopi_gelondong",
            "kopigelondong": "kopi_gelondong",
            "bijikulittanduk": "biji_berkulit_tanduk",
            "bijiberkulittanduk": "biji_berkulit_tanduk",
            "bijiberlubanglebihdarisatu": "biji_berlubang_lebih_satu",
            "bijiberlubanglebihsatu": "biji_berlubang_lebih_satu",
        }
    )
    for size in ("besar", "sedang", "kecil"):
        canonical = f"tanah_batu_ranting_{size}"
        for material in ("tanah", "batu", "ranting"):
            aliases[_normalized(f"{material} berukuran {size}")] = canonical
        aliases[_normalized(f"tanah batu ranting {size}")] = canonical
    return aliases


CLASS_ALIASES = _aliases()


def canonical_class(label: str) -> str:
    normalized = _normalized(label)
    if normalized not in CLASS_ALIASES:
        raise ValueError(f"Label tidak memiliki pemetaan SNI: {label!r}")
    return CLASS_ALIASES[normalized]


def source_identity(file_name: str) -> str:
    stem = Path(file_name).stem
    return _normalized(ROBOFLOW_SUFFIX.sub("", stem))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_polygon_segmentation(segmentation: object) -> bool:
    if not isinstance(segmentation, list) or not segmentation:
        return False
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            return False
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in polygon):
            return False
    return True


@dataclass(frozen=True)
class ImageRecord:
    uid: str
    dataset: str
    archive_split: str
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    source_identity: str
    sha256: str


@dataclass(frozen=True)
class InstanceRecord:
    uid: str
    image_uid: str
    annotation_id: int
    original_class: str
    canonical_class: str
    bbox: tuple[float, float, float, float]


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            keep, merge = sorted((first_root, second_root))
            self.parent[merge] = keep


def _find_coco_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Root ekspor COCO tidak ditemukan: {root}")
    found: dict[str, Path] = {}
    for path in root.rglob("_annotations.coco.json"):
        parent = _normalized(path.parent.name)
        if parent not in ARCHIVE_SPLITS:
            continue
        split = ARCHIVE_SPLITS[parent]
        if split in found:
            raise ValueError(f"Lebih dari satu anotasi untuk split {split}: {root}")
        found[split] = path
    missing = sorted(set(SPLITS).difference(found))
    if missing:
        raise FileNotFoundError(f"Split COCO belum lengkap pada {root}: {missing}")
    return found


def _resolve_image(annotation_path: Path, file_name: str) -> Path:
    direct = annotation_path.parent / file_name
    if direct.is_file():
        return direct
    candidates = list(annotation_path.parent.rglob(Path(file_name).name))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Gambar {file_name!r} tidak ditemukan secara unik dekat {annotation_path}"
        )
    return candidates[0]


def read_coco_dataset(
    dataset: str,
    root: Path,
) -> tuple[list[ImageRecord], list[InstanceRecord], dict]:
    images: list[ImageRecord] = []
    instances: list[InstanceRecord] = []
    invalid_boxes: list[dict] = []
    invalid_segmentations: list[dict] = []
    segmentation_instances = 0
    observed_original: Counter[str] = Counter()
    archive_counts: dict[str, dict] = {}
    for split, annotation_path in sorted(_find_coco_files(root).items()):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(row["id"]): str(row["name"]) for row in data["categories"]}
        image_rows = {int(row["id"]): row for row in data["images"]}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in data["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        split_instances = 0
        for image_id, image_row in sorted(image_rows.items()):
            image_path = _resolve_image(annotation_path, str(image_row["file_name"]))
            uid = f"{dataset}:{split}:{image_id}"
            image = ImageRecord(
                uid=uid,
                dataset=dataset,
                archive_split=split,
                image_id=image_id,
                file_name=str(image_row["file_name"]),
                path=image_path,
                width=int(image_row["width"]),
                height=int(image_row["height"]),
                source_identity=source_identity(str(image_row["file_name"])),
                sha256=_sha256(image_path),
            )
            images.append(image)
            for annotation in annotations_by_image.get(image_id, []):
                annotation_id = int(annotation["id"])
                original = categories[int(annotation["category_id"])]
                mapped = canonical_class(original)
                bbox = annotation.get("bbox")
                if dataset == "faruq_segmentation":
                    segmentation = annotation.get("segmentation")
                    if valid_polygon_segmentation(segmentation):
                        segmentation_instances += 1
                    else:
                        invalid_segmentations.append(
                            {
                                "image": str(image_path),
                                "annotation_id": annotation_id,
                            }
                        )
                if not isinstance(bbox, list) or len(bbox) != 4:
                    invalid_boxes.append(
                        {"image": str(image_path), "annotation_id": annotation_id, "bbox": bbox}
                    )
                    continue
                x, y, width, height = (float(value) for value in bbox)
                tolerance = 0.02
                valid = (
                    width > 0
                    and height > 0
                    and x >= -tolerance
                    and y >= -tolerance
                    and x + width <= image.width + tolerance
                    and y + height <= image.height + tolerance
                )
                if not valid:
                    invalid_boxes.append(
                        {"image": str(image_path), "annotation_id": annotation_id, "bbox": bbox}
                    )
                    continue
                instances.append(
                    InstanceRecord(
                        uid=f"{uid}:{annotation_id}",
                        image_uid=uid,
                        annotation_id=annotation_id,
                        original_class=original,
                        canonical_class=mapped,
                        bbox=(x, y, width, height),
                    )
                )
                observed_original[original] += 1
                split_instances += 1
        archive_counts[split] = {
            "images": len(image_rows),
            "instances": split_instances,
        }
    observed_canonical = {instance.canonical_class for instance in instances}
    missing = sorted(set(CANONICAL_CLASSES).difference(observed_canonical))
    if missing:
        raise ValueError(f"Dataset {dataset} tidak mencakup kelas canonical: {missing}")
    if invalid_boxes:
        raise ValueError(
            f"Dataset {dataset} memiliki {len(invalid_boxes)} bounding box tidak valid; "
            "lihat contoh pada audit internal sebelum menyiapkan crop."
        )
    if invalid_segmentations:
        raise ValueError(
            f"Dataset {dataset} memiliki {len(invalid_segmentations)} polygon "
            "segmentasi tidak valid atau kosong."
        )
    audit = {
        "root": str(root.resolve()),
        "images": len(images),
        "instances": len(instances),
        "archive_counts": archive_counts,
        "original_class_counts": dict(sorted(observed_original.items())),
        "invalid_box_count": len(invalid_boxes),
        "invalid_box_examples": invalid_boxes[:100],
        "valid_polygon_segmentations": segmentation_instances,
        "invalid_polygon_segmentation_count": len(invalid_segmentations),
    }
    return images, instances, audit


def group_images(images: list[ImageRecord]) -> tuple[dict[str, str], dict]:
    union = _UnionFind([image.uid for image in images])
    by_identity: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in images:
        by_identity[(image.dataset, image.source_identity)].append(image)
        by_hash[image.sha256].append(image)
    for rows in list(by_identity.values()) + list(by_hash.values()):
        for row in rows[1:]:
            union.union(rows[0].uid, row.uid)
    members: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in images:
        members[union.find(image.uid)].append(image)
    group_ids: dict[str, str] = {}
    for rows in members.values():
        identity = "|".join(sorted(image.uid for image in rows))
        group_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        for image in rows:
            group_ids[image.uid] = group_id
    archive_leaks = []
    for rows in members.values():
        archive_splits = sorted({image.archive_split for image in rows})
        if len(archive_splits) > 1:
            archive_leaks.append(
                {
                    "datasets": sorted({image.dataset for image in rows}),
                    "source_identities": sorted({image.source_identity for image in rows}),
                    "archive_splits": archive_splits,
                    "files": sorted(str(image.path) for image in rows),
                }
            )
    exact_duplicates = [
        {
            "sha256": digest,
            "datasets": sorted({image.dataset for image in rows}),
            "files": sorted(str(image.path) for image in rows),
        }
        for digest, rows in by_hash.items()
        if len(rows) > 1
    ]
    return group_ids, {
        "identity_groups": len(members),
        "archive_cross_split_identity_count": len(archive_leaks),
        "archive_cross_split_identity_examples": archive_leaks[:100],
        "exact_duplicate_groups": len(exact_duplicates),
        "cross_dataset_exact_duplicate_groups": sum(
            len(row["datasets"]) > 1 for row in exact_duplicates
        ),
        "exact_duplicate_examples": exact_duplicates[:100],
    }


def allocate_groups(
    images: list[ImageRecord],
    instances: list[InstanceRecord],
    group_ids: dict[str, str],
    seed: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Rasio split harus berjumlah satu.")
    image_by_uid = {image.uid: image for image in images}
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for instance in instances:
        image = image_by_uid[instance.image_uid]
        group = group_ids[instance.image_uid]
        group_counts[group][f"class:{instance.canonical_class}"] += 1
        group_counts[group][f"dataset:{image.dataset}"] += 1
        group_counts[group]["total"] += 1
    totals: Counter[str] = Counter()
    for counts in group_counts.values():
        totals.update(counts)
    targets = {
        split: {key: totals[key] * ratio for key in totals}
        for split, ratio in zip(SPLITS, ratios)
    }
    assigned = {split: Counter() for split in SPLITS}
    objective_weights = {
        key: (8.0 if key == "total" else 1.0)
        for key in totals
    }
    generator = random.Random(seed)
    groups = list(group_counts)
    jitter = {group: generator.random() for group in groups}

    def rarity(group: str) -> tuple[float, int, float]:
        counts = group_counts[group]
        score = max(
            (value / totals[key] for key, value in counts.items() if key.startswith("class:")),
            default=0.0,
        )
        return (-score, -counts["total"], jitter[group])

    groups.sort(key=rarity)
    assignments: dict[str, str] = {}
    for group in groups:
        counts = group_counts[group]
        candidate_costs = []
        for split in SPLITS:
            # Only this split changes for a candidate assignment. Comparing the
            # *increment* in global squared error is therefore equivalent to
            # recomputing the full objective, without repeatedly summing the
            # two unchanged splits. Using the absolute post-assignment error
            # here would incorrectly favour the smaller validation/test targets.
            cost_delta = 0.0
            for key, total in totals.items():
                target = targets[split][key]
                before = assigned[split][key]
                after = before + counts[key]
                denominator = max(target, 1.0)
                cost_delta += objective_weights[key] * (
                    ((after - target) / denominator) ** 2
                    - ((before - target) / denominator) ** 2
                )
            candidate_costs.append((cost_delta, SPLITS.index(split), split))
        split = min(candidate_costs)[2]
        assignments[group] = split
        assigned[split].update(counts)

    # Greedy placement is order-sensitive for dense multi-object photographs.
    # Refine it with deterministic single-group moves until the same objective
    # no longer improves. This keeps the 70/15/15 targets much closer without
    # ever splitting instances originating from one photograph.
    refinement_order = groups.copy()
    generator.shuffle(refinement_order)
    for _ in range(12):
        moved = False
        for group in refinement_order:
            current = assignments[group]
            counts = group_counts[group]
            candidate_moves: list[tuple[float, int, str]] = []
            for candidate in SPLITS:
                if candidate == current:
                    continue
                cost_delta = 0.0
                for key in totals:
                    current_target = targets[current][key]
                    candidate_target = targets[candidate][key]
                    current_before = assigned[current][key]
                    candidate_before = assigned[candidate][key]
                    value = counts[key]
                    current_denominator = max(current_target, 1.0)
                    candidate_denominator = max(candidate_target, 1.0)
                    cost_delta += objective_weights[key] * (
                        ((current_before - value - current_target) / current_denominator) ** 2
                        - ((current_before - current_target) / current_denominator) ** 2
                        + ((candidate_before + value - candidate_target) / candidate_denominator)
                        ** 2
                        - ((candidate_before - candidate_target) / candidate_denominator) ** 2
                    )
                candidate_moves.append((cost_delta, SPLITS.index(candidate), candidate))
            best_delta, _, best_split = min(candidate_moves)
            if best_delta < -1e-12:
                assigned[current].subtract(counts)
                assigned[best_split].update(counts)
                assignments[group] = best_split
                moved = True
        if not moved:
            break

    # Guarantee loader-compatible splits: every class must occur at least once
    # in train, validation, and test whenever the grouped data make that
    # feasible. The move may be slightly less ratio-optimal, which is preferable
    # to an empty ImageFolder class or an undefined per-class F1.
    for split in SPLITS:
        for class_name in CANONICAL_CLASSES:
            class_key = f"class:{class_name}"
            if assigned[split][class_key] > 0:
                continue
            repair_candidates: list[tuple[float, str, str]] = []
            for group, current in assignments.items():
                counts = group_counts[group]
                if current == split or counts[class_key] == 0:
                    continue
                # Do not repair one split by creating an empty class in another.
                if any(
                    counts[key] > 0 and assigned[current][key] - counts[key] <= 0
                    for key in counts
                    if key.startswith("class:")
                ):
                    continue
                cost_delta = 0.0
                for key in totals:
                    current_target = targets[current][key]
                    split_target = targets[split][key]
                    current_before = assigned[current][key]
                    split_before = assigned[split][key]
                    value = counts[key]
                    current_denominator = max(current_target, 1.0)
                    split_denominator = max(split_target, 1.0)
                    cost_delta += objective_weights[key] * (
                        ((current_before - value - current_target) / current_denominator) ** 2
                        - ((current_before - current_target) / current_denominator) ** 2
                        + ((split_before + value - split_target) / split_denominator) ** 2
                        - ((split_before - split_target) / split_denominator) ** 2
                    )
                repair_candidates.append((cost_delta, group, current))
            if not repair_candidates:
                raise ValueError(
                    f"Tidak dapat menempatkan kelas {class_name!r} pada split {split!r} "
                    "tanpa memecah grup foto sumber."
                )
            _, group, current = min(repair_candidates)
            counts = group_counts[group]
            assigned[current].subtract(counts)
            assigned[split].update(counts)
            assignments[group] = split
    return assignments


def square_crop(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    margin_fraction: float,
) -> Image.Image:
    if not 0.0 <= margin_fraction <= 1.0:
        raise ValueError("margin_fraction harus berada pada [0, 1].")
    image = image.convert("RGB")
    x, y, width, height = bbox
    side = max(1, math.ceil(max(width, height) * (1.0 + 2.0 * margin_fraction)))
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    right = left + side
    bottom = top + side
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[0] >= source_box[2] or source_box[1] >= source_box[3]:
        raise ValueError(f"Bounding box tidak beririsan dengan gambar: {bbox}")
    average = tuple(round(value) for value in ImageStat.Stat(image.resize((1, 1))).mean)
    output = Image.new("RGB", (side, side), average)
    crop = image.crop(source_box)
    output.paste(crop, (source_box[0] - left, source_box[1] - top))
    return output


def _contact_sheets(output_root: Path, manifest: list[dict]) -> list[str]:
    outputs: list[str] = []
    for dataset in sorted({row["dataset"] for row in manifest}):
        by_class: dict[str, list[Path]] = defaultdict(list)
        for row in manifest:
            if row["dataset"] == dataset and row["generated_split"] == "train":
                by_class[row["canonical_class"]].append(output_root / row["crop_path"])
        cell = 112
        label_width = 245
        samples_per_class = 3
        sheet = Image.new(
            "RGB",
            (label_width + samples_per_class * cell, len(CANONICAL_CLASSES) * cell),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, class_name in enumerate(CANONICAL_CLASSES):
            top = row_index * cell
            draw.text((6, top + 44), class_name, fill="black")
            for column, path in enumerate(sorted(by_class[class_name])[:samples_per_class]):
                with Image.open(path) as opened:
                    thumbnail = opened.convert("RGB")
                    thumbnail.thumbnail((cell - 4, cell - 4), Image.Resampling.LANCZOS)
                    x = label_width + column * cell + (cell - thumbnail.width) // 2
                    y = top + (cell - thumbnail.height) // 2
                    sheet.paste(thumbnail, (x, y))
        destination = output_root / f"contact_sheet_{dataset}.jpg"
        sheet.save(destination, quality=90)
        outputs.append(str(destination.relative_to(output_root)))
    return outputs


def prepare_sni_instance_crops(
    adrian_root: Path,
    faruq_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    margin_fraction: float = 0.10,
    jpeg_quality: int = 95,
) -> dict:
    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: crop SNI sudah lengkap: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"Output parsial ditemukan: {output_root}. Pindahkan atau hapus manual."
        )
    datasets = {"adrian_detection": adrian_root, "faruq_segmentation": faruq_root}
    images: list[ImageRecord] = []
    instances: list[InstanceRecord] = []
    source_audits = {}
    for name, root in datasets.items():
        source_images, source_instances, source_audit = read_coco_dataset(name, root)
        images.extend(source_images)
        instances.extend(source_instances)
        source_audits[name] = source_audit
    image_by_uid = {image.uid: image for image in images}
    instances_by_image: dict[str, list[InstanceRecord]] = defaultdict(list)
    for instance in instances:
        instances_by_image[instance.image_uid].append(instance)
    group_ids, identity_audit = group_images(images)
    assignments = allocate_groups(images, instances, group_ids, seed)
    output_root.mkdir(parents=True)
    for split in SPLITS:
        for class_name in CANONICAL_CLASSES:
            (output_root / "source" / split / class_name).mkdir(parents=True)
    manifest: list[dict] = []
    crop_hashes: dict[str, list[str]] = defaultdict(list)
    for image_index, image_record in enumerate(images, start=1):
        image_instances = instances_by_image.get(image_record.uid, [])
        if not image_instances:
            continue
        split = assignments[group_ids[image_record.uid]]
        with Image.open(image_record.path) as opened:
            image = opened.convert("RGB")
            for instance in image_instances:
                crop = square_crop(image, instance.bbox, margin_fraction)
                filename = (
                    f"{image_record.dataset}__{image_record.archive_split}__"
                    f"{image_record.image_id:07d}__{instance.annotation_id:08d}.jpg"
                )
                destination = output_root / "source" / split / instance.canonical_class / filename
                crop.save(destination, quality=jpeg_quality, subsampling=0)
                digest = _sha256(destination)
                relative = destination.relative_to(output_root).as_posix()
                crop_hashes[digest].append(relative)
                x, y, width, height = instance.bbox
                manifest.append(
                    {
                        "dataset": image_record.dataset,
                        "archive_split": image_record.archive_split,
                        "generated_split": split,
                        "group_id": group_ids[image_record.uid],
                        "source_identity": image_record.source_identity,
                        "source_file": str(image_record.path),
                        "source_sha256": image_record.sha256,
                        "image_id": image_record.image_id,
                        "annotation_id": instance.annotation_id,
                        "original_class": instance.original_class,
                        "canonical_class": instance.canonical_class,
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": width,
                        "bbox_height": height,
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "crop_sha256": digest,
                        "crop_path": relative,
                    }
                )
        if image_index % 250 == 0 or image_index == len(images):
            print(
                f"CROP {image_index}/{len(images)} images | {len(manifest)} instances",
                flush=True,
            )

    # Exact duplicate crops can still arise from separately encoded source
    # photographs. Conflicting labels are unsafe and remain fatal. Same-label
    # duplicates are removed deterministically, retaining test before val before
    # train so no test sample is silently moved into model development data.
    manifest_by_path = {row["crop_path"]: row for row in manifest}
    original_duplicate_crop_groups = [
        paths for paths in crop_hashes.values() if len(paths) > 1
    ]
    conflicting_duplicate_crops = [
        paths
        for paths in original_duplicate_crop_groups
        if len({manifest_by_path[path]["canonical_class"] for path in paths}) > 1
    ]
    if conflicting_duplicate_crops:
        failure = {
            "status": "failed_conflicting_crop_labels",
            "groups": conflicting_duplicate_crops[:100],
        }
        (output_root / "audit_failed.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            "Crop identik memiliki label berbeda. Dataset tidak aman; "
            "lihat audit_failed.json."
        )
    split_priority = {"test": 0, "val": 1, "train": 2}
    removed_duplicate_paths: list[str] = []
    for paths in original_duplicate_crop_groups:
        ordered = sorted(
            paths,
            key=lambda path: (
                split_priority[manifest_by_path[path]["generated_split"]],
                path,
            ),
        )
        removed_duplicate_paths.extend(ordered[1:])
    if removed_duplicate_paths:
        removed = set(removed_duplicate_paths)
        for relative in removed_duplicate_paths:
            (output_root / relative).unlink()
        manifest = [row for row in manifest if row["crop_path"] not in removed]
        print(
            f"DEDUP: membuang {len(removed_duplicate_paths)} crop identik "
            f"dari {len(original_duplicate_crop_groups)} grup",
            flush=True,
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    split_counts = {
        split: dict(
            sorted(
                Counter(
                    row["canonical_class"]
                    for row in manifest
                    if row["generated_split"] == split
                ).items()
            )
        )
        for split in SPLITS
    }
    missing_by_split = {
        split: sorted(set(CANONICAL_CLASSES).difference(counts))
        for split, counts in split_counts.items()
    }
    if any(missing_by_split.values()):
        raise RuntimeError(f"Kelas kosong pada split hasil: {missing_by_split}")
    split_totals = {
        split: sum(counts.values()) for split, counts in split_counts.items()
    }
    split_ratios = {
        split: total / len(manifest) for split, total in split_totals.items()
    }
    dataset_split_counts = {
        dataset: {
            split: sum(
                row["dataset"] == dataset and row["generated_split"] == split
                for row in manifest
            )
            for split in SPLITS
        }
        for dataset in datasets
    }
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        group_splits[row["group_id"]].add(row["generated_split"])
    generated_overlap = {
        group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1
    }
    if generated_overlap:
        raise RuntimeError("Bug internal: satu foto sumber tersebar ke beberapa split.")
    contact_sheets = _contact_sheets(output_root, manifest)
    audit = {
        "status": "complete",
        "protocol": "SNI_instance_crop_v1",
        "seed": seed,
        "classes": list(CANONICAL_CLASSES),
        "num_classes": len(CANONICAL_CLASSES),
        "sni_defect_weights": SNI_DEFECT_WEIGHTS,
        "source_audits": source_audits,
        "identity_audit": identity_audit,
        "input_images": len(images),
        "output_crops": len(manifest),
        "split_strategy": "deterministic_grouped_multilabel_70_15_15",
        "grouping_rule": (
            "Semua gambar dengan dataset+nama sumber Roboflow yang sama atau SHA256 "
            "piksel identik ditempatkan pada split yang sama."
        ),
        "generated_cross_split_identity_count": 0,
        "split_counts": split_counts,
        "split_totals": split_totals,
        "split_ratios": split_ratios,
        "missing_classes_by_split": missing_by_split,
        "dataset_split_counts": dataset_split_counts,
        "exact_duplicate_crop_groups_before_dedup": len(
            original_duplicate_crop_groups
        ),
        "removed_exact_duplicate_crops": len(removed_duplicate_paths),
        "conflicting_exact_duplicate_crop_groups": 0,
        "cross_split_exact_duplicate_crop_groups_after_dedup": 0,
        "exact_duplicate_crop_examples_before_dedup": (
            original_duplicate_crop_groups[:100]
        ),
        "crop_protocol": {
            "shape": "square centered on COCO bbox",
            "margin_fraction": margin_fraction,
            "padding": "mean RGB of source image when crop crosses an edge",
            "resize": "none; model transform performs resize",
            "jpeg_quality": jpeg_quality,
        },
        "contact_sheets": contact_sheets,
        "test_locked": True,
        "claim_note": (
            "Dataset klasifikasi per-instance; evaluasi end-to-end tetap memerlukan "
            "detector. Split test tidak boleh digunakan untuk pemilihan model."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\n=== SNI INSTANCE-CROP DATASET ===")
    print(f"Images input : {len(images):,}")
    print(f"Crops output : {len(manifest):,}")
    print(f"Classes      : {len(CANONICAL_CLASSES)}")
    print(f"Archive identity leaks: {identity_audit['archive_cross_split_identity_count']}")
    print("Generated identity leaks: 0")
    print("Split totals:", split_totals)
    print("Split ratios:", {split: f"{ratio:.2%}" for split, ratio in split_ratios.items()})
    print("Dataset totals:", dataset_split_counts)
    print("SAVED:", audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare grouped 21-class SNI instance crops from COCO datasets"
    )
    parser.add_argument("--adrian-root", type=Path, required=True)
    parser.add_argument("--faruq-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    prepare_sni_instance_crops(
        args.adrian_root,
        args.faruq_root,
        args.output_root,
        seed=args.seed,
        margin_fraction=args.margin,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
