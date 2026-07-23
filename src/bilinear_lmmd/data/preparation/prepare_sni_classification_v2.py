from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from bilinear_lmmd.data.sni_ontology import SNI_CLASSES


SPLITS = ("train", "val", "test")
DATASETS = ("adrian_detection", "faruq_segmentation")
SIZE_LEVELS = ("besar", "sedang", "kecil")

BEAN_CLASSES = SNI_CLASSES[:12]
VISUAL_CLASSES = (
    *BEAN_CLASSES,
    "kulit_kopi",
    "kulit_tanduk",
    "benda_asing",
)
ATTRIBUTE_NAMES = (
    "black",
    "broken",
    "one_hole",
    "multiple_holes",
    "spotted",
    "brown",
    "immature",
    "parchment_covered",
    "cherry",
    "normal",
)

REQUIRED_COLUMNS = {
    "dataset",
    "generated_split",
    "group_id",
    "source_identity",
    "original_class",
    "canonical_class",
    "bbox_width",
    "bbox_height",
    "crop_sha256",
    "crop_path",
}


def visual_label(class_name: str) -> str:
    if class_name in BEAN_CLASSES:
        return class_name
    if class_name.startswith("kulit_kopi_ukuran_"):
        return "kulit_kopi"
    if class_name.startswith("kulit_tanduk_ukuran_"):
        return "kulit_tanduk"
    if class_name.startswith("tanah_batu_ranting_"):
        return "benda_asing"
    raise ValueError(f"Kelas SNI tidak dikenal: {class_name}")


def family_label(class_name: str) -> str:
    if class_name in BEAN_CLASSES:
        return "coffee_bean"
    if class_name.startswith("kulit_kopi_ukuran_"):
        return "coffee_skin"
    if class_name.startswith("kulit_tanduk_ukuran_"):
        return "parchment_skin"
    if class_name.startswith("tanah_batu_ranting_"):
        return "foreign_matter"
    raise ValueError(f"Kelas SNI tidak dikenal: {class_name}")


def size_label(class_name: str) -> str:
    for size in SIZE_LEVELS:
        if class_name.endswith(f"_{size}"):
            return size
    return "not_applicable"


def partial_attributes(class_name: str) -> dict[str, int]:
    """Return positive-only defect attributes with -1 meaning unobserved.

    The source datasets provide one category per annotation. Except for normal
    beans and the explicit compound ``biji_hitam_pecah``, absence of another
    defect is not known and must not be converted into a negative target.
    """

    values = {name: -1 for name in ATTRIBUTE_NAMES}
    if class_name not in BEAN_CLASSES:
        return values
    values["normal"] = 0
    if class_name == "biji_normal":
        values = {name: 0 for name in ATTRIBUTE_NAMES}
        values["normal"] = 1
        return values

    positive = {
        "biji_berkulit_tanduk": ("parchment_covered",),
        "biji_berlubang_lebih_satu": ("multiple_holes",),
        "biji_berlubang_satu": ("one_hole",),
        "biji_bertutul_tutul": ("spotted",),
        "biji_coklat": ("brown",),
        "biji_hitam": ("black",),
        "biji_hitam_pecah": ("black", "broken"),
        "biji_hitam_sebagian": ("black",),
        "biji_muda": ("immature",),
        "biji_pecah": ("broken",),
        "kopi_gelondong": ("cherry",),
    }
    for attribute in positive[class_name]:
        values[attribute] = 1
    return values


def _read_manifest(input_root: Path) -> tuple[list[dict[str, str]], dict]:
    audit_path = input_root / "audit.json"
    manifest_path = input_root / "manifest.csv"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Dataset v1 belum lengkap: {audit_path} dan {manifest_path} wajib ada."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("Audit SNI instance-crop v1 belum berstatus complete.")
    if tuple(audit.get("classes", ())) != SNI_CLASSES:
        raise ValueError("Urutan kelas audit v1 tidak cocok dengan ontologi SNI.")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest SNI instance-crop kosong.")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Kolom manifest v1 kurang: {sorted(missing)}")
    if int(audit["output_crops"]) != len(rows):
        raise ValueError(
            f"Jumlah manifest ({len(rows)}) tidak cocok audit ({audit['output_crops']})."
        )
    return rows, audit


def _validate_rows(input_root: Path, rows: list[dict[str, str]]) -> dict:
    missing_files = []
    unsafe_paths = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_labels: dict[str, set[str]] = defaultdict(set)
    observed_classes = set()
    observed_datasets = set()

    resolved_root = input_root.resolve()
    for row in rows:
        split = row["generated_split"]
        if split not in SPLITS:
            raise ValueError(f"Split tidak dikenal: {split}")
        class_name = row["canonical_class"]
        if class_name not in SNI_CLASSES:
            raise ValueError(f"Kelas canonical tidak dikenal: {class_name}")
        dataset = row["dataset"]
        if dataset not in DATASETS:
            raise ValueError(f"Dataset asal tidak dikenal: {dataset}")
        observed_classes.add(class_name)
        observed_datasets.add(dataset)
        group_splits[row["group_id"]].add(split)
        hash_splits[row["crop_sha256"]].add(split)
        hash_labels[row["crop_sha256"]].add(class_name)

        relative = Path(row["crop_path"])
        candidate = (input_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            unsafe_paths.append(row["crop_path"])
            continue
        if not candidate.is_file():
            missing_files.append(row["crop_path"])

    missing_classes = sorted(set(SNI_CLASSES).difference(observed_classes))
    missing_datasets = sorted(set(DATASETS).difference(observed_datasets))
    cross_split_groups = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    cross_split_hashes = {
        digest: sorted(splits)
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }
    conflicting_hashes = {
        digest: sorted(labels)
        for digest, labels in hash_labels.items()
        if len(labels) > 1
    }
    errors = {
        "missing_classes": missing_classes,
        "missing_datasets": missing_datasets,
        "unsafe_paths": unsafe_paths[:20],
        "missing_crop_files": missing_files[:20],
        "cross_split_groups": dict(list(cross_split_groups.items())[:20]),
        "cross_split_crop_hashes": dict(list(cross_split_hashes.items())[:20]),
        "conflicting_crop_hashes": dict(list(conflicting_hashes.items())[:20]),
    }
    if any(errors.values()):
        raise ValueError(f"Integritas dataset v1 gagal: {errors}")
    return {
        "crop_files_checked": len(rows),
        "missing_crop_files": 0,
        "unsafe_paths": 0,
        "cross_split_groups": 0,
        "cross_split_crop_hashes": 0,
        "conflicting_crop_hashes": 0,
    }


def _normalized_weights(
    counts: Counter[str],
    *,
    method: str,
    beta: float,
) -> dict[str, float]:
    if method == "inverse_sqrt":
        raw = {name: 1.0 / math.sqrt(count) for name, count in counts.items()}
    elif method == "effective_number":
        raw = {
            name: (1.0 - beta) / (1.0 - beta**count)
            for name, count in counts.items()
        }
    else:
        raise ValueError(f"Metode weight tidak dikenal: {method}")
    total_samples = sum(counts.values())
    weighted_mean = (
        sum(counts[name] * raw[name] for name in counts) / total_samples
    )
    return {name: value / weighted_mean for name, value in raw.items()}


def _count_by(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _group_count_by(rows: list[dict], field: str) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row[field])].add(str(row["group_id"]))
    return {name: len(groups[name]) for name in sorted(groups)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Tidak boleh menulis manifest kosong: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _external_protocols(rows: list[dict]) -> tuple[dict[str, dict], set[str]]:
    domains_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        domains_by_group[str(row["group_id"])].add(str(row["dataset"]))
    shared_groups = {
        group for group, domains in domains_by_group.items() if len(domains) > 1
    }
    protocols = {}
    for source, target in (
        ("adrian_detection", "faruq_segmentation"),
        ("faruq_segmentation", "adrian_detection"),
    ):
        source_train = [
            row
            for row in rows
            if row["dataset"] == source
            and row["generated_split"] == "train"
            and row["group_id"] not in shared_groups
        ]
        source_val = [
            row
            for row in rows
            if row["dataset"] == source
            and row["generated_split"] == "val"
            and row["group_id"] not in shared_groups
        ]
        external_test = [
            row
            for row in rows
            if row["dataset"] == target and row["group_id"] not in shared_groups
        ]
        coverage = {
            "train": sorted(set(VISUAL_CLASSES).difference(_count_by(source_train, "visual_label"))),
            "val": sorted(set(VISUAL_CLASSES).difference(_count_by(source_val, "visual_label"))),
            "external_test": sorted(
                set(VISUAL_CLASSES).difference(_count_by(external_test, "visual_label"))
            ),
        }
        name = f"{source}_to_{target}"
        protocols[name] = {
            "source": source,
            "target": target,
            "train": source_train,
            "val": source_val,
            "external_test": external_test,
            "missing_visual_classes": coverage,
            "ready": not any(coverage.values()),
        }
    return protocols, shared_groups


def prepare_sni_classification_v2(
    input_root: Path,
    output_root: Path,
    *,
    beta: float = 0.9999,
    min_eval_samples_per_class: int = 50,
    min_eval_groups_per_class: int = 20,
) -> dict:
    if not 0.0 < beta < 1.0:
        raise ValueError("Beta effective-number harus di antara nol dan satu.")
    if min_eval_samples_per_class <= 0 or min_eval_groups_per_class <= 0:
        raise ValueError("Ambang evaluasi harus lebih besar dari nol.")

    source_rows, source_audit = _read_manifest(input_root)
    integrity = _validate_rows(input_root, source_rows)
    enriched = []
    for row in source_rows:
        class_name = row["canonical_class"]
        width = float(row["bbox_width"])
        height = float(row["bbox_height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"BBox non-positif pada {row['crop_path']}")
        attributes = partial_attributes(class_name)
        enriched.append(
            {
                "crop_path": row["crop_path"],
                "generated_split": row["generated_split"],
                "dataset": row["dataset"],
                "group_id": row["group_id"],
                "source_identity": row["source_identity"],
                "flat_label_v1": class_name,
                "visual_label": visual_label(class_name),
                "family_label": family_label(class_name),
                "size_label_metadata": size_label(class_name),
                "size_visual_supervision_allowed": 0,
                "bbox_width_px": width,
                "bbox_height_px": height,
                "bbox_aspect_ratio": max(width, height) / min(width, height),
                "crop_sha256": row["crop_sha256"],
                **{f"attr_{name}": value for name, value in attributes.items()},
            }
        )

    train_rows = [row for row in enriched if row["generated_split"] == "train"]
    train_counts = Counter(row["visual_label"] for row in train_rows)
    missing_train = sorted(set(VISUAL_CLASSES).difference(train_counts))
    if missing_train:
        raise ValueError(f"Visual class kosong pada train: {missing_train}")
    inverse_sqrt = _normalized_weights(
        train_counts, method="inverse_sqrt", beta=beta
    )
    effective_number = _normalized_weights(
        train_counts, method="effective_number", beta=beta
    )
    for row in enriched:
        if row["generated_split"] == "train":
            row["train_weight_inverse_sqrt"] = inverse_sqrt[row["visual_label"]]
            row["train_weight_effective_number"] = effective_number[
                row["visual_label"]
            ]
        else:
            row["train_weight_inverse_sqrt"] = ""
            row["train_weight_effective_number"] = ""

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "manifests" / "all.csv", enriched)
    split_rows = {}
    for split in SPLITS:
        split_rows[split] = [
            row for row in enriched if row["generated_split"] == split
        ]
        _write_csv(output_root / "manifests" / f"{split}.csv", split_rows[split])
    _write_csv(output_root / "manifests" / "train_weighted.csv", train_rows)

    protocols, shared_domain_groups = _external_protocols(enriched)
    protocol_audit = {}
    for name, protocol in protocols.items():
        protocol_root = output_root / "cross_domain" / name
        for split in ("train", "val", "external_test"):
            _write_csv(protocol_root / f"{split}.csv", protocol[split])
        protocol_audit[name] = {
            "source": protocol["source"],
            "target": protocol["target"],
            "counts": {
                split: len(protocol[split])
                for split in ("train", "val", "external_test")
            },
            "missing_visual_classes": protocol["missing_visual_classes"],
            "ready": protocol["ready"],
        }

    split_statistics = {}
    weak_classes = {}
    for split, rows in split_rows.items():
        counts = _count_by(rows, "visual_label")
        group_counts = _group_count_by(rows, "visual_label")
        split_statistics[split] = {
            "samples": len(rows),
            "visual_class_counts": counts,
            "visual_class_group_counts": group_counts,
            "flat_class_counts": _count_by(rows, "flat_label_v1"),
            "family_counts": _count_by(rows, "family_label"),
            "dataset_counts": _count_by(rows, "dataset"),
            "visual_imbalance_ratio": max(counts.values()) / min(counts.values()),
        }
        if split in {"val", "test"}:
            weak_classes[split] = {
                class_name: {
                    "samples": counts.get(class_name, 0),
                    "groups": group_counts.get(class_name, 0),
                    "enough_samples": counts.get(class_name, 0)
                    >= min_eval_samples_per_class,
                    "enough_groups": group_counts.get(class_name, 0)
                    >= min_eval_groups_per_class,
                }
                for class_name in VISUAL_CLASSES
                if counts.get(class_name, 0) < min_eval_samples_per_class
                or group_counts.get(class_name, 0) < min_eval_groups_per_class
            }

    audit = {
        "status": "complete",
        "protocol": "SNI_classification_manifest_v2",
        "source_protocol": source_audit["protocol"],
        "input_root": str(input_root),
        "input_crops": len(enriched),
        "integrity": integrity,
        "label_design": {
            "flat_v1_classes": list(SNI_CLASSES),
            "visual_v2_classes": list(VISUAL_CLASSES),
            "visual_v2_num_classes": len(VISUAL_CLASSES),
            "families": [
                "coffee_bean",
                "coffee_skin",
                "parchment_skin",
                "foreign_matter",
            ],
            "size_policy": (
                "large/medium/small retained as metadata but excluded from the "
                "crop-only visual target because physical scale is not calibrated"
            ),
            "attribute_policy": (
                "partial positive attributes; -1 means unobserved, not negative"
            ),
            "attribute_names": list(ATTRIBUTE_NAMES),
        },
        "split_statistics": split_statistics,
        "training_balance": {
            "files_duplicated_or_deleted": False,
            "recommended_default": "inverse_sqrt sample weights",
            "inverse_sqrt_weights": dict(sorted(inverse_sqrt.items())),
            "effective_number_beta": beta,
            "effective_number_weights": dict(sorted(effective_number.items())),
            "validation_and_test_weights": "none; natural grouped distributions retained",
        },
        "statistical_readiness": {
            "minimum_eval_samples_per_class": min_eval_samples_per_class,
            "minimum_eval_groups_per_class": min_eval_groups_per_class,
            "weak_classes": weak_classes,
            "strong_per_class_claim_ready": not any(weak_classes.values()),
        },
        "cross_domain": {
            "shared_groups_excluded": len(shared_domain_groups),
            "protocols": protocol_audit,
        },
        "test_locked": True,
        "training_authorized": False,
        "next_required_action": (
            "Review audit.json, weak classes, ontology.json, and random visual "
            "samples before freezing a training protocol."
        ),
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    ontology = {
        "schema_version": 2,
        "visual_classes": list(VISUAL_CLASSES),
        "flat_to_visual": {
            class_name: visual_label(class_name) for class_name in SNI_CLASSES
        },
        "flat_to_family": {
            class_name: family_label(class_name) for class_name in SNI_CLASSES
        },
        "flat_to_size_metadata": {
            class_name: size_label(class_name) for class_name in SNI_CLASSES
        },
        "partial_attribute_values": {
            class_name: partial_attributes(class_name)
            for class_name in SNI_CLASSES
        },
        "unknown_attribute_value": -1,
        "size_visual_supervision_allowed": False,
    }
    (output_root / "ontology.json").write_text(
        json.dumps(ontology, indent=2), encoding="utf-8"
    )
    print("\n=== SNI CLASSIFICATION MANIFEST V2 ===")
    print(f"Input crops       : {len(enriched):,}")
    print(f"Visual classes    : {len(VISUAL_CLASSES)}")
    print("Split totals      :", {split: len(rows) for split, rows in split_rows.items()})
    print("Weak val classes  :", len(weak_classes["val"]))
    print("Weak test classes :", len(weak_classes["test"]))
    print(
        "Strong claim ready:",
        audit["statistical_readiness"]["strong_per_class_claim_ready"],
    )
    print("Training authorized: False")
    print("SAVED:", output_root / "audit.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe hierarchical SNI classification v2 manifest."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--min-eval-samples-per-class", type=int, default=50)
    parser.add_argument("--min-eval-groups-per-class", type=int, default=20)
    args = parser.parse_args()
    prepare_sni_classification_v2(
        args.input_root,
        args.output_root,
        beta=args.beta,
        min_eval_samples_per_class=args.min_eval_samples_per_class,
        min_eval_groups_per_class=args.min_eval_groups_per_class,
    )


if __name__ == "__main__":
    main()
