from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _identity(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _collect(source_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split source tidak ditemukan: {split_root}")
        for path in sorted(split_root.glob("*/*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            identity = _identity(path)
            if identity in images:
                raise ValueError(f"Identity filename duplikat: {identity}")
            images[identity] = path
    return images


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_exact_duplicates(
    images: dict[str, Path],
) -> tuple[dict[str, Path], list[dict], list[dict]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for identity, path in sorted(images.items()):
        by_hash[_sha256(path)].append(identity)

    kept: dict[str, Path] = {}
    same_class_duplicates: list[dict] = []
    label_conflicts: list[dict] = []
    for digest, identities in sorted(by_hash.items()):
        classes = {identity.split("/", 1)[0] for identity in identities}
        if len(identities) == 1:
            identity = identities[0]
            kept[identity] = images[identity]
        elif len(classes) == 1:
            canonical = sorted(identities)[0]
            kept[canonical] = images[canonical]
            same_class_duplicates.append(
                {
                    "sha256": digest,
                    "class": next(iter(classes)),
                    "kept": canonical,
                    "removed": sorted(
                        identity for identity in identities if identity != canonical
                    ),
                }
            )
        else:
            label_conflicts.append(
                {
                    "sha256": digest,
                    "classes": sorted(classes),
                    "quarantined": sorted(identities),
                }
            )
    return kept, same_class_duplicates, label_conflicts


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_folds(
    clean_source: Path,
    output_root: Path,
    folds: int,
    seed: int,
    validation_ratio: float,
) -> dict:
    if folds < 3:
        raise ValueError("Jumlah fold minimal 3.")
    if not 0 < validation_ratio < 1 / folds:
        raise ValueError(
            "validation_ratio harus positif dan lebih kecil dari test fold."
        )
    by_class = {
        class_dir.name: sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for class_dir in sorted(path for path in clean_source.iterdir() if path.is_dir())
    }
    if not by_class or any(len(paths) < folds for paths in by_class.values()):
        raise ValueError("Setiap kelas harus memiliki minimal satu gambar per fold.")

    assignments = {
        f"fold_{fold}": {"train": [], "val": [], "test": []}
        for fold in range(1, folds + 1)
    }
    test_union: set[str] = set()
    for class_name, original_images in sorted(by_class.items()):
        images = list(original_images)
        random.Random(f"{seed}:{class_name}:outer").shuffle(images)
        test_bins = [images[index::folds] for index in range(folds)]
        for fold_index in range(folds):
            fold_name = f"fold_{fold_index + 1}"
            test_images = test_bins[fold_index]
            test_set = set(test_images)
            remaining = [path for path in images if path not in test_set]
            random.Random(f"{seed}:{class_name}:{fold_index}:validation").shuffle(
                remaining
            )
            validation_count = max(1, round(len(images) * validation_ratio))
            split_images = {
                "train": remaining[validation_count:],
                "val": remaining[:validation_count],
                "test": test_images,
            }
            for split, paths in split_images.items():
                for path in paths:
                    identity = _identity(path)
                    destination = (
                        output_root
                        / fold_name
                        / "source"
                        / split
                        / class_name
                        / path.name
                    )
                    _link_or_copy(path, destination)
                    assignments[fold_name][split].append(identity)
                    if split == "test":
                        if identity in test_union:
                            raise RuntimeError(
                                f"Identity test muncul lebih dari sekali: {identity}"
                            )
                        test_union.add(identity)

    total = sum(len(paths) for paths in by_class.values())
    if len(test_union) != total:
        raise RuntimeError(
            f"Outer test hanya mencakup {len(test_union)} dari {total} gambar."
        )
    metadata = {
        "folds": folds,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "total_originals": total,
        "class_counts": {name: len(paths) for name, paths in by_class.items()},
        "fold_counts": {
            fold: {split: len(items) for split, items in values.items()}
            for fold, values in assignments.items()
        },
        "assignments": assignments,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def prepare_clean_grouped_folds(
    source_root: Path,
    output_root: Path,
    expected_count: int = 979,
    folds: int = 5,
    seed: int = 42,
    validation_ratio: float = 0.1,
) -> dict:
    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: dataset bersih sudah lengkap: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi belum lengkap. Pindahkan atau hapus "
            "folder tersebut secara manual, lalu jalankan ulang."
        )

    images = _collect(source_root)
    if expected_count and len(images) != expected_count:
        raise ValueError(
            f"Diharapkan {expected_count} gambar, ditemukan {len(images)}."
        )
    kept, same_class_duplicates, label_conflicts = analyze_exact_duplicates(images)
    output_root.mkdir(parents=True)
    clean_source = output_root / "clean_source"
    for identity, path in kept.items():
        _link_or_copy(path, clean_source / identity)
    quarantine_root = output_root / "quarantine_label_conflicts"
    for group in label_conflicts:
        group_root = quarantine_root / group["sha256"][:12]
        for identity in group["quarantined"]:
            _link_or_copy(images[identity], group_root / identity)

    folds_metadata = _build_folds(
        clean_source,
        output_root / "folds",
        folds,
        seed,
        validation_ratio,
    )
    original_counts = Counter(identity.split("/", 1)[0] for identity in images)
    clean_counts = Counter(identity.split("/", 1)[0] for identity in kept)
    audit = {
        "status": "complete",
        "source_root": str(source_root.resolve()),
        "input_count": len(images),
        "clean_count": len(kept),
        "removed_same_class_count": sum(
            len(group["removed"]) for group in same_class_duplicates
        ),
        "quarantined_conflict_count": sum(
            len(group["quarantined"]) for group in label_conflicts
        ),
        "original_class_counts": dict(sorted(original_counts.items())),
        "clean_class_counts": dict(sorted(clean_counts.items())),
        "same_class_duplicate_groups": same_class_duplicates,
        "label_conflict_groups": label_conflicts,
        "fold_counts": folds_metadata["fold_counts"],
        "policy": (
            "Keep one lexicographic canonical file for exact same-class hashes; "
            "quarantine every file in an exact cross-class hash group."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (output_root / "class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "original", "clean", "removed"))
        for name in sorted(original_counts):
            writer.writerow(
                (name, original_counts[name], clean_counts[name], original_counts[name] - clean_counts[name])
            )

    print("\n=== CLEAN DATASET ===")
    print(f"Input                    : {len(images)}")
    print(f"Duplikat label sama      : {audit['removed_same_class_count']}")
    print(f"Konflik label dikarantina: {audit['quarantined_conflict_count']}")
    print(f"Total bersih             : {len(kept)}")
    print(f"Clean source             : {clean_source}")
    print(f"Grouped folds            : {output_root / 'folds'}")
    print(f"Audit                    : {audit_path}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate exact coffee images dan buat grouped folds bersih"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=979)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    prepare_clean_grouped_folds(
        source_root=args.source_root,
        output_root=args.output_root,
        expected_count=args.expected_count,
        folds=args.folds,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )


if __name__ == "__main__":
    main()
