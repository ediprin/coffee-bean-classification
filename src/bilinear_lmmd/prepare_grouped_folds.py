from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from .prepare_coffee17 import EXPECTED_COUNTS


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect_originals(source_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    identities: set[str] = set()
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split source tidak ditemukan: {split_root}")
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for image in sorted(class_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                identity = f"{class_dir.name}/{image.name}"
                if identity in identities:
                    raise ValueError(f"Identitas gambar duplikat: {identity}")
                identities.add(identity)
                by_class.setdefault(class_dir.name, []).append(image)

    observed = {name: len(images) for name, images in sorted(by_class.items())}
    if observed != EXPECTED_COUNTS:
        raise ValueError(
            f"Jumlah citra asli tidak sesuai paper. Ditemukan {observed}, "
            f"diharapkan {EXPECTED_COUNTS}."
        )
    return by_class


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _output_is_complete(output_root: Path, folds: int) -> bool:
    metadata = output_root / "metadata.json"
    if not metadata.is_file():
        return False
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    expected_total = sum(EXPECTED_COUNTS.values())
    if payload.get("folds") != folds or payload.get("total_originals") != expected_total:
        return False
    for fold_index in range(1, folds + 1):
        fold_root = output_root / f"fold_{fold_index}" / "source"
        total = sum(
            len(list((fold_root / split).glob("*/*")))
            for split in ("train", "val", "test")
        )
        if total != expected_total:
            return False
    return True


def prepare_grouped_folds(
    source_root: Path,
    output_root: Path,
    folds: int = 5,
    seed: int = 42,
    validation_ratio: float = 0.1,
) -> None:
    if folds < 3:
        raise ValueError("Jumlah fold minimal 3.")
    if not 0 < validation_ratio < 1 / folds:
        raise ValueError("validation_ratio harus positif dan lebih kecil dari test fold.")
    if output_root.exists():
        if _output_is_complete(output_root, folds):
            print(f"Grouped folds sudah lengkap: {output_root}")
            return
        raise FileExistsError(
            f"{output_root} sudah ada tetapi tidak lengkap. Hapus/pindahkan manual, lalu ulangi."
        )

    by_class = _collect_originals(source_root)
    expected_total = sum(EXPECTED_COUNTS.values())
    fold_assignments: dict[str, dict[str, list[str]]] = {}
    test_identity_union: set[str] = set()

    for fold_index in range(folds):
        fold_name = f"fold_{fold_index + 1}"
        fold_assignments[fold_name] = {"train": [], "val": [], "test": []}

    for class_name, original_images in sorted(by_class.items()):
        images = list(original_images)
        random.Random(f"{seed}:{class_name}:outer").shuffle(images)
        test_bins = [images[index::folds] for index in range(folds)]

        for fold_index in range(folds):
            fold_name = f"fold_{fold_index + 1}"
            test_images = test_bins[fold_index]
            test_set = set(test_images)
            remaining = [image for image in images if image not in test_set]
            random.Random(f"{seed}:{class_name}:{fold_index}:validation").shuffle(
                remaining
            )
            validation_count = max(1, round(len(images) * validation_ratio))
            val_images = remaining[:validation_count]
            train_images = remaining[validation_count:]
            split_images = {
                "train": train_images,
                "val": val_images,
                "test": test_images,
            }

            for split, selected in split_images.items():
                for image in selected:
                    identity = f"{class_name}/{image.name}"
                    destination = (
                        output_root
                        / fold_name
                        / "source"
                        / split
                        / class_name
                        / image.name
                    )
                    _link_or_copy(image, destination)
                    fold_assignments[fold_name][split].append(identity)
                    if split == "test":
                        if identity in test_identity_union:
                            raise RuntimeError(
                                f"Identitas muncul pada test lebih dari sekali: {identity}"
                            )
                        test_identity_union.add(identity)

    if len(test_identity_union) != expected_total:
        raise RuntimeError(
            f"Out-of-fold test hanya mencakup {len(test_identity_union)} "
            f"dari {expected_total} citra."
        )

    payload = {
        "source_root": str(source_root.resolve()),
        "folds": folds,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "total_originals": expected_total,
        "fold_counts": {
            fold: {split: len(items) for split, items in assignments.items()}
            for fold, assignments in fold_assignments.items()
        },
        "assignments": fold_assignments,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["fold_counts"], indent=2))
    print("PASS: setiap identitas citra menjadi test tepat satu kali.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bentuk stratified grouped folds dari 979 citra asli"
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/coffee/source"))
    parser.add_argument("--output-root", type=Path, default=Path("data/coffee_5fold"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    prepare_grouped_folds(
        args.source_root,
        args.output_root,
        args.folds,
        args.seed,
        args.validation_ratio,
    )


if __name__ == "__main__":
    main()
