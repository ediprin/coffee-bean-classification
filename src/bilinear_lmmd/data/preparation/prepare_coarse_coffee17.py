from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


COARSE_GROUPS = {
    "Black": ("Full Black", "Partial Black"),
    "Sour": ("Full Sour", "Partial Sour"),
    "Insect Damage": ("Severe Insect Damage", "Slight Insect Damage"),
    "Physical Form": ("Broken", "Cut", "Shell"),
    "Covering Residue": ("Husk", "Parchment"),
    "Developmental": ("Immature", "Withered"),
    "Processing Density": ("Dry Cherry", "Floater"),
    "Fade": ("Fade",),
    "Fungus Damage": ("Fungus Damage",),
}

FINE_TO_COARSE = {
    fine: coarse for coarse, members in COARSE_GROUPS.items() for fine in members
}
EXPECTED_FINE_CLASSES = tuple(sorted(FINE_TO_COARSE))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_coarse_coffee17(fine_root: Path, output_root: Path) -> dict:
    """Remap labels while preserving every split and image identity."""

    audit_path = output_root / "coarse_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: coarse Coffee-17 sudah lengkap: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi audit belum lengkap. "
            "Pindahkan folder tersebut dan jalankan ulang."
        )

    source_root = fine_root / "source"
    observed_classes: set[str] = set()
    split_fine_counts: dict[str, Counter] = {}
    split_coarse_counts: dict[str, Counter] = {}
    assignments: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split fine tidak ditemukan: {split_root}")
        fine_counts: Counter = Counter()
        coarse_counts: Counter = Counter()
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            fine_class = class_dir.name
            observed_classes.add(fine_class)
            if fine_class not in FINE_TO_COARSE:
                raise ValueError(f"Kelas fine belum memiliki parent: {fine_class}")
            coarse_class = FINE_TO_COARSE[fine_class]
            for image in sorted(class_dir.iterdir()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                # Prefix prevents filename collisions when fine folders merge.
                destination = (
                    output_root
                    / "source"
                    / split
                    / coarse_class
                    / f"{fine_class}__{image.name}"
                )
                _link_or_copy(image, destination)
                fine_counts[fine_class] += 1
                coarse_counts[coarse_class] += 1
                assignments.append(
                    {
                        "split": split,
                        "fine_class": fine_class,
                        "coarse_class": coarse_class,
                        "source": str(image),
                        "destination": str(destination),
                    }
                )
        split_fine_counts[split] = fine_counts
        split_coarse_counts[split] = coarse_counts

    missing = sorted(set(EXPECTED_FINE_CLASSES).difference(observed_classes))
    extra = sorted(observed_classes.difference(EXPECTED_FINE_CLASSES))
    if missing or extra:
        raise ValueError(f"Taxonomy Coffee-17 tidak cocok; missing={missing}, extra={extra}")

    audit = {
        "status": "complete",
        "fine_root": str(fine_root),
        "output_root": str(output_root),
        "mapping_kind": "operational_visual_mechanism_grouping_not_official_standard",
        "coarse_groups": {name: list(members) for name, members in COARSE_GROUPS.items()},
        "fine_classes": list(EXPECTED_FINE_CLASSES),
        "coarse_classes": sorted(COARSE_GROUPS),
        "split_fine_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in split_fine_counts.items()
        },
        "split_coarse_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in split_coarse_counts.items()
        },
        "split_totals": {
            split: sum(counts.values()) for split, counts in split_coarse_counts.items()
        },
        "assignments": assignments,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\n=== COARSE COFFEE-17 ===")
    print(f"Fine classes  : {len(EXPECTED_FINE_CLASSES)}")
    print(f"Coarse classes: {len(COARSE_GROUPS)}")
    print(f"Split totals  : {audit['split_totals']}")
    print(f"SAVED         : {audit_path}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap Coffee-17 menjadi 9 parent coarse")
    parser.add_argument("--fine-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    prepare_coarse_coffee17(args.fine_root, args.output_root)


if __name__ == "__main__":
    main()
