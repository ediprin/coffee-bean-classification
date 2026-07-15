from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

from .prepare_coffee17 import EXPECTED_COUNTS, discover_directory_samples


PAPER_ANGLES = (0, 45, 90, 135, 180, 225, 270)


def _variant_records(
    samples: dict[str, list[Path]],
    angles: tuple[int, ...],
) -> list[tuple[str, Path, int]]:
    return [
        (class_name, path, angle)
        for class_name, paths in sorted(samples.items())
        for path in sorted(paths)
        for angle in angles
    ]


def _split_records(
    records: list[tuple[str, Path, int]], seed: int
) -> dict[str, list[tuple[str, Path, int]]]:
    train, remainder = train_test_split(
        records, test_size=0.30, random_state=seed, shuffle=True
    )
    validation, test = train_test_split(
        remainder, test_size=1 / 3, random_state=seed, shuffle=True
    )
    return {"train": train, "val": validation, "test": test}


def _write_variant(
    source: Path,
    destination: Path,
    angle: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if angle == 0:
        shutil.copy2(source, destination)
        return
    with Image.open(source) as image:
        rotated = image.convert("RGB").rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )
        rotated.save(destination, quality=95)


def _overlap_audit(
    splits: dict[str, list[tuple[str, Path, int]]]
) -> dict:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, records in splits.items():
        for class_name, path, _ in records:
            locations[f"{class_name}/{path.name}"].add(split)
    patterns = Counter("+".join(sorted(value)) for value in locations.values())
    leaked = {key: value for key, value in locations.items() if len(value) > 1}
    return {
        "unique_originals": len(locations),
        "originals_crossing_splits": len(leaked),
        "crossing_fraction": len(leaked) / max(len(locations), 1),
        "split_membership_patterns": dict(sorted(patterns.items())),
    }


def prepare_paper_protocol(
    raw_root: Path,
    output_root: Path,
    seed: int = 42,
    expected_counts: dict[str, int] | None = None,
    angles: tuple[int, ...] = PAPER_ANGLES,
) -> dict:
    """Reproduce the paper's augment-before-split protocol with an audit."""

    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: paper-protocol dataset sudah lengkap: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi audit belum lengkap. "
            "Pindahkan atau hapus folder tersebut secara manual."
        )
    if len(set(angles)) != len(angles) or 0 not in angles:
        raise ValueError("Daftar sudut harus unik dan memuat sudut 0.")

    expected = expected_counts or EXPECTED_COUNTS
    samples = discover_directory_samples(raw_root, expected_counts=expected)
    records = _variant_records(samples, angles)
    expected_total = sum(expected.values()) * len(angles)
    if len(records) != expected_total:
        raise RuntimeError(
            f"Variant berjumlah {len(records)}, diharapkan {expected_total}."
        )
    splits = _split_records(records, seed)
    output_root.mkdir(parents=True)
    source_root = output_root / "source"
    for split, split_records in splits.items():
        for index, (class_name, source, angle) in enumerate(split_records):
            filename = f"{index:05d}__{source.stem}__rot{angle:03d}.jpg"
            _write_variant(
                source,
                source_root / split / class_name / filename,
                angle,
            )

    split_counts = {name: len(items) for name, items in splits.items()}
    class_counts = {
        split: dict(sorted(Counter(item[0] for item in items).items()))
        for split, items in splits.items()
    }
    overlap = _overlap_audit(splits)
    audit = {
        "status": "complete",
        "claim": "paper-protocol reproduction; leakage-prone, not primary evidence",
        "raw_root": str(raw_root.resolve()),
        "seed": seed,
        "angles": list(angles),
        "original_count": sum(expected.values()),
        "augmented_count": len(records),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "identity_overlap": overlap,
        "operational_split": (
            "Unstratified sklearn 70/20/10 after augmentation; second split "
            "uses test_size=1/3, yielding the paper-aligned test support."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n=== PAPER-PROTOCOL DATASET ===")
    print(f"Original : {audit['original_count']}")
    print(f"Augmented: {audit['augmented_count']}")
    print(f"Split    : {split_counts}")
    print(
        "Leakage  : "
        f"{overlap['originals_crossing_splits']}/"
        f"{overlap['unique_originals']} original melintasi split"
    )
    print("KLAIM    : reproduksi paper, bukan bukti generalisasi utama")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Siapkan reproduksi augment-before-split Arwatchananukul et al."
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_paper_protocol(args.raw_root, args.output_root, args.seed)


if __name__ == "__main__":
    main()
