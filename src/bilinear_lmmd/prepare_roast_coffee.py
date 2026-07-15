from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("Dark", "Green", "Light", "Medium")
CLASS_ALIASES = {name.lower(): name for name in CLASSES}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    split: str


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def discover_roast_samples(raw_root: Path) -> tuple[list[Sample], list[str]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Root dataset roast tidak ditemukan: {raw_root}")
    samples: list[Sample] = []
    ignored: list[str] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(raw_root)
        class_matches = {
            CLASS_ALIASES[_normalized(part)]
            for part in relative.parts[:-1]
            if _normalized(part) in CLASS_ALIASES
        }
        split_matches = {
            SPLIT_ALIASES[_normalized(part)]
            for part in relative.parts[:-1]
            if _normalized(part) in SPLIT_ALIASES
        }
        if len(class_matches) != 1 or len(split_matches) != 1:
            ignored.append(relative.as_posix())
            continue
        samples.append(
            Sample(path, next(iter(class_matches)), next(iter(split_matches)))
        )
    if not samples:
        raise FileNotFoundError(
            "Tidak menemukan struktur train/test dengan kelas Dark/Green/Light/Medium "
            f"di {raw_root}."
        )
    missing = sorted(set(CLASSES).difference(sample.class_name for sample in samples))
    if missing:
        raise ValueError(f"Kelas roast belum lengkap: {missing}")
    return samples, ignored


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_and_deduplicate(
    samples: list[Sample],
) -> tuple[list[Sample], list[dict], list[dict], list[dict]]:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for index, sample in enumerate(samples, 1):
        if index == 1 or index % 250 == 0 or index == len(samples):
            print(f"hash audit: {index}/{len(samples)}", flush=True)
        by_hash[_sha256(sample.path)].append(sample)

    kept: list[Sample] = []
    same_split_duplicates: list[dict] = []
    cross_split_duplicates: list[dict] = []
    class_conflicts: list[dict] = []
    for digest, group in sorted(by_hash.items()):
        classes = sorted({sample.class_name for sample in group})
        splits = sorted({sample.split for sample in group})
        paths = sorted(str(sample.path) for sample in group)
        if len(classes) > 1:
            class_conflicts.append(
                {"sha256": digest, "classes": classes, "splits": splits, "paths": paths}
            )
            continue
        canonical = sorted(group, key=lambda sample: str(sample.path))[0]
        kept.append(canonical)
        if len(group) > 1:
            record = {
                "sha256": digest,
                "class": canonical.class_name,
                "splits": splits,
                "kept": str(canonical.path),
                "removed": [path for path in paths if path != str(canonical.path)],
            }
            if len(splits) > 1:
                cross_split_duplicates.append(record)
            else:
                same_split_duplicates.append(record)
    return kept, same_split_duplicates, cross_split_duplicates, class_conflicts


def _validation_assignments(
    samples: list[Sample], seed: int, validation_ratio: float
) -> dict[Path, str]:
    present = {sample.split for sample in samples}
    if "train" not in present or "test" not in present:
        raise ValueError(f"Dataset harus memiliki train dan test, ditemukan {sorted(present)}")
    if "val" in present:
        return {sample.path: sample.split for sample in samples}

    assignments = {sample.path: sample.split for sample in samples}
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.split == "train":
            by_class[sample.class_name].append(sample)
    for class_name, class_samples in sorted(by_class.items()):
        random.Random(f"{seed}:{class_name}").shuffle(class_samples)
        val_count = max(1, round(len(class_samples) * validation_ratio))
        for sample in class_samples[:val_count]:
            assignments[sample.path] = "val"
    return assignments


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_roast_coffee(
    raw_root: Path,
    output_root: Path,
    seed: int = 42,
    validation_ratio: float = 0.2,
) -> dict:
    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: dataset roast sudah siap: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi audit belum lengkap. "
            "Hapus/pindahkan folder tersebut secara manual."
        )
    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio harus di antara 0 dan 0,5.")

    discovered, ignored = discover_roast_samples(raw_root)
    clean, same_split, cross_split, conflicts = _audit_and_deduplicate(discovered)
    assignments = _validation_assignments(clean, seed, validation_ratio)
    had_validation = "val" in {sample.split for sample in clean}

    output_root.mkdir(parents=True)
    for sample in clean:
        relative = sample.path.relative_to(raw_root).as_posix()
        prefix = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
        destination = (
            output_root
            / "source"
            / assignments[sample.path]
            / sample.class_name
            / f"{prefix}_{sample.path.name}"
        )
        _link_or_copy(sample.path, destination)

    split_counts = {
        split: dict(
            sorted(
                Counter(
                    sample.class_name
                    for sample in clean
                    if assignments[sample.path] == split
                ).items()
            )
        )
        for split in ("train", "val", "test")
    }
    audit = {
        "status": "complete",
        "raw_root": str(raw_root.resolve()),
        "input_images": len(discovered),
        "clean_images": len(clean),
        "class_counts": dict(sorted(Counter(s.class_name for s in clean).items())),
        "split_counts": split_counts,
        "split_strategy": (
            "preserved_archive_train_val_test"
            if had_validation
            else f"preserved_test_train_to_val_{validation_ratio:.2f}"
        ),
        "seed": seed,
        "same_split_exact_duplicates": same_split,
        "cross_split_exact_duplicates": cross_split,
        "cross_class_exact_conflicts": conflicts,
        "ignored_images": ignored,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n=== AUDIT COFFEE ROAST ===")
    print(f"Input                   : {len(discovered)}")
    print(f"Bersih                  : {len(clean)}")
    print(f"Strategi split          : {audit['split_strategy']}")
    print(f"Duplicate lintas split : {len(cross_split)}")
    print(f"Konflik label           : {len(conflicts)}")
    for split, counts in split_counts.items():
        print(f"{split:5s}: {sum(counts.values()):5d} | {counts}")
    print(f"SAVED: {audit_path}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Coffee Bean Dataset Resized dan siapkan ImageFolder"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    args = parser.parse_args()
    prepare_roast_coffee(
        args.raw_root, args.output_root, args.seed, args.validation_ratio
    )


if __name__ == "__main__":
    main()
