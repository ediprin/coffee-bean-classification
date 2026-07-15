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
CLASSES = ("Defect", "Longberry", "Peaberry", "Premium")
CLASS_ALIASES = {
    "defect": "Defect",
    "defective": "Defect",
    "longberry": "Longberry",
    "peaberry": "Peaberry",
    "premium": "Premium",
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
VIEW_SUFFIX = re.compile(
    r"(?:[-_ ](?:front|back|depan|belakang|side[12]|view[12]))$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    split: str | None
    pair_id: str


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _class_from_path(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    for part in reversed(relative.parts[:-1]):
        match = CLASS_ALIASES.get(_normalized(part))
        if match is not None:
            return match
    stem = _normalized(path.stem)
    matches = {
        canonical for alias, canonical in CLASS_ALIASES.items() if stem.startswith(alias)
    }
    if len(matches) > 1:
        raise ValueError(f"Nama file ambigu terhadap kelas USK: {path}")
    return next(iter(matches), None)


def _split_from_path(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    matches = {
        SPLIT_ALIASES[_normalized(part)]
        for part in relative.parts[:-1]
        if _normalized(part) in SPLIT_ALIASES
    }
    if len(matches) > 1:
        raise ValueError(f"Path memuat lebih dari satu nama split: {path}")
    return next(iter(matches), None)


def _pair_id(path: Path, root: Path, class_name: str) -> str:
    stripped_stem = VIEW_SUFFIX.sub("", path.stem)
    if stripped_stem == path.stem:
        return f"{class_name}/unique/{path.relative_to(root).as_posix().lower()}"
    parent_parts = [
        part.lower()
        for part in path.relative_to(root).parent.parts
        if _normalized(part) not in CLASS_ALIASES
        and _normalized(part) not in SPLIT_ALIASES
    ]
    relative_parent = "/".join(parent_parts)
    stem = stripped_stem.lower()
    return f"{class_name}/{relative_parent}/{stem}"


def discover_samples(raw_root: Path) -> tuple[list[Sample], list[str]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Root USK tidak ditemukan: {raw_root}")
    samples: list[Sample] = []
    ignored: list[str] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        class_name = _class_from_path(path, raw_root)
        if class_name is None:
            ignored.append(path.relative_to(raw_root).as_posix())
            continue
        samples.append(
            Sample(
                path=path,
                class_name=class_name,
                split=_split_from_path(path, raw_root),
                pair_id=_pair_id(path, raw_root, class_name),
            )
        )
    if not samples:
        raise FileNotFoundError(
            "Tidak menemukan gambar dalam folder Defect/Longberry/Peaberry/Premium "
            f"di {raw_root}."
        )
    missing = sorted(set(CLASSES).difference(sample.class_name for sample in samples))
    if missing:
        raise ValueError(f"Kelas USK belum lengkap: {missing}")
    return samples, ignored


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deduplicate(samples: list[Sample]) -> tuple[list[Sample], list[dict], list[dict]]:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_hash[_sha256(sample.path)].append(sample)
    kept: list[Sample] = []
    duplicates: list[dict] = []
    conflicts: list[dict] = []
    for digest, group in sorted(by_hash.items()):
        classes = sorted({sample.class_name for sample in group})
        paths = sorted(str(sample.path) for sample in group)
        if len(classes) > 1:
            conflicts.append({"sha256": digest, "classes": classes, "paths": paths})
            continue
        canonical = sorted(group, key=lambda sample: str(sample.path))[0]
        kept.append(canonical)
        if len(group) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "class": canonical.class_name,
                    "kept": str(canonical.path),
                    "removed": [path for path in paths if path != str(canonical.path)],
                }
            )
    return kept, duplicates, conflicts


def _allocate_groups(
    samples: list[Sample], seed: int
) -> dict[Path, str]:
    by_class_group: dict[str, dict[str, list[Sample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        by_class_group[sample.class_name][sample.pair_id].append(sample)

    assignments: dict[Path, str] = {}
    for class_name, grouped in sorted(by_class_group.items()):
        groups = list(grouped.values())
        random.Random(f"{seed}:{class_name}").shuffle(groups)
        train_end = round(len(groups) * 0.60)
        val_end = train_end + round(len(groups) * 0.20)
        for index, group in enumerate(groups):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            for sample in group:
                assignments[sample.path] = split
    return assignments


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_usk_coffee(
    raw_root: Path,
    output_root: Path,
    seed: int = 42,
) -> dict:
    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: USK sudah disiapkan: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi audit belum lengkap. "
            "Hapus/pindahkan folder tersebut secara manual."
        )

    discovered, ignored = discover_samples(raw_root)
    samples, duplicates, conflicts = _deduplicate(discovered)
    split_presence = {sample.split is not None for sample in samples}
    if len(split_presence) > 1:
        raise ValueError(
            "Sebagian gambar memiliki folder split dan sebagian tidak. "
            "Rapikan arsip atau pilih subfolder dataset yang tepat."
        )
    if split_presence == {True}:
        present_splits = {sample.split for sample in samples}
        if present_splits != {"train", "val", "test"}:
            raise ValueError(f"Split arsip belum lengkap: {sorted(present_splits)}")
        assignments = {sample.path: str(sample.split) for sample in samples}
        split_strategy = "preserved_from_archive"
    else:
        assignments = _allocate_groups(samples, seed)
        split_strategy = "generated_60_20_20"

    pair_splits: dict[str, set[str]] = defaultdict(set)
    pair_counts: Counter[str] = Counter()
    for sample in samples:
        pair_splits[sample.pair_id].add(assignments[sample.path])
        pair_counts[sample.pair_id] += 1
    cross_split_pairs = {
        pair_id: sorted(splits)
        for pair_id, splits in pair_splits.items()
        if len(splits) > 1
    }

    output_root.mkdir(parents=True)
    for sample in samples:
        split = assignments[sample.path]
        relative = sample.path.relative_to(raw_root).as_posix()
        prefix = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
        destination = (
            output_root
            / "source"
            / split
            / sample.class_name
            / f"{prefix}_{sample.path.name}"
        )
        _link_or_copy(sample.path, destination)

    split_counts = {
        split: dict(
            sorted(
                Counter(
                    sample.class_name
                    for sample in samples
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
        "clean_images": len(samples),
        "class_counts": dict(sorted(Counter(s.class_name for s in samples).items())),
        "split_counts": split_counts,
        "split_strategy": split_strategy,
        "seed": seed,
        "same_class_exact_duplicates": duplicates,
        "cross_class_exact_conflicts": conflicts,
        "ignored_images": ignored,
        "detected_view_pairs": sum(count > 1 for count in pair_counts.values()),
        "cross_split_view_pairs": cross_split_pairs,
        "pairing_note": (
            "Pairing hanya dikenali bila filename berakhiran front/back, depan/belakang, "
            "side1/side2, atau view1/view2. Audit tambahan diperlukan bila konvensi "
            "nama dataset berbeda."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n=== AUDIT USK-COFFEE ===")
    print(f"Input                 : {len(discovered)}")
    print(f"Bersih                : {len(samples)}")
    print(f"Strategi split        : {split_strategy}")
    print(f"Exact duplicate group : {len(duplicates)}")
    print(f"Konflik label         : {len(conflicts)}")
    print(f"View pair terdeteksi  : {audit['detected_view_pairs']}")
    print(f"Pair lintas split     : {len(cross_split_pairs)}")
    for split, counts in split_counts.items():
        print(f"{split:5s}: {sum(counts.values()):5d} | {counts}")
    print(f"SAVED: {audit_path}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalisasi dan audit dataset USK-Coffee untuk ImageFolder"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_usk_coffee(args.raw_root, args.output_root, args.seed)


if __name__ == "__main__":
    main()
