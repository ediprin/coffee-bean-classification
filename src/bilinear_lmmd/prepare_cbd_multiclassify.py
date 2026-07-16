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
EXPECTED_CLASSES = (
    "Black",
    "Broken",
    "Cherry",
    "Damage",
    "Dried",
    "Floater",
    "Fungus",
    "Good",
    "Insect",
    "Sour",
)
CLASS_ALIASES = {
    re.sub(r"[^a-z0-9]+", "", name.lower()): name for name in EXPECTED_CLASSES
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
}
UNLABELED = {"unlabeled", "unlabelled"}
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[a-z0-9_-]+$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    archive_split: str | None
    identity_id: str


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _class_and_split(path: Path, root: Path) -> tuple[str | None, str | None]:
    relative = path.relative_to(root)
    directories = list(relative.parts[:-1])
    split_matches = [
        (index, SPLIT_ALIASES[_normalized(part)])
        for index, part in enumerate(directories)
        if _normalized(part) in SPLIT_ALIASES
    ]
    if len(split_matches) > 1:
        raise ValueError(f"Path memuat lebih dari satu split: {path}")
    if split_matches:
        index, split = split_matches[0]
        if index + 1 >= len(directories):
            return None, split
        class_part = directories[index + 1]
    else:
        split = None
        class_part = path.parent.name

    normalized = _normalized(class_part)
    if normalized in UNLABELED:
        return "Unlabeled", split
    return CLASS_ALIASES.get(normalized), split


def _identity_id(path: Path, root: Path, class_name: str) -> str:
    base = ROBOFLOW_SUFFIX.sub("", path.stem)
    base = _normalized(base)
    if not base:
        base = hashlib.sha1(
            path.relative_to(root).as_posix().encode("utf-8")
        ).hexdigest()
    return f"{class_name}/{base}"


def discover_samples(
    raw_root: Path,
) -> tuple[list[Sample], list[str], list[str]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Root cbd-multiclassify tidak ditemukan: {raw_root}")
    samples: list[Sample] = []
    unlabeled: list[str] = []
    unknown: list[str] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        class_name, archive_split = _class_and_split(path, raw_root)
        relative = path.relative_to(raw_root).as_posix()
        if class_name == "Unlabeled":
            unlabeled.append(relative)
            continue
        if class_name is None:
            unknown.append(relative)
            continue
        samples.append(
            Sample(
                path=path,
                class_name=class_name,
                archive_split=archive_split,
                identity_id=_identity_id(path, raw_root, class_name),
            )
        )
    if not samples:
        raise FileNotFoundError(
            "Tidak menemukan kelas cbd-multiclassify di root yang diberikan."
        )
    observed = {sample.class_name for sample in samples}
    missing = sorted(set(EXPECTED_CLASSES).difference(observed))
    if missing:
        raise ValueError(
            f"Kelas cbd-multiclassify belum lengkap: {missing}. "
            "Pastikan root menunjuk ke ekspor dataset yang benar."
        )
    return samples, unlabeled, unknown


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deduplicate(
    samples: list[Sample],
) -> tuple[list[Sample], list[dict], list[dict]]:
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


def _archive_identity_overlap(samples: list[Sample]) -> dict[str, list[str]]:
    identity_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.archive_split is not None:
            identity_splits[sample.identity_id].add(sample.archive_split)
    return {
        identity: sorted(splits)
        for identity, splits in sorted(identity_splits.items())
        if len(splits) > 1
    }


def _allocate_identity_groups(samples: list[Sample], seed: int) -> dict[Path, str]:
    by_class: dict[str, dict[str, list[Sample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        by_class[sample.class_name][sample.identity_id].append(sample)

    assignments: dict[Path, str] = {}
    for class_name, grouped in sorted(by_class.items()):
        groups = list(grouped.values())
        if len(groups) < 5:
            raise ValueError(
                f"Kelas {class_name} hanya memiliki {len(groups)} identitas; "
                "minimal lima untuk split 60/20/20."
            )
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


def prepare_cbd_multiclassify(
    raw_root: Path,
    output_root: Path,
    seed: int = 42,
) -> dict:
    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete":
            print(f"SKIP: cbd-multiclassify sudah disiapkan: {output_root}")
            return audit
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi audit belum lengkap. "
            "Hapus/pindahkan folder tersebut secara manual."
        )

    discovered, unlabeled, unknown = discover_samples(raw_root)
    archive_overlap = _archive_identity_overlap(discovered)
    samples, duplicates, conflicts = _deduplicate(discovered)
    assignments = _allocate_identity_groups(samples, seed)

    output_root.mkdir(parents=True)
    for sample in samples:
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
                    for sample in samples
                    if assignments[sample.path] == split
                ).items()
            )
        )
        for split in ("train", "val", "test")
    }
    identity_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        identity_splits[sample.identity_id].add(assignments[sample.path])
    generated_overlap = {
        identity: sorted(splits)
        for identity, splits in identity_splits.items()
        if len(splits) > 1
    }
    if generated_overlap:
        raise RuntimeError("Bug internal: identitas Roboflow tersebar lintas split.")

    audit = {
        "status": "complete",
        "dataset": "asdasd-zsar1/cbd-multiclassify",
        "source_url": "https://universe.roboflow.com/asdasd-zsar1/cbd-multiclassify",
        "raw_root": str(raw_root.resolve()),
        "input_labeled_images": len(discovered),
        "clean_labeled_images": len(samples),
        "classes": list(EXPECTED_CLASSES),
        "class_counts": dict(sorted(Counter(s.class_name for s in samples).items())),
        "excluded_unlabeled_count": len(unlabeled),
        "excluded_unlabeled_paths": unlabeled,
        "unknown_image_count": len(unknown),
        "unknown_image_examples": unknown[:100],
        "same_class_exact_duplicates": duplicates,
        "cross_class_exact_conflicts": conflicts,
        "archive_cross_split_identity_count": len(archive_overlap),
        "archive_cross_split_identities": archive_overlap,
        "generated_cross_split_identity_count": 0,
        "split_strategy": "generated_identity_grouped_60_20_20",
        "split_counts": split_counts,
        "seed": seed,
        "identity_rule": (
            "Nama dasar sebelum suffix .rf.<hash> dipakai untuk mengelompokkan "
            "variant Roboflow dari gambar asal yang sama."
        ),
        "claim_note": (
            "Benchmark publik tambahan dengan label kasar; tidak digabungkan "
            "langsung dengan Coffee-17 dan bukan external test 17 kelas."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n=== AUDIT CBD-MULTICLASSIFY ===")
    print(f"Input berlabel       : {len(discovered)}")
    print(f"Bersih berlabel      : {len(samples)}")
    print(f"Unlabeled dikeluarkan: {len(unlabeled)}")
    print(f"Exact duplicate group: {len(duplicates)}")
    print(f"Konflik label        : {len(conflicts)}")
    print(f"Identity leak arsip  : {len(archive_overlap)}")
    print("Identity leak baru   : 0")
    for split, counts in split_counts.items():
        print(f"{split:5s}: {sum(counts.values()):5d} | {counts}")
    print(f"SAVED: {audit_path}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dan grouped split dataset Roboflow cbd-multiclassify"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_cbd_multiclassify(args.raw_root, args.output_root, args.seed)


if __name__ == "__main__":
    main()
