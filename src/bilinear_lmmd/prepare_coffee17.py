from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sujitraarw/coffee-green-bean-with-17-defects-original"
)

EXPECTED_COUNTS = {
    "Broken": 62,
    "Cut": 66,
    "Dry Cherry": 54,
    "Fade": 35,
    "Floater": 48,
    "Full Black": 41,
    "Full Sour": 75,
    "Fungus Damage": 75,
    "Husk": 53,
    "Immature": 78,
    "Parchment": 54,
    "Partial Black": 65,
    "Partial Sour": 50,
    "Severe Insect Damage": 57,
    "Shell": 57,
    "Slight Insect Damage": 55,
    "Withered": 54,
}

CLASS_ALIASES = {
    # The public archive contains these two typos; the paper uses "Damage".
    "Fungus Damange": "Fungus Damage",
    "Severe Insect Damange": "Severe Insect Damage",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh dataset Kaggle ke {destination} ...")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _allocate(count: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_count = round(count * train_ratio)
    val_count = round(count * val_ratio)
    if train_count + val_count >= count:
        val_count = max(1, count - train_count - 1)
    return train_count, val_count


def discover_directory_samples(
    raw_root: Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, list[Path]]:
    """Discover Coffee17 class folders inside a mounted Kaggle input tree."""

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Folder input Coffee17 tidak ditemukan: {raw_root}")
    expected = expected_counts or EXPECTED_COUNTS
    source_names = set(expected)
    source_names.update(
        alias for alias, canonical in CLASS_ALIASES.items() if canonical in expected
    )
    by_class: dict[str, list[Path]] = {}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(raw_root)
        matches = {
            CLASS_ALIASES.get(part, part)
            for part in relative.parts[:-1]
            if part in source_names
        }
        if len(matches) == 1:
            by_class.setdefault(next(iter(matches)), []).append(path)

    observed = {name: len(paths) for name, paths in sorted(by_class.items())}
    if observed != expected:
        raise ValueError(
            f"Isi folder berbeda dari paper. Ditemukan {observed}, "
            f"diharapkan {expected}. Pastikan hanya satu versi Coffee17 terpasang."
        )
    return by_class


def prepare_from_directory(
    output: Path,
    raw_root: Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> None:
    """Prepare Coffee17 directly from Kaggle's read-only mounted folders."""

    source_root = output / "source"
    if source_root.exists():
        raise FileExistsError(
            f"{source_root} sudah ada. Hapus/pindahkan secara manual jika ingin menyiapkan ulang."
        )
    by_class = discover_directory_samples(raw_root)
    rng = random.Random(seed)
    for class_name, original_paths in sorted(by_class.items()):
        paths = list(original_paths)
        rng.shuffle(paths)
        train_count, val_count = _allocate(len(paths), train_ratio, val_ratio)
        split_paths = {
            "train": paths[:train_count],
            "val": paths[train_count : train_count + val_count],
            "test": paths[train_count + val_count :],
        }
        for split, selected in split_paths.items():
            destination = source_root / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for path in selected:
                target = destination / path.name
                if target.exists():
                    raise ValueError(
                        f"Nama file Coffee17 bertabrakan dalam kelas {class_name}: {path.name}"
                    )
                shutil.copy2(path, target)

    _print_split_summary(source_root)


def _print_split_summary(source_root: Path) -> None:
    split_totals = {
        split: len(list((source_root / split).glob("*/*")))
        for split in ("train", "val", "test")
    }
    print(f"Selesai: {split_totals}; total={sum(split_totals.values())}")
    print("Rotasi tidak ditulis sebagai file baru; augmentasi hanya dijalankan online pada train.")


def prepare(
    output: Path,
    archive: Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> None:
    if not archive.exists():
        _download(archive)
    source_root = output / "source"
    if source_root.exists():
        raise FileExistsError(
            f"{source_root} sudah ada. Hapus/pindahkan secara manual jika ingin menyiapkan ulang."
        )

    rng = random.Random(seed)
    observed: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as bundle:
        by_class: dict[str, list[zipfile.ZipInfo]] = {}
        for item in bundle.infolist():
            path = Path(item.filename)
            if item.is_dir() or len(path.parts) != 2:
                continue
            class_name = CLASS_ALIASES.get(path.parts[0], path.parts[0])
            if class_name in EXPECTED_COUNTS and path.suffix.lower() in IMAGE_SUFFIXES:
                by_class.setdefault(class_name, []).append(item)

        observed.update({name: len(items) for name, items in by_class.items()})
        if dict(observed) != EXPECTED_COUNTS:
            raise ValueError(
                f"Isi dataset berbeda dari paper. Ditemukan {dict(observed)}, "
                f"diharapkan {EXPECTED_COUNTS}."
            )

        for class_name, items in sorted(by_class.items()):
            rng.shuffle(items)
            train_count, val_count = _allocate(len(items), train_ratio, val_ratio)
            split_items = {
                "train": items[:train_count],
                "val": items[train_count : train_count + val_count],
                "test": items[train_count + val_count :],
            }
            for split, selected in split_items.items():
                destination = source_root / split / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for item in selected:
                    filename = Path(item.filename).name
                    with bundle.open(item) as src, (destination / filename).open("wb") as dst:
                        shutil.copyfileobj(src, dst)

    _print_split_summary(source_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unduh dan split Coffee Green Bean with 17 Defects tanpa leakage rotasi"
    )
    parser.add_argument("--output", type=Path, default=Path("data/coffee"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/coffee17.zip"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Folder Kaggle input yang sudah diekstrak; jika diberikan, download dilewati.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Rasio train/val harus positif dan jumlahnya kurang dari 1.")
    if args.raw_root is not None:
        prepare_from_directory(
            args.output,
            args.raw_root,
            args.seed,
            args.train_ratio,
            args.val_ratio,
        )
    else:
        prepare(args.output, args.archive, args.seed, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
