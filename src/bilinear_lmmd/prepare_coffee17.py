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
            if class_name in EXPECTED_COUNTS and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
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

    split_totals = {
        split: len(list((source_root / split).glob("*/*")))
        for split in ("train", "val", "test")
    }
    print(f"Selesai: {split_totals}; total={sum(split_totals.values())}")
    print("Rotasi tidak ditulis sebagai file baru; augmentasi hanya dijalankan online pada train.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unduh dan split Coffee Green Bean with 17 Defects tanpa leakage rotasi"
    )
    parser.add_argument("--output", type=Path, default=Path("data/coffee"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/coffee17.zip"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Rasio train/val harus positif dan jumlahnya kurang dari 1.")
    prepare(args.output, args.archive, args.seed, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
