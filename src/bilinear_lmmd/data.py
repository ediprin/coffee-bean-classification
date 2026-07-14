from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class DomainLoaders:
    source_train: DataLoader
    source_val: DataLoader
    target_train: DataLoader | None
    target_val: DataLoader | None
    classes: list[str]


class DiscreteRotation:
    """Choose one paper-aligned rotation without creating duplicate files."""

    def __init__(self, angles: list[float]):
        self.angles = angles or [0]

    def __call__(self, image: Image.Image) -> Image.Image:
        return TF.rotate(image, random.choice(self.angles), fill=255)


def _transforms(image_size: int, train: bool, rotation_angles: list[float]):
    if train:
        return transforms.Compose(
            [
                DiscreteRotation(rotation_angles),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_loaders(cfg: dict, require_target: bool = True) -> DomainLoaders:
    root = Path(cfg["root"])
    train_split = cfg.get("train_split", "train")
    val_split = cfg.get("val_split", "val")
    image_size = int(cfg["image_size"])

    source_paths = {
        "source_train": root / cfg["source"] / train_split,
        "source_val": root / cfg["source"] / val_split,
    }
    target_paths = {
        "target_train": root / cfg["target"] / train_split,
        "target_val": root / cfg["target"] / val_split,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_dir()]
    target_exists = all(path.is_dir() for path in target_paths.values())
    if require_target and not target_exists:
        missing.extend(str(path) for path in target_paths.values() if not path.is_dir())
    if missing:
        raise FileNotFoundError(
            "Folder dataset belum lengkap:\n- " + "\n- ".join(missing)
        )

    paths = dict(source_paths)
    if target_exists:
        paths.update(target_paths)
    rotation_angles = [float(angle) for angle in cfg.get("rotation_angles", [0])]
    datasets_by_split = {
        name: datasets.ImageFolder(
            path,
            transform=_transforms(
                image_size,
                train=name.endswith("train"),
                rotation_angles=rotation_angles,
            ),
        )
        for name, path in paths.items()
    }
    expected = datasets_by_split["source_train"].class_to_idx
    for name, dataset in datasets_by_split.items():
        if dataset.class_to_idx != expected:
            raise ValueError(
                f"Pemetaan kelas {name} berbeda. Closed-set UDA mensyaratkan "
                "nama folder kelas source dan target identik."
            )

    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("workers", 4)),
        "pin_memory": True,
    }
    loaders = {
        name: DataLoader(
            dataset,
            shuffle=name.endswith("train"),
            drop_last=name.endswith("train"),
            **loader_kwargs,
        )
        for name, dataset in datasets_by_split.items()
    }
    return DomainLoaders(
        source_train=loaders["source_train"],
        source_val=loaders["source_val"],
        target_train=loaders.get("target_train"),
        target_val=loaders.get("target_val"),
        classes=datasets_by_split["source_train"].classes,
    )
