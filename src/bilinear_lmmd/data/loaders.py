from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from bilinear_lmmd.data.attribute_features import segment_bean


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


class ObjectCentricCrop:
    """Crop one bean from a light background without feeding a mask to CNN."""

    def __init__(self, margin_fraction: float = 0.10, segmentation_size: int = 256):
        if not 0.0 <= margin_fraction <= 1.0:
            raise ValueError("object_crop_margin harus berada di rentang [0, 1].")
        if segmentation_size < 32:
            raise ValueError("segmentation_size minimal 32 piksel.")
        self.margin_fraction = margin_fraction
        self.segmentation_size = segmentation_size

    @staticmethod
    def _background_color(rgb: np.ndarray) -> tuple[int, int, int]:
        border = np.concatenate(
            (rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0
        )
        return tuple(np.median(border, axis=0).round().astype(np.uint8))

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        rgb_uint8 = np.asarray(image)
        segmentation_image = image.copy()
        segmentation_image.thumbnail(
            (self.segmentation_size, self.segmentation_size),
            Image.Resampling.BILINEAR,
        )
        segmentation_rgb = np.asarray(segmentation_image, dtype=np.float32) / 255.0
        mask = segment_bean(segmentation_rgb)
        rows, columns = np.nonzero(mask)
        scale_x = image.width / segmentation_image.width
        scale_y = image.height / segmentation_image.height
        top = math.floor(rows.min() * scale_y)
        bottom = math.ceil((rows.max() + 1) * scale_y)
        left = math.floor(columns.min() * scale_x)
        right = math.ceil((columns.max() + 1) * scale_x)
        margin = round(max(bottom - top, right - left) * self.margin_fraction)
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(image.width, right + margin)
        bottom = min(image.height, bottom + margin)
        crop = image.crop((left, top, right, bottom))

        side = max(crop.size)
        square = Image.new("RGB", (side, side), self._background_color(rgb_uint8))
        offset = ((side - crop.width) // 2, (side - crop.height) // 2)
        square.paste(crop, offset)
        return square


def _transforms(
    image_size: int,
    train: bool,
    rotation_angles: list[float],
    object_crop: bool = False,
    object_crop_margin: float = 0.10,
    augmentation_mode: str = "standard",
):
    if augmentation_mode not in {"standard", "paper"}:
        raise ValueError("augmentation_mode harus 'standard' atau 'paper'.")
    object_transforms = (
        [ObjectCentricCrop(object_crop_margin)] if object_crop else []
    )
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if augmentation_mode == "paper":
        return transforms.Compose(
            object_transforms
            + [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if train:
        return transforms.Compose(
            object_transforms
            + [
                DiscreteRotation(rotation_angles),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        object_transforms
        + [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_image_transform(
    image_size: int,
    train: bool = False,
    rotation_angles: list[float] | None = None,
    object_crop: bool = False,
    object_crop_margin: float = 0.10,
    augmentation_mode: str = "standard",
):
    """Public transform factory for evaluation tools outside DomainLoaders."""

    return _transforms(
        image_size=image_size,
        train=train,
        rotation_angles=rotation_angles or [0],
        object_crop=object_crop,
        object_crop_margin=object_crop_margin,
        augmentation_mode=augmentation_mode,
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
    object_crop = bool(cfg.get("object_crop", False))
    object_crop_margin = float(cfg.get("object_crop_margin", 0.10))
    augmentation_mode = str(cfg.get("augmentation_mode", "standard"))
    datasets_by_split = {
        name: datasets.ImageFolder(
            path,
            transform=_transforms(
                image_size,
                train=name.endswith("train"),
                rotation_angles=rotation_angles,
                object_crop=object_crop,
                object_crop_margin=object_crop_margin,
                augmentation_mode=augmentation_mode,
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
