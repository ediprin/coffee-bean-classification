from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, transform
from skimage.feature import graycoprops, local_binary_pattern


FEATURE_VERSION = 2
COLOR_PREFIX = "color::"
SHAPE_PREFIX = "shape::"
TEXTURE_PREFIX = "texture::"


@dataclass(frozen=True)
class AttributeFeatures:
    values: np.ndarray
    names: list[str]
    mask: np.ndarray
    mask_area_fraction: float


def load_rgb(path: str | Path, max_size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def segment_bean(rgb: np.ndarray) -> np.ndarray:
    """Segment one bean against a mostly uniform light background."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image expected, received shape={rgb.shape}.")
    height, width = rgb.shape[:2]
    border_width = max(2, round(min(height, width) * 0.03))
    lab = color.rgb2lab(rgb)
    border = np.concatenate(
        (
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    threshold = filters.threshold_otsu(distance)
    mask = distance > threshold

    radius = max(1, round(min(height, width) * 0.008))
    footprint = morphology.disk(radius)
    mask = morphology.opening(mask, footprint)
    mask = morphology.closing(mask, footprint)
    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        raise ValueError("Mask biji kosong setelah thresholding.")
    largest = max(regions, key=lambda region: region.area)
    mask = labels == largest.label
    mask = ndi.binary_fill_holes(mask)

    fraction = float(mask.mean())
    if not 0.02 <= fraction <= 0.90:
        raise ValueError(f"Luas mask tidak masuk akal: {fraction:.2%}.")
    return mask.astype(bool)


def _distribution_features(
    values: np.ndarray, prefix: str
) -> tuple[list[float], list[str]]:
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    features = [float(values.mean()), float(values.std())]
    names = [f"{prefix}_mean", f"{prefix}_std"]
    features.extend(float(value) for value in np.quantile(values, quantiles))
    names.extend(f"{prefix}_q{int(quantile * 100):02d}" for quantile in quantiles)
    return features, names


def color_features(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    spaces = {
        "lab": (color.rgb2lab(rgb), ("l", "a", "b")),
        "hsv": (color.rgb2hsv(rgb), ("h", "s", "v")),
    }
    features: list[float] = []
    names: list[str] = []
    for space_name, (converted, channels) in spaces.items():
        for channel_index, channel_name in enumerate(channels):
            channel_features, channel_names = _distribution_features(
                converted[..., channel_index][mask],
                f"{COLOR_PREFIX}{space_name}_{channel_name}",
            )
            features.extend(channel_features)
            names.extend(channel_names)
    return np.asarray(features, dtype=np.float64), names


def shape_features(mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    region = measure.regionprops(measure.label(mask))[0]
    height, width = mask.shape
    image_area = height * width
    scale = float(max(height, width))
    area = float(region.area)
    perimeter = max(float(region.perimeter), 1e-8)
    minor_axis = max(float(region.axis_minor_length), 1e-8)
    base = {
        "area_fraction": area / image_area,
        "perimeter_norm": perimeter / (2.0 * (height + width)),
        "major_axis_norm": float(region.axis_major_length) / scale,
        "minor_axis_norm": float(region.axis_minor_length) / scale,
        "aspect_ratio": float(region.axis_major_length) / minor_axis,
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "extent": float(region.extent),
        "equivalent_diameter_norm": float(region.equivalent_diameter_area) / scale,
        "feret_diameter_norm": float(region.feret_diameter_max) / scale,
        "circularity": float(4.0 * np.pi * area / (perimeter**2)),
    }
    names = [f"{SHAPE_PREFIX}{name}" for name in base]
    features = list(base.values())
    for index, moment in enumerate(region.moments_hu):
        signed_log = -np.sign(moment) * np.log10(abs(moment) + 1e-30)
        names.append(f"{SHAPE_PREFIX}hu_{index + 1}")
        features.append(float(signed_log))
    return np.asarray(features, dtype=np.float64), names


def _masked_glcm(
    quantized: np.ndarray,
    mask: np.ndarray,
    distances: tuple[int, ...],
    angles: tuple[float, ...],
    levels: int,
) -> np.ndarray:
    height, width = quantized.shape
    matrices = np.zeros(
        (levels, levels, len(distances), len(angles)), dtype=np.float64
    )
    for distance_index, distance in enumerate(distances):
        for angle_index, angle in enumerate(angles):
            dx = int(round(np.cos(angle) * distance))
            dy = int(round(np.sin(angle) * distance))
            y_start = max(0, -dy)
            y_stop = min(height, height - dy)
            x_start = max(0, -dx)
            x_stop = min(width, width - dx)
            source = quantized[y_start:y_stop, x_start:x_stop]
            target = quantized[
                y_start + dy : y_stop + dy,
                x_start + dx : x_stop + dx,
            ]
            valid = mask[y_start:y_stop, x_start:x_stop] & mask[
                y_start + dy : y_stop + dy,
                x_start + dx : x_stop + dx,
            ]
            if not valid.any():
                matrices[0, 0, distance_index, angle_index] = 1.0
                continue
            pairs = source[valid] * levels + target[valid]
            matrix = np.bincount(pairs, minlength=levels * levels).reshape(
                levels, levels
            )
            matrix = matrix + matrix.T
            matrices[:, :, distance_index, angle_index] = matrix / matrix.sum()
    return matrices


def texture_features(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    gray = color.rgb2gray(rgb)
    if max(gray.shape) > 256:
        scale = 256.0 / max(gray.shape)
        output_shape = tuple(max(1, round(size * scale)) for size in gray.shape)
        gray = transform.resize(
            gray, output_shape, preserve_range=True, anti_aliasing=True
        )
        mask = transform.resize(
            mask.astype(np.uint8),
            output_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(bool)
    levels = 16
    quantized = np.clip((gray * levels).astype(np.int64), 0, levels - 1)
    distances = (1, 2, 4, 8)
    angles = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
    matrices = _masked_glcm(quantized, mask, distances, angles, levels)
    features: list[float] = []
    names: list[str] = []
    for property_name in (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ):
        values = np.nan_to_num(graycoprops(matrices, property_name))
        for distance_index, distance in enumerate(distances):
            across_angles = values[distance_index]
            features.extend(
                (float(across_angles.mean()), float(across_angles.std()))
            )
            names.extend(
                (
                    f"{TEXTURE_PREFIX}glcm_{property_name}_d{distance}_mean",
                    f"{TEXTURE_PREFIX}glcm_{property_name}_d{distance}_std",
                )
            )

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    inner_mask = morphology.erosion(mask, morphology.disk(2))
    if inner_mask.sum() < 100:
        inner_mask = mask
    histogram, _ = np.histogram(
        lbp[inner_mask], bins=np.arange(0, 11), density=False
    )
    histogram = histogram.astype(np.float64)
    histogram /= max(histogram.sum(), 1.0)
    features.extend(histogram.tolist())
    names.extend(f"{TEXTURE_PREFIX}lbp_uniform_{index}" for index in range(10))
    return np.asarray(features, dtype=np.float64), names


def extract_attributes(path: str | Path) -> AttributeFeatures:
    rgb = load_rgb(path)
    mask = segment_bean(rgb)
    groups = (
        color_features(rgb, mask),
        shape_features(mask),
        texture_features(rgb, mask),
    )
    values = np.concatenate([group[0] for group in groups])
    names = [name for group in groups for name in group[1]]
    if not np.isfinite(values).all():
        raise ValueError(f"Fitur non-finite ditemukan pada {path}.")
    return AttributeFeatures(
        values=values,
        names=names,
        mask=mask,
        mask_area_fraction=float(mask.mean()),
    )


def feature_group_indices(names: list[str]) -> dict[str, np.ndarray]:
    prefixes = {
        "color": COLOR_PREFIX,
        "shape": SHAPE_PREFIX,
        "texture": TEXTURE_PREFIX,
    }
    return {
        group: np.asarray(
            [index for index, name in enumerate(names) if name.startswith(prefix)],
            dtype=np.int64,
        )
        for group, prefix in prefixes.items()
    }
