from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology


SHAPE_FEATURES = (
    "area_fraction",
    "perimeter_over_sqrt_area",
    "circularity",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "extent",
)

LAB_FEATURES = tuple(
    f"{channel}_{stat}"
    for channel in ("L", "a", "b")
    for stat in ("mean", "std", "q10", "q25", "q50", "q75", "q90")
)

L_COVERAGE_FEATURES = tuple(f"L_frac_lt_{v}" for v in (20, 30, 40, 50, 60, 70))

WARM_COVERAGE_FEATURES = (
    "warm_frac_a0_b10",
    "warm_frac_a5_b10",
    "warm_frac_a5_b15",
    "warm_frac_a10_b15",
    "warm_frac_a10_b20",
)

DARK_COMPONENT_FEATURES = tuple(
    f"{prefix}_k{k:g}".replace(".", "p")
    for k in (2.0, 2.5, 3.0)
    for prefix in (
        "dark_component_count",
        "dark_component_area_fraction",
        "largest_dark_component_fraction",
    )
)

ALL_FEATURES = (
    SHAPE_FEATURES
    + LAB_FEATURES
    + L_COVERAGE_FEATURES
    + WARM_COVERAGE_FEATURES
    + DARK_COMPONENT_FEATURES
)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def bean_mask_from_background_otsu(rgb: np.ndarray) -> np.ndarray:
    """Segment the single Coffee17 bean from its controlled light background.

    A robust background color is estimated from image-border pixels. Each pixel
    is scored by Euclidean RGB distance from that border background, then Otsu
    thresholding separates bean from background. This avoids a failure mode of
    red-channel Otsu where a small very-dark defect can become the foreground
    mode while the rest of a pale bean is discarded.

    The extraction is label-free and uses no Coffee17 class information.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb harus HxWx3.")

    height, width, _ = rgb.shape
    border_width = max(2, min(height, width) // 20)

    top = rgb[:border_width, :, :].reshape(-1, 3)
    bottom = rgb[-border_width:, :, :].reshape(-1, 3)
    if height > 2 * border_width:
        left = rgb[border_width:-border_width, :border_width, :].reshape(-1, 3)
        right = rgb[border_width:-border_width, -border_width:, :].reshape(-1, 3)
        border = np.concatenate([top, bottom, left, right], axis=0)
    else:
        border = np.concatenate([top, bottom], axis=0)

    background_rgb = np.median(border, axis=0)
    distance = np.linalg.norm(
        rgb - background_rgb.reshape(1, 1, 3),
        axis=2,
    )

    if float(distance.max()) <= 1.0e-8:
        raise ValueError("Image hampir seragam; bean tidak dapat disegmentasi.")

    threshold = filters.threshold_otsu(distance)
    mask = distance > threshold

    min_size = max(16, int(round(mask.size * 0.0005)))
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)

    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        raise ValueError("Bean mask kosong setelah background-distance Otsu.")
    region = max(regions, key=lambda r: r.area)
    return labels == region.label


def _shape_features(mask: np.ndarray) -> dict[str, float]:
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    region = max(measure.regionprops(labels), key=lambda r: r.area)

    area = float(region.area)
    perimeter = float(measure.perimeter(mask, neighborhood=8))
    circularity = (
        float(4.0 * math.pi * area / (perimeter * perimeter))
        if perimeter > 0.0
        else float("nan")
    )
    minor = float(region.axis_minor_length)
    major = float(region.axis_major_length)

    return {
        "area_fraction": area / float(mask.size),
        "perimeter_over_sqrt_area": perimeter / math.sqrt(area),
        "circularity": circularity,
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "aspect_ratio": major / minor if minor > 0.0 else float("nan"),
        "extent": float(region.extent),
    }


def _quantile_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90),
    }


def _lab_and_coverage_features(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    lab = color.rgb2lab(rgb)
    values = lab[mask]

    out: dict[str, float] = {}
    for index, channel in enumerate(("L", "a", "b")):
        out.update(_quantile_stats(values[:, index], channel))

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    n = float(mask.sum())

    for threshold in (20, 30, 40, 50, 60, 70):
        out[f"L_frac_lt_{threshold}"] = float(np.count_nonzero(mask & (L < threshold)) / n)

    # Fixed, label-free warm/yellow-red coverage proxies. These are explicitly
    # proxies, not validated sour masks. Positive a* is redward; positive b* is
    # yellowward in CIELAB. No thresholds are fit from Coffee17 labels.
    warm_specs = (
        ("warm_frac_a0_b10", 0.0, 10.0),
        ("warm_frac_a5_b10", 5.0, 10.0),
        ("warm_frac_a5_b15", 5.0, 15.0),
        ("warm_frac_a10_b15", 10.0, 15.0),
        ("warm_frac_a10_b20", 10.0, 20.0),
    )
    for name, ath, bth in warm_specs:
        out[name] = float(np.count_nonzero(mask & (a > ath) & (b > bth)) / n)

    return out, lab


def _dark_component_features(
    lab: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Label-free dark-spot topology proxies.

    They are NOT interpreted as exact insect-hole counts. The bean interior is
    eroded to reduce contour/edge artifacts. Thresholds are robust deviations
    below the within-bean median L*, frozen before class-wise analysis.
    """

    interior = morphology.binary_erosion(mask, morphology.disk(3))
    if interior.sum() < 32:
        interior = mask.copy()

    L = lab[..., 0]
    vals = L[interior]
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    robust_sigma = max(1.4826 * mad, 1.0e-6)
    bean_area = float(mask.sum())

    out: dict[str, float] = {}
    min_area = max(2, int(round(bean_area * 0.0002)))
    max_area = max(min_area + 1, int(round(bean_area * 0.03)))

    for k in (2.0, 2.5, 3.0):
        threshold = median - k * robust_sigma
        dark = interior & (L < threshold)
        labels = measure.label(dark, connectivity=2)
        components = [
            float(r.area)
            for r in measure.regionprops(labels)
            if min_area <= r.area <= max_area
        ]
        tag = f"k{k:g}".replace(".", "p")
        total = float(sum(components))
        largest = float(max(components)) if components else 0.0
        out[f"dark_component_count_{tag}"] = float(len(components))
        out[f"dark_component_area_fraction_{tag}"] = total / bean_area
        out[f"largest_dark_component_fraction_{tag}"] = largest / bean_area

    return out


def extract_physical_descriptors(path: Path) -> dict[str, float | int | bool]:
    rgb = _load_rgb(path)
    mask = bean_mask_from_background_otsu(rgb)

    area_fraction = float(mask.mean())
    rows, cols = np.nonzero(mask)
    touches_border = bool(
        rows.min() == 0
        or cols.min() == 0
        or rows.max() == mask.shape[0] - 1
        or cols.max() == mask.shape[1] - 1
    )

    result: dict[str, float | int | bool] = {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mask_area_pixels": int(mask.sum()),
        "mask_area_fraction_qc": area_fraction,
        "mask_touches_border": touches_border,
        "mask_qc_pass": bool(0.03 <= area_fraction <= 0.85 and not touches_border),
    }
    result.update(_shape_features(mask))
    color_features, lab = _lab_and_coverage_features(rgb, mask)
    result.update(color_features)
    result.update(_dark_component_features(lab, mask))
    return result
