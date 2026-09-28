from __future__ import annotations

import numpy as np
from PIL import Image

from bilinear_lmmd.analysis.coffee17_physical_descriptors import (
    ALL_FEATURES,
    bean_mask_from_background_otsu,
    extract_physical_descriptors,
)


def _synthetic_bean() -> np.ndarray:
    image = np.ones((128, 128, 3), dtype=np.float32)
    yy, xx = np.mgrid[:128, :128]
    mask = ((xx - 64) / 35.0) ** 2 + ((yy - 64) / 24.0) ** 2 <= 1.0
    image[mask] = np.array([0.45, 0.55, 0.35], dtype=np.float32)
    # Add a small dark internal spot.
    spot = (xx - 72) ** 2 + (yy - 58) ** 2 <= 3 ** 2
    image[spot] = np.array([0.08, 0.07, 0.05], dtype=np.float32)
    return image


def test_background_otsu_mask_extracts_center_bean() -> None:
    image = _synthetic_bean()
    mask = bean_mask_from_background_otsu(image)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool
    assert mask[64, 64]
    assert not mask[0, 0]
    assert 0.05 < float(mask.mean()) < 0.5


def test_descriptor_extractor_is_finite_and_complete(tmp_path) -> None:
    image = (_synthetic_bean() * 255.0).round().astype(np.uint8)
    path = tmp_path / "bean.png"
    Image.fromarray(image).save(path)

    row = extract_physical_descriptors(path)
    for feature in ALL_FEATURES:
        assert feature in row
        assert np.isfinite(float(row[feature]))
    assert row["mask_qc_pass"] is True


def test_descriptor_extractor_is_label_free_by_signature(tmp_path) -> None:
    image = (_synthetic_bean() * 255.0).round().astype(np.uint8)
    path = tmp_path / "bean.png"
    Image.fromarray(image).save(path)

    row_a = extract_physical_descriptors(path)
    row_b = extract_physical_descriptors(path)
    for feature in ALL_FEATURES:
        assert float(row_a[feature]) == float(row_b[feature])
