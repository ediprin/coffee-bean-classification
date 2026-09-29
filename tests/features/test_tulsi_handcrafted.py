from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bilinear_lmmd.features.tulsi_handcrafted import (
    FEATURE_NAMES,
    extract_tulsi71,
    mrmr_select,
)


def _synthetic_bean(path: Path) -> None:
    h = w = 128
    yy, xx = np.mgrid[:h, :w]
    rgb = np.full((h, w, 3), 245, dtype=np.uint8)
    bean = ((xx - 64) / 38.0) ** 2 + ((yy - 64) / 48.0) ** 2 <= 1.0
    rgb[bean, 0] = 120 + ((xx[bean] + yy[bean]) % 20).astype(np.uint8)
    rgb[bean, 1] = 145 + ((2 * xx[bean]) % 15).astype(np.uint8)
    rgb[bean, 2] = 90 + ((3 * yy[bean]) % 12).astype(np.uint8)
    defect = (xx - 76) ** 2 + (yy - 55) ** 2 <= 8 ** 2
    rgb[defect, :] = np.array([55, 45, 35], dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def test_tulsi_feature_contract_has_exactly_71_unique_names() -> None:
    assert len(FEATURE_NAMES) == 71
    assert len(set(FEATURE_NAMES)) == 71
    assert FEATURE_NAMES[0] == "Average_Blue"
    assert FEATURE_NAMES[-1] == "Eccentricity"


def test_tulsi71_extracts_finite_vector(tmp_path) -> None:
    path = tmp_path / "bean.png"
    _synthetic_bean(path)
    values = extract_tulsi71(path)
    assert values.shape == (71,)
    assert np.isfinite(values).all()


def test_mrmr_is_deterministic_and_finds_signal() -> None:
    rng = np.random.default_rng(7)
    n = 240
    y = np.repeat(np.arange(3), n // 3)
    x = rng.normal(size=(n, 71))
    x[:, 5] = y + rng.normal(scale=0.03, size=n)
    x[:, 17] = (y == 2).astype(float) + rng.normal(scale=0.03, size=n)

    first = mrmr_select(x, y, k=20, random_state=42)
    second = mrmr_select(x, y, k=20, random_state=42)

    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    assert 5 in first
