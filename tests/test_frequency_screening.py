from __future__ import annotations

import torch

from bilinear_lmmd.modeling.frequency_screening import (
    assert_shared_core_equal,
    build_frequency_screening_model,
    haar_dwt2d,
    haar_iwt2d,
)


def _cfg() -> dict:
    return {
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": False,
            "classifier": "linear",
            "num_classes": 17,
            "projection_dim": 64,
            "dropout": 0.0,
        },
        "frequency": {
            "attention_reduction": 16,
            "fda_hidden": 4,
        },
    }


def test_haar_roundtrip_even_and_odd() -> None:
    torch.manual_seed(7)
    for shape in ((2, 5, 14, 14), (2, 5, 7, 7)):
        x = torch.randn(*shape)
        bands, original_hw = haar_dwt2d(x)
        reconstructed = haar_iwt2d(bands, original_hw)
        torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-6)


def test_frequency_gap_candidates_forward() -> None:
    cfg = _cfg()
    images = torch.randn(2, 3, 224, 224)
    for candidate in ("W1", "W3", "W5"):
        torch.manual_seed(42)
        model = build_frequency_screening_model(candidate, cfg)
        output = model(images)
        assert output.logits.shape == (2, 17)
        assert output.embedding.ndim == 2


def test_frequency_hbp_candidates_forward() -> None:
    cfg = _cfg()
    images = torch.randn(2, 3, 224, 224)
    for candidate in ("W2", "W4", "W6"):
        torch.manual_seed(42)
        model = build_frequency_screening_model(candidate, cfg)
        output = model(images)
        assert output.logits.shape == (2, 17)
        assert output.embedding.shape == (2, 64 * 3)


def test_shared_core_initialization_matches_anchor() -> None:
    cfg = _cfg()
    for candidate, anchor in (
        ("W1", "B0"),
        ("W2", "B1"),
        ("W3", "B0"),
        ("W4", "B1"),
        ("W5", "B0"),
        ("W6", "B1"),
    ):
        torch.manual_seed(42)
        baseline = build_frequency_screening_model(anchor, cfg)
        torch.manual_seed(42)
        model = build_frequency_screening_model(candidate, cfg)
        assert_shared_core_equal(baseline, model)
