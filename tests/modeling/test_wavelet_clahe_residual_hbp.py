from __future__ import annotations

import copy

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.engine.wavelet_clahe_residual_hbp import (
    build_clahe_frontend,
    build_wr_control,
    build_wrc_candidate,
    clahe_luminance_residual,
)
from bilinear_lmmd.modeling.wavelet_clahe_residual_hbp import assert_shared_wr_equal


def _cfg() -> dict:
    cfg = copy.deepcopy(
        load_config("configs/wavelet_clahe_residual_hbp/WRC_HBP_V1.yaml")
    )
    cfg["model"]["pretrained"] = False
    return cfg


def test_clahe_luminance_residual_shape_and_finite() -> None:
    cfg = _cfg()
    frontend = build_clahe_frontend(cfg)
    raw = torch.rand(2, 3, 224, 224)
    residual = clahe_luminance_residual(raw, frontend)
    assert residual.shape == (2, 1, 224, 224)
    assert torch.isfinite(residual).all()


def test_wrc_starts_from_exact_wr_state() -> None:
    cfg = _cfg()

    torch.manual_seed(42)
    control = build_wr_control(cfg)

    torch.manual_seed(42)
    candidate = build_wrc_candidate(cfg)

    assert_shared_wr_equal(control, candidate)
    assert candidate.contrast_gate_value().item() == 0.0
    assert candidate.gate_value().item() == control.gate_value().item() == 0.0


def test_zero_contrast_gate_logits_match_wr_control() -> None:
    cfg = _cfg()

    torch.manual_seed(42)
    control = build_wr_control(cfg).eval()

    torch.manual_seed(42)
    candidate = build_wrc_candidate(cfg).eval()

    frontend = build_clahe_frontend(cfg)
    raw = torch.rand(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)
    contrast = clahe_luminance_residual(raw, frontend)

    with torch.no_grad():
        control_logits = control(normalized, raw_rgb=raw).logits
        candidate_logits = candidate(
            normalized,
            raw_rgb=raw,
            contrast_residual=contrast,
        ).logits

    torch.testing.assert_close(
        control_logits,
        candidate_logits,
        rtol=0.0,
        atol=1.0e-7,
    )
