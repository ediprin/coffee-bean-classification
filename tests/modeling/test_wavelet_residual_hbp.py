from __future__ import annotations

import copy

import pytest
import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.engine.wavelet_residual_hbp import build_candidate, build_control
from bilinear_lmmd.modeling.wavelet_residual_hbp import (
    assert_shared_hbp_core_equal,
    luminance_l1_visushrink_details,
)


def _cfg() -> dict:
    cfg = load_config("configs/wavelet_residual_hbp/WR_HBP_V1.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["pretrained"] = False
    return cfg


def test_luminance_l1_details_shape_and_finite() -> None:
    raw = torch.rand(2, 3, 224, 224)
    details = luminance_l1_visushrink_details(raw)
    assert details.shape == (2, 3, 112, 112)
    assert torch.isfinite(details).all()


def test_candidate_starts_from_exact_shared_hbp_core() -> None:
    cfg = _cfg()

    torch.manual_seed(42)
    control = build_control(cfg)

    torch.manual_seed(42)
    candidate = build_candidate(cfg)

    assert_shared_hbp_core_equal(control, candidate)
    assert candidate.gate_value().item() == 0.0


def test_gate_zero_logits_match_control() -> None:
    cfg = _cfg()

    torch.manual_seed(42)
    control = build_control(cfg).eval()

    torch.manual_seed(42)
    candidate = build_candidate(cfg).eval()

    raw = torch.rand(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)

    with torch.no_grad():
        control_logits = control(normalized).logits
        candidate_logits = candidate(normalized, raw_rgb=raw).logits

    torch.testing.assert_close(
        control_logits,
        candidate_logits,
        rtol=0.0,
        atol=1.0e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diperlukan")
def test_luminance_details_runs_under_strict_deterministic_cuda() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        raw = torch.rand(2, 3, 224, 224, device="cuda")
        details = luminance_l1_visushrink_details(raw)
        assert details.is_cuda
        assert details.shape == (2, 3, 112, 112)
        assert torch.isfinite(details).all()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_deterministic_hbp_alignment_matches_adaptive_pool_on_exact_bins() -> None:
    from bilinear_lmmd.modeling.models import HierarchicalBilinearPooling

    x = torch.arange(2 * 3 * 56 * 56, dtype=torch.float32).reshape(2, 3, 56, 56)
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        observed = HierarchicalBilinearPooling._align_to_target(x, (7, 7))
    finally:
        torch.use_deterministic_algorithms(previous)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diperlukan")
def test_wr_hbp_full_backward_runs_under_strict_deterministic_cuda() -> None:
    cfg = _cfg()
    previous = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = build_candidate(cfg).cuda().train()

        raw = torch.rand(2, 3, 224, 224, device="cuda")
        normalized = imagenet_normalize(raw)
        labels = torch.tensor([0, 1], device="cuda")

        output = model(normalized, raw_rgb=raw, labels=labels)
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
        loss.backward()

        assert model.classifier.weight.grad is not None
        assert torch.isfinite(model.classifier.weight.grad).all()
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
