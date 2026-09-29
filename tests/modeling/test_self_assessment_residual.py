from __future__ import annotations

import copy

import pytest
import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.engine.wavelet_residual_hbp import build_candidate
from bilinear_lmmd.engine.wr_hbp_self_assessment import (
    build_self_assessment,
    validate_config,
    zero_residual_preflight,
)
from bilinear_lmmd.modeling.self_assessment_residual import (
    TopKSelfAssessmentResidual,
)


def _cfg() -> dict:
    cfg = copy.deepcopy(
        load_config("configs/wr_hbp_self_assessment/WR_HBP_SAR_V2.yaml")
    )
    cfg["model"]["pretrained"] = False
    return cfg


def _base_checkpoint(tmp_path):
    cfg = _cfg()
    torch.manual_seed(42)
    base = build_candidate(cfg)
    path = tmp_path / "base.pt"
    torch.save({"model": base.state_dict(), "epoch": 1}, path)
    return cfg, path


def test_zero_residual_matches_base_exactly(tmp_path) -> None:
    cfg, path = _base_checkpoint(tmp_path)
    validate_config(cfg)

    base = build_candidate(cfg).eval()
    base.load_state_dict(torch.load(path, weights_only=False)["model"])
    candidate = build_self_assessment(cfg, base_checkpoint=path).eval()

    raw = torch.rand(3, 3, 224, 224)
    normalized = imagenet_normalize(raw)
    with torch.no_grad():
        expected = base(normalized, raw_rgb=raw).logits
        output = candidate(normalized, raw_rgb=raw)

    torch.testing.assert_close(output.logits, expected, rtol=0.0, atol=1e-7)
    assert output.expert_logits is not None
    assert output.expert_logits["residual"].abs().max().item() == 0.0
    assert all(not p.requires_grad for p in candidate.base.parameters())


def test_residual_changes_only_current_topk(tmp_path) -> None:
    cfg, path = _base_checkpoint(tmp_path)
    candidate = build_self_assessment(cfg, base_checkpoint=path).eval()

    # Make the residual head emit a constant non-zero residual.
    with torch.no_grad():
        candidate.joint[-1].bias.fill_(0.25)

    raw = torch.rand(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)
    with torch.no_grad():
        output = candidate(normalized, raw_rgb=raw)

    base_logits = output.expert_logits["base"]
    residual = output.expert_logits["residual"]
    topk = torch.topk(base_logits, k=5, dim=1).indices

    changed = residual.ne(0.0)
    expected = torch.zeros_like(changed)
    expected.scatter_(1, topk, True)
    assert torch.equal(changed, expected)


def test_backward_does_not_touch_frozen_base(tmp_path) -> None:
    cfg, path = _base_checkpoint(tmp_path)
    candidate = build_self_assessment(cfg, base_checkpoint=path).train()

    # Depart from exact-zero final layer so upstream reassessment receives grads.
    with torch.no_grad():
        candidate.joint[-1].weight.normal_(0.0, 1.0e-3)

    raw = torch.rand(2, 3, 224, 224)
    labels = torch.tensor([0, 1], dtype=torch.long)
    normalized = imagenet_normalize(raw)
    output = candidate(normalized, raw_rgb=raw)
    loss = torch.nn.functional.cross_entropy(output.logits, labels)
    loss.backward()

    assert all(parameter.grad is None for parameter in candidate.base.parameters())
    grads = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if not name.startswith("base.") and parameter.requires_grad
    ]
    assert any(grad is not None for grad in grads)


def test_zero_residual_preflight(tmp_path) -> None:
    cfg, path = _base_checkpoint(tmp_path)
    result = zero_residual_preflight(cfg, base_checkpoint=path)
    assert result["base_frozen"] is True
    assert result["zero_residual_max_abs"] == 0.0
    assert result["zero_residual_initial_logit_max_abs_difference"] <= 1e-7
    assert result["trainable_parameter_count"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diperlukan")
def test_strict_deterministic_cuda_backward(tmp_path) -> None:
    cfg, path = _base_checkpoint(tmp_path)
    previous = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        model = build_self_assessment(cfg, base_checkpoint=path).cuda().train()
        with torch.no_grad():
            model.joint[-1].weight.normal_(0.0, 1.0e-3)

        raw = torch.rand(2, 3, 224, 224, device="cuda")
        labels = torch.tensor([0, 1], device="cuda")
        normalized = imagenet_normalize(raw)
        output = model(normalized, raw_rgb=raw)
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
        loss.backward()

        assert model.joint[-1].weight.grad is not None
        assert torch.isfinite(model.joint[-1].weight.grad).all()
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
