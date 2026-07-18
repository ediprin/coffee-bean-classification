from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("timm")

from bilinear_lmmd.modeling.models import (
    AdaptationModel,
    HierarchicalBilinearPooling,
    LinearProjectionHBP,
    ProjectedHierarchicalGAP,
)


def test_linear_hbp_matches_hand_computed_oracle() -> None:
    pool = LinearProjectionHBP([1, 1, 1], projection_dim=2).eval()
    weights = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([2.0, 0.5]),
    )
    with torch.no_grad():
        for projection, weight in zip(pool.projections, weights):
            projection.weight.copy_(weight.reshape(2, 1, 1, 1))
            projection.bias.zero_()

    features = [
        torch.tensor([[[[1.0, -2.0]]]]),
        torch.tensor([[[[3.0, 4.0]]]]),
        torch.tensor([[[[-1.0, 2.0]]]]),
    ]
    actual = pool(features)

    raw_pairs = torch.tensor(
        [[[-2.5, 5.0], [-5.0, -2.5], [5.0, -1.25]]]
    )
    transformed = torch.sign(raw_pairs) * torch.sqrt(
        torch.abs(raw_pairs) + 1e-8
    )
    expected = F.normalize(transformed, p=2, dim=2).reshape(1, 6)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual.reshape(1, 3, 2).norm(dim=2),
        torch.ones(1, 3),
    )


def test_linear_hbp_preserves_negative_interactions() -> None:
    pool = LinearProjectionHBP([1, 1, 1], projection_dim=1).eval()
    with torch.no_grad():
        for projection, weight in zip(pool.projections, (1.0, -1.0, 1.0)):
            projection.weight.fill_(weight)
            projection.bias.zero_()

    features = [torch.ones(1, 1, 2, 2) for _ in range(3)]
    actual = pool(features)

    torch.testing.assert_close(actual, torch.tensor([[-1.0, 1.0, -1.0]]))
    assert int((actual < 0).sum()) == 2


def test_linear_hbp_projections_have_no_norm_or_activation() -> None:
    pool = LinearProjectionHBP([3, 5, 7], projection_dim=11)

    assert len(pool.projections) == 3
    assert all(isinstance(module, nn.Conv2d) for module in pool.projections)
    assert not any(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU, nn.SiLU, nn.GELU))
        for module in pool.modules()
    )
    for projection in pool.projections:
        assert projection.kernel_size == (1, 1)
        assert projection.bias is not None
        torch.testing.assert_close(
            projection.bias,
            torch.zeros_like(projection.bias),
        )


def test_linear_hbp_keeps_legacy_same_seed_initialization_controlled() -> None:
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (1, 3, 4),
        "projection_dim": 32,
        "pretrained": False,
    }
    torch.manual_seed(2026)
    legacy = AdaptationModel(head="hbp", **kwargs)
    torch.manual_seed(2026)
    linear = AdaptationModel(head="hbp_linear", **kwargs)

    assert isinstance(legacy.pool, HierarchicalBilinearPooling)
    assert isinstance(linear.pool, LinearProjectionHBP)
    for legacy_parameter, linear_parameter in zip(
        legacy.encoder.parameters(), linear.encoder.parameters()
    ):
        assert torch.equal(legacy_parameter, linear_parameter)
    for legacy_projection, linear_projection in zip(
        legacy.pool.projections, linear.pool.projections
    ):
        assert torch.equal(legacy_projection[0].weight, linear_projection.weight)
    assert torch.equal(legacy.classifier.weight, linear.classifier.weight)
    assert torch.equal(legacy.classifier.bias, linear.classifier.bias)


def test_legacy_hbp_state_dict_still_loads_strictly() -> None:
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "head": "hbp",
        "out_indices": (1, 3, 4),
        "projection_dim": 16,
        "pretrained": False,
    }
    original = AdaptationModel(**kwargs)
    restored = AdaptationModel(**kwargs)

    result = restored.load_state_dict(original.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert "pool.projections.0.0.weight" in original.state_dict()
    assert "pool.projections.0.1.running_mean" in original.state_dict()


def test_linear_hbp_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="tepat tiga"):
        LinearProjectionHBP([3, 5], projection_dim=4)
    with pytest.raises(ValueError, match="lebih besar dari nol"):
        LinearProjectionHBP([3, 5, 7], projection_dim=0)


def test_projected_hierarchical_gap_is_capacity_matched_to_hbp() -> None:
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (1, 3, 4),
        "projection_dim": 32,
        "pretrained": False,
    }
    torch.manual_seed(42)
    hbp = AdaptationModel(head="hbp", **kwargs)
    torch.manual_seed(42)
    first_order = AdaptationModel(head="hierarchical_gap", **kwargs)

    assert isinstance(hbp.pool, HierarchicalBilinearPooling)
    assert isinstance(first_order.pool, ProjectedHierarchicalGAP)
    assert hbp.pool.output_dim == first_order.pool.output_dim == 96
    assert sum(p.numel() for p in hbp.parameters()) == sum(
        p.numel() for p in first_order.parameters()
    )
    for hbp_projection, first_order_projection in zip(
        hbp.pool.projections, first_order.pool.projections
    ):
        for hbp_parameter, first_order_parameter in zip(
            hbp_projection.parameters(), first_order_projection.parameters()
        ):
            assert torch.equal(hbp_parameter, first_order_parameter)
    assert torch.equal(hbp.classifier.weight, first_order.classifier.weight)
    assert torch.equal(hbp.classifier.bias, first_order.classifier.bias)


def test_projected_hierarchical_gap_forward_and_validation() -> None:
    pool = ProjectedHierarchicalGAP([3, 5, 7], projection_dim=11).eval()
    features = [
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 5, 8, 8),
        torch.randn(2, 7, 4, 4),
    ]

    output = pool(features)

    assert output.shape == (2, 33)
    torch.testing.assert_close(
        output.reshape(2, 3, 11).norm(dim=2),
        torch.ones(2, 3),
    )
    with pytest.raises(ValueError, match="3 feature map"):
        pool(features[:2])
    with pytest.raises(ValueError, match="tepat tiga"):
        ProjectedHierarchicalGAP([3, 5], projection_dim=11)
