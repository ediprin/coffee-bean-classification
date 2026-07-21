from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import (
    AdaptationModel,
    FactorizedBilinearConvClassifier,
    build_model,
)


def test_fb_conv_matches_factorized_quadratic_equation_at_inference() -> None:
    layer = FactorizedBilinearConvClassifier(
        channels=2,
        num_classes=1,
        rank=2,
        keep_prob=0.5,
        quadratic=True,
    )
    layer.eval()
    with torch.no_grad():
        layer.linear.weight.copy_(torch.tensor([[[[2.0]], [[-1.0]]]]))
        layer.linear.bias.fill_(0.3)
        layer.factors.copy_(torch.tensor([[[1.0, 2.0], [-1.0, 0.5]]]))

    feature = torch.tensor([[[[1.0]], [[2.0]]]])
    bounded = torch.tanh(feature.flatten())
    linear = 0.3 + 2.0 * bounded[0] - bounded[1]
    projected = layer.factors[0] @ bounded
    expected = linear + 0.5 * projected.square().sum()

    actual = layer(feature)
    assert actual.shape == (1, 1)
    assert torch.allclose(actual.squeeze(), expected, atol=1.0e-6)


def test_fb_linear_control_is_capacity_matched_and_removes_square() -> None:
    quadratic = FactorizedBilinearConvClassifier(8, 4, rank=3, quadratic=True)
    control = FactorizedBilinearConvClassifier(8, 4, rank=3, quadratic=False)
    assert sum(p.numel() for p in quadratic.parameters()) == sum(
        p.numel() for p in control.parameters()
    )

    control.train()
    output = control(torch.randn(2, 8, 5, 5))
    output.sum().backward()
    assert control.factors.grad is not None
    assert control.linear.weight.grad is not None


@pytest.mark.parametrize(
    ("head", "quadratic"),
    [
        ("factorized_linear_conv_control", False),
        ("factorized_bilinear_conv", True),
    ],
)
def test_adaptation_model_runs_direct_fb_classifier(head: str, quadratic: bool) -> None:
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head=head,
        out_indices=(4,),
        fb_rank=3,
        pretrained=False,
    )
    output = model(torch.randn(2, 3, 96, 96))
    assert output.logits.shape == (2, 4)
    assert output.embedding.ndim == 2
    assert model.direct_classifier is not None
    assert model.direct_classifier.quadratic is quadratic
    assert tuple(model.slow_start_parameters())


def test_fb0_fb1_configs_are_controlled_and_capacity_matched() -> None:
    control_cfg = load_config(
        "configs/finegrained/FB0_efficientnetv2_factorized_linear_control.yaml"
    )
    quadratic_cfg = load_config(
        "configs/finegrained/FB1_efficientnetv2_factorized_bilinear_conv.yaml"
    )
    for cfg in (control_cfg, quadratic_cfg):
        cfg["model"]["pretrained"] = False
        assert cfg["model"]["fb_rank"] == 20
        assert cfg["model"]["fb_dropfactor_keep_prob"] == 0.5
        assert cfg["training"]["slow_start_lr_epochs"] == 3
        assert cfg["adaptation"]["method"] == "source_only"

    control = build_model(control_cfg["model"])
    quadratic = build_model(quadratic_cfg["model"])
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in quadratic.parameters()
    )
