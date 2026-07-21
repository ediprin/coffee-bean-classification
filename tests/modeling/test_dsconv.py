from __future__ import annotations

import torch
from torch import nn

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.dsconv import (
    DistributionShiftConv2d,
    replace_spatial_convolutions_with_dsconv,
)
from bilinear_lmmd.modeling.models import build_model


def test_dsconv_initializes_close_to_source_convolution_and_backpropagates() -> None:
    torch.manual_seed(7)
    source = nn.Conv2d(9, 6, kernel_size=3, padding=1, bias=True)
    candidate = DistributionShiftConv2d.from_conv2d(
        source, bits=4, block_size=4
    )
    inputs = torch.randn(3, 9, 12, 12)
    reference = source(inputs)
    output = candidate(inputs)

    assert output.shape == reference.shape
    assert torch.mean(torch.abs(output - reference)).item() < 0.03
    output.square().mean().backward()
    assert candidate.weight.grad is not None
    assert candidate.kds_scale.grad is not None
    assert candidate.cds_scale.grad is not None


def test_dsconv_theoretical_storage_is_below_fp32_kernel() -> None:
    module = DistributionShiftConv2d(
        128, 128, (3, 3), padding=(1, 1), bits=4, block_size=128
    )
    full_precision_bits = module.weight.numel() * 32
    assert module.theoretical_kernel_bits() < full_precision_bits


def test_dsconv_preserves_timm_dynamic_same_padding_shape() -> None:
    from timm.layers import Conv2dSame

    source = Conv2dSame(7, 11, kernel_size=3, stride=2, bias=False)
    candidate = DistributionShiftConv2d.from_conv2d(
        source, bits=4, block_size=8
    )
    for spatial_size in (15, 16):
        inputs = torch.randn(2, 7, spatial_size, spatial_size + 2)
        assert candidate(inputs).shape == source(inputs).shape


def test_replacement_is_stage_scoped_and_preserves_rng_stream() -> None:
    class ToyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(3, 8, 3, padding=1)),
                    nn.Sequential(nn.Conv2d(8, 8, 1)),
                    nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8)),
                ]
            )

    torch.manual_seed(31)
    backbone = ToyBackbone()
    expected_state = torch.random.get_rng_state().clone()
    replaced = replace_spatial_convolutions_with_dsconv(
        backbone,
        stage_prefixes=("blocks.0", "blocks.1", "blocks.2"),
        bits=4,
        block_size=8,
    )
    assert replaced == ["blocks.0.0"]
    assert isinstance(backbone.blocks[0][0], DistributionShiftConv2d)
    assert isinstance(backbone.blocks[1][0], nn.Conv2d)
    assert isinstance(backbone.blocks[2][0], nn.Conv2d)
    assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_hong_classification_configs_build_and_forward() -> None:
    paths = (
        "configs/finegrained/HCD1_efficientnetv2_dsconv_gap.yaml",
        "configs/finegrained/HCS1_efficientnetv2_sppf_attention_gap.yaml",
        "configs/finegrained/HCDS1_efficientnetv2_dsconv_sppf_attention_gap.yaml",
    )
    for path in paths:
        cfg = load_config(path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"]).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        assert output.logits.shape == (1, 17)
        expected_dsconv = "dsconv" in path and "sppf_attention_gap" not in path
        if path.endswith("dsconv_sppf_attention_gap.yaml"):
            expected_dsconv = True
        assert bool(model.dsconv_replaced_layers) is expected_dsconv


def test_dsconv_candidate_preserves_paired_backbone_and_classifier_initialization() -> None:
    baseline_cfg = load_config(
        "configs/backbones/BE2G_efficientnetv2_gap_source.yaml"
    )
    candidate_cfg = load_config(
        "configs/finegrained/HCD1_efficientnetv2_dsconv_gap.yaml"
    )
    baseline_cfg["model"]["pretrained"] = False
    candidate_cfg["model"]["pretrained"] = False

    torch.manual_seed(101)
    baseline = build_model(baseline_cfg["model"])
    torch.manual_seed(101)
    candidate = build_model(candidate_cfg["model"])

    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    baseline_state = baseline.encoder.state_dict()
    candidate_state = candidate.encoder.state_dict()
    for path in candidate.dsconv_replaced_layers:
        key = f"{path}.weight"
        assert torch.equal(baseline_state[key], candidate_state[key])
