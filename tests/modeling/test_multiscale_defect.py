from __future__ import annotations

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import (
    AdaptationModel,
    CapacityResidualGAP,
    MultiScaleDefectExtraction,
    MultiScaleDefectGAP,
    build_model,
)


def test_multiscale_defect_extraction_preserves_shape_and_backpropagates() -> None:
    module = MultiScaleDefectExtraction(channels=12, branch_channels=4)
    feature = torch.randn(2, 12, 9, 9, requires_grad=True)

    output = module(feature)
    assert output.shape == feature.shape
    output.square().mean().backward()
    assert feature.grad is not None
    assert module.branch_3x3[0].weight.grad is not None
    assert module.branch_5x5[0].weight.grad is not None
    assert module.fuse[0].weight.grad is not None


def test_mde_gap_preserves_paired_gap_core_initialization() -> None:
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (4,),
        "pretrained": False,
    }
    torch.manual_seed(71)
    baseline = AdaptationModel(head="gap", **kwargs)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(71)
    candidate = AdaptationModel(
        head="multiscale_defect_gap",
        mde_branch_channels=4,
        **kwargs,
    )

    assert isinstance(candidate.pool, MultiScaleDefectGAP)
    for baseline_parameter, candidate_parameter in zip(
        baseline.encoder.parameters(), candidate.encoder.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    assert torch.equal(baseline.classifier.bias, candidate.classifier.bias)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())


def test_mde_candidate_and_capacity_control_are_parameter_matched() -> None:
    control_cfg = load_config(
        "configs/finegrained/MDE0_efficientnetv2_capacity_control_gap.yaml"
    )
    candidate_cfg = load_config(
        "configs/finegrained/MDE1_efficientnetv2_multiscale_defect_gap.yaml"
    )
    for cfg in (control_cfg, candidate_cfg):
        cfg["model"]["pretrained"] = False

    control = build_model(control_cfg["model"])
    candidate = build_model(candidate_cfg["model"])
    assert isinstance(control.pool, CapacityResidualGAP)
    assert isinstance(candidate.pool, MultiScaleDefectGAP)

    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    assert abs(control_parameters - candidate_parameters) / candidate_parameters < 0.001


def test_mde_configs_change_only_the_controlled_spatial_operator() -> None:
    control = load_config(
        "configs/finegrained/MDE0_efficientnetv2_capacity_control_gap.yaml"
    )
    candidate = load_config(
        "configs/finegrained/MDE1_efficientnetv2_multiscale_defect_gap.yaml"
    )

    assert control["model"]["head"] == "capacity_residual_gap"
    assert candidate["model"]["head"] == "multiscale_defect_gap"
    for key in ("backbone", "pretrained", "out_indices", "classifier", "dropout"):
        assert control["model"][key] == candidate["model"][key]
    assert control["data"] == candidate["data"]
    assert control["adaptation"] == candidate["adaptation"]
    for key, value in control["training"].items():
        if key != "output_dir":
            assert candidate["training"][key] == value
