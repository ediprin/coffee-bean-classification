from __future__ import annotations

import torch
from torch import nn

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.multistage_recalibration import (
    MultistageRecalibrationModel,
)
from bilinear_lmmd.modeling.models import build_model


class _FeatureInfo:
    @staticmethod
    def channels() -> list[int]:
        return [8, 12, 16]


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = _FeatureInfo()
        self.shallow = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.middle = nn.Conv2d(8, 12, 3, stride=2, padding=1)
        self.deep = nn.Conv2d(12, 16, 3, stride=2, padding=1)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        shallow = self.shallow(images)
        middle = self.middle(shallow)
        deep = self.deep(middle)
        return [shallow, middle, deep]


def _model(monkeypatch, mode: str) -> MultistageRecalibrationModel:
    monkeypatch.setattr(
        "bilinear_lmmd.modeling.multistage_recalibration.timm.create_model",
        lambda *args, **kwargs: _Encoder(),
    )
    return MultistageRecalibrationModel(
        backbone="fake",
        num_classes=5,
        out_indices=(1, 3, 4),
        fusion_dim=10,
        target_stage=1,
        gate_hidden_dim=7,
        dropout=0.0,
        pretrained=False,
        mode=mode,
    )


def test_fixed_fusion_preserves_spatial_map_and_backpropagates(monkeypatch) -> None:
    model = _model(monkeypatch, "fixed")
    images = torch.randn(2, 3, 32, 32)

    aligned = model.aligned_features(images)
    output = model(images)

    assert aligned.shape == (2, 3, 10, 8, 8)
    assert output.embedding.shape == (2, 10)
    assert output.logits.shape == (2, 5)
    torch.testing.assert_close(
        output.gate_weights,
        torch.full_like(output.gate_weights, 1.0 / 3.0),
    )

    output.logits.square().mean().backward()
    for projection in model.projections:
        assert projection.layers[0].weight.grad is not None


def test_adaptive_weights_are_normalized_and_gate_receives_gradient(monkeypatch) -> None:
    model = _model(monkeypatch, "adaptive")
    output = model(torch.randn(3, 3, 32, 32))

    assert output.gate_weights.shape == (3, 3, 10)
    assert torch.isfinite(output.gate_weights).all()
    assert (output.gate_weights >= 0).all()
    torch.testing.assert_close(
        output.gate_weights.sum(dim=1),
        torch.ones(3, 10),
    )

    output.logits.square().mean().backward()
    assert model.gate is not None
    assert model.gate.network[-1].weight.grad is not None
    assert model.gate.network[-1].weight.grad.abs().sum() > 0


def test_channel_control_is_uniform_over_stages_and_receives_gradient(
    monkeypatch,
) -> None:
    model = _model(monkeypatch, "channel_control")
    output = model(torch.randn(3, 3, 32, 32))

    assert output.gate_weights.shape == (3, 3, 10)
    assert torch.isfinite(output.gate_weights).all()
    torch.testing.assert_close(
        output.gate_weights[:, 0],
        output.gate_weights[:, 1],
    )
    torch.testing.assert_close(
        output.gate_weights[:, 1],
        output.gate_weights[:, 2],
    )

    output.logits.square().mean().backward()
    assert model.gate is not None
    assert model.gate.network[-1].weight.grad is not None
    assert model.gate.network[-1].weight.grad.abs().sum() > 0


def test_adaptive_starts_as_exact_fixed_fusion(monkeypatch) -> None:
    torch.manual_seed(2026)
    fixed = _model(monkeypatch, "fixed").eval()
    torch.manual_seed(2026)
    adaptive = _model(monkeypatch, "adaptive").eval()

    fixed_state = fixed.state_dict()
    adaptive_state = adaptive.state_dict()
    for name, parameter in fixed_state.items():
        torch.testing.assert_close(parameter, adaptive_state[name])

    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        fixed_output = fixed(images)
        adaptive_output = adaptive(images)
    torch.testing.assert_close(fixed_output.embedding, adaptive_output.embedding)
    torch.testing.assert_close(fixed_output.logits, adaptive_output.logits)


def test_channel_control_matches_adaptive_capacity_and_fixed_initialization(
    monkeypatch,
) -> None:
    torch.manual_seed(42)
    fixed = _model(monkeypatch, "fixed").eval()
    torch.manual_seed(42)
    control = _model(monkeypatch, "channel_control").eval()
    torch.manual_seed(42)
    adaptive = _model(monkeypatch, "adaptive").eval()

    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in adaptive.parameters()
    )
    for name, parameter in fixed.state_dict().items():
        torch.testing.assert_close(parameter, control.state_dict()[name])

    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        fixed_output = fixed(images)
        control_output = control(images)
        adaptive_output = adaptive(images)
    torch.testing.assert_close(fixed_output.logits, control_output.logits)
    torch.testing.assert_close(fixed_output.logits, adaptive_output.logits)
    torch.testing.assert_close(
        control_output.gate_weights,
        torch.full_like(control_output.gate_weights, 1.0 / 3.0),
    )


def test_adaptive_gate_can_learn_sample_specific_preferences(monkeypatch) -> None:
    model = _model(monkeypatch, "adaptive").eval()
    assert model.gate is not None
    with torch.no_grad():
        nn.init.normal_(model.gate.network[-1].weight, std=0.2)

    output = model(
        torch.stack(
            (
                torch.zeros(3, 32, 32),
                torch.ones(3, 32, 32),
            )
        )
    )
    assert not torch.allclose(
        output.gate_weights[0],
        output.gate_weights[1],
    )


def test_configs_are_a_controlled_fixed_vs_adaptive_ablation() -> None:
    fixed = load_config(
        "configs/finegrained/MSF0_efficientnetv2_fixed_multistage.yaml"
    )
    adaptive = load_config(
        "configs/finegrained/MSF1_efficientnetv2_adaptive_multistage.yaml"
    )
    control = load_config(
        "configs/finegrained/MSFC_efficientnetv2_channel_control.yaml"
    )

    assert fixed["model"]["head"] == "multistage_fixed"
    assert control["model"]["head"] == "multistage_channel_control"
    assert adaptive["model"]["head"] == "multistage_adaptive"
    for key in (
        "backbone",
        "pretrained",
        "out_indices",
        "multistage_fusion_dim",
        "multistage_target_stage",
        "multistage_gate_hidden",
        "classifier",
        "dropout",
    ):
        assert fixed["model"][key] == adaptive["model"][key]
        assert control["model"][key] == adaptive["model"][key]
    assert fixed["data"] == adaptive["data"]
    assert control["data"] == adaptive["data"]
    assert fixed["adaptation"] == adaptive["adaptation"]
    assert control["adaptation"] == adaptive["adaptation"]
    for key, value in fixed["training"].items():
        if key != "output_dir":
            assert adaptive["training"][key] == value
            assert control["training"][key] == value


def test_build_model_registry_constructs_all_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        "bilinear_lmmd.modeling.multistage_recalibration.timm.create_model",
        lambda *args, **kwargs: _Encoder(),
    )
    for path, mode in (
        (
            "configs/finegrained/MSF0_efficientnetv2_fixed_multistage.yaml",
            "fixed",
        ),
        (
            "configs/finegrained/MSF1_efficientnetv2_adaptive_multistage.yaml",
            "adaptive",
        ),
        (
            "configs/finegrained/MSFC_efficientnetv2_channel_control.yaml",
            "channel_control",
        ),
    ):
        cfg = load_config(path)
        cfg["model"]["pretrained"] = False
        cfg["model"]["multistage_fusion_dim"] = 10
        cfg["model"]["multistage_gate_hidden"] = 7
        model = build_model(cfg["model"])
        assert isinstance(model, MultistageRecalibrationModel)
        assert model.mode == mode
        assert model(torch.randn(2, 3, 32, 32)).logits.shape == (2, 17)
