from __future__ import annotations

import torch
from torch import nn

from bilinear_lmmd.modeling.progressive_multigranularity import (
    ProgressiveMultiGranularityModel,
    jigsaw_generator,
)


class _FeatureInfo:
    @staticmethod
    def channels() -> list[int]:
        return [8, 12, 16]


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = _FeatureInfo()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, 8, 3, padding=1),
                nn.Conv2d(8, 12, 3, stride=2, padding=1),
                nn.Conv2d(12, 16, 3, stride=2, padding=1),
            ]
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        first = self.layers[0](images)
        second = self.layers[1](first)
        third = self.layers[2](second)
        return [first, second, third]


def _model(monkeypatch, consistency: bool = False):
    monkeypatch.setattr(
        "bilinear_lmmd.modeling.progressive_multigranularity.timm.create_model",
        lambda *args, **kwargs: _Encoder(),
    )
    return ProgressiveMultiGranularityModel(
        backbone="fake",
        num_classes=5,
        out_indices=(1, 3, 4),
        feature_dim=8,
        branch_dim=16,
        dropout=0.0,
        pretrained=False,
        category_consistency=consistency,
    )


def test_progressive_model_returns_three_branches_and_combined_logits(monkeypatch):
    model = _model(monkeypatch).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))
    assert output.logits.shape == (2, 5)
    assert output.embedding.shape == (2, 48)
    assert tuple(output.expert_logits) == ("fine", "medium", "coarse", "concat")
    expected = sum(output.expert_logits.values())
    torch.testing.assert_close(output.logits, expected)


def test_progressive_branch_descriptor_is_finite_and_normalized(monkeypatch):
    model = _model(monkeypatch, consistency=True).eval()
    with torch.no_grad():
        output = model.forward_branch(torch.randn(4, 3, 32, 32), 1)
    assert output.logits.shape == (4, 5)
    assert torch.isfinite(output.descriptor).all()
    assert output.descriptor.min() >= 0
    assert output.descriptor.max() <= 1


def test_jigsaw_preserves_patch_multiset():
    patches = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    image = patches.repeat_interleave(2, 2).repeat_interleave(2, 3)
    shuffled = jigsaw_generator(image, 4)
    observed = shuffled.reshape(1, 1, 4, 2, 4, 2)[:, :, :, 0, :, 0].flatten()
    assert sorted(observed.tolist()) == list(range(16))


def test_jigsaw_rejects_incompatible_grid():
    with torch.no_grad():
        try:
            jigsaw_generator(torch.randn(2, 3, 15, 16), 4)
        except ValueError as exc:
            assert "tidak habis dibagi" in str(exc)
        else:
            raise AssertionError("Grid yang tidak kompatibel harus ditolak.")
