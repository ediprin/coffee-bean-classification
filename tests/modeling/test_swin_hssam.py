from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.losses import FusionCrossEntropyFocalLoss
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.modeling.swin_hssam import (
    HighLevelScreeningFPN,
    SelectiveAttentionModule,
    _to_nchw,
)


CONFIGS = {
    "SJ0": "configs/sni/jiao/SJ0_swin_tiny_gap_ce.yaml",
    "SJH": "configs/sni/jiao/SJH_swin_tiny_hsfpn_ce.yaml",
    "SJS": "configs/sni/jiao/SJS_swin_tiny_sam_ce.yaml",
    "SJL": "configs/sni/jiao/SJL_swin_tiny_gap_fusion_loss.yaml",
    "SJHL": "configs/sni/jiao/SJHL_swin_tiny_hsfpn_fusion_loss.yaml",
    "SJSL": "configs/sni/jiao/SJSL_swin_tiny_sam_fusion_loss.yaml",
    "SJHS": "configs/sni/jiao/SJHS_swin_tiny_hssam_ce.yaml",
    "SJFULL": "configs/sni/jiao/SJFULL_swin_hssam_fusion_loss.yaml",
}


def test_swin_feature_layout_is_converted_explicitly():
    nhwc = torch.randn(2, 8, 8, 12)
    nchw = _to_nchw(nhwc, 12)
    assert nchw.shape == (2, 12, 8, 8)
    torch.testing.assert_close(nchw, nhwc.permute(0, 3, 1, 2))
    with pytest.raises(ValueError):
        _to_nchw(torch.randn(2, 7, 7, 9), 12)


def test_hsfpn_uses_all_three_stages_and_backpropagates():
    module = HighLevelScreeningFPN([8, 16, 32], output_channels=6)
    features = [
        torch.randn(2, 16, 16, 8, requires_grad=True),
        torch.randn(2, 8, 8, 16, requires_grad=True),
        torch.randn(2, 4, 4, 32, requires_grad=True),
    ]
    output = module(features)
    assert output.shape == (2, 6, 16, 16)
    output.mean().backward()
    assert all(feature.grad is not None for feature in features)


def test_sam_preserves_feature_shape_and_multiplies_original_map():
    module = SelectiveAttentionModule(channels=8, hidden_dim=4)
    feature = torch.randn(2, 8, 7, 7, requires_grad=True)
    output = module(feature)
    assert output.shape == feature.shape
    output.square().mean().backward()
    assert feature.grad is not None
    assert module.expand_channels.weight.grad is not None


def test_fusion_loss_matches_manual_multiclass_definition():
    logits = torch.tensor([[2.0, -0.5], [-0.2, 1.1]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = FusionCrossEntropyFocalLoss(
        alpha=0.25,
        gamma=2.0,
        cross_entropy_weight=0.7,
        focal_weight=0.3,
    )(logits, labels)
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = 0.7 * ce.mean() + 0.3 * (
        0.25 * (1.0 - torch.exp(-ce)).square() * ce
    ).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_jiao_configs_form_complete_three_factor_factorial():
    observed = set()
    for code, path in CONFIGS.items():
        cfg = load_config(path)
        model = cfg["model"]
        assert model["backbone"] == "swin_tiny_patch4_window7_224.ms_in1k"
        assert model["num_classes"] == 21
        assert tuple(model["out_indices"]) == (1, 2, 3)
        assert cfg["adaptation"]["method"] == "source_only"
        hsfpn = model["head"] in {"swin_hsfpn", "swin_hssam"}
        sam = model["head"] in {"swin_sam", "swin_hssam"}
        fusion_loss = (
            cfg["training"]["classification_loss"] == "fusion_ce_focal"
        )
        observed.add((hsfpn, sam, fusion_loss))
        if fusion_loss:
            assert cfg["training"]["fusion_loss_ce_weight"] == 0.7
            assert cfg["training"]["fusion_loss_focal_weight"] == 0.3
    assert len(observed) == 8


def test_swin_hssam_builds_from_full_config():
    cfg = load_config(CONFIGS["SJFULL"])
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    assert model.use_hsfpn is True
    assert model.use_sam is True
    assert model.embedding_dim == 256
