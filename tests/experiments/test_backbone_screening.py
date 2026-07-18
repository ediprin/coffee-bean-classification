from __future__ import annotations

import pytest
import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.experiments.run_backbone_screening import BACKBONES, _selected_models


EXPECTED = {
    "MV3": ("mobilenetv3_large_100", [1, 3, 4]),
    "MV4": ("mobilenetv4_conv_medium.e500_r224_in1k", [1, 3, 4]),
    "EV2": ("tf_efficientnetv2_b0.in1k", [1, 3, 4]),
    "CV2": ("convnextv2_atto.fcmae_ft_in1k", [0, 2, 3]),
    "PV2": ("pvt_v2_b0.in1k", [0, 2, 3]),
    "SHV": ("shvit_s1.in1k", [0, 1, 2]),
}


@pytest.mark.parametrize("family", tuple(EXPECTED))
def test_backbone_pair_is_controlled(family: str) -> None:
    expected_backbone, expected_hbp_indices = EXPECTED[family]
    gap = load_config(BACKBONES[family]["gap"][1])
    hbp = load_config(BACKBONES[family]["hbp"][1])
    linear = load_config(BACKBONES[family]["hbp_linear"][1])

    assert gap["model"]["backbone"] == expected_backbone
    assert hbp["model"]["backbone"] == expected_backbone
    assert linear["model"]["backbone"] == expected_backbone
    assert gap["model"]["head"] == "gap"
    assert hbp["model"]["head"] == "hbp"
    assert linear["model"]["head"] == "hbp_linear"
    assert hbp["model"]["out_indices"] == expected_hbp_indices
    assert linear["model"]["out_indices"] == expected_hbp_indices
    assert len(gap["model"]["out_indices"]) == 1
    assert gap["model"]["pretrained"] is True
    assert hbp["model"]["pretrained"] is True
    assert linear["model"]["pretrained"] is True
    assert gap["data"]["image_size"] == hbp["data"]["image_size"] == 224
    assert gap["adaptation"]["method"] == hbp["adaptation"]["method"] == "source_only"
    assert linear["adaptation"]["method"] == "source_only"
    assert gap["training"]["classification_loss"] == "cross_entropy"
    assert hbp["training"]["classification_loss"] == "cross_entropy"
    assert linear["training"]["classification_loss"] == "cross_entropy"

    legacy_model = dict(hbp["model"])
    linear_model = dict(linear["model"])
    legacy_model.pop("head")
    linear_model.pop("head")
    assert legacy_model == linear_model
    assert hbp["data"] == linear["data"]
    assert hbp["adaptation"] == linear["adaptation"]
    legacy_training = dict(hbp["training"])
    linear_training = dict(linear["training"])
    legacy_training.pop("output_dir")
    linear_training.pop("output_dir")
    assert legacy_training == linear_training


def test_backbone_selection_keeps_family_then_head_order() -> None:
    selected = _selected_models(["PV2", "SHV"], ["gap", "hbp"])
    assert [(family, head, code) for family, head, code, _ in selected] == [
        ("PV2", "gap", "BP2G"),
        ("PV2", "hbp", "BP2H"),
        ("SHV", "gap", "BSHG"),
        ("SHV", "hbp", "BSHH"),
    ]


@pytest.mark.parametrize("family", tuple(EXPECTED))
@pytest.mark.parametrize("head", ("gap", "hbp", "hbp_linear"))
def test_backbone_model_forward_shape(family: str, head: str) -> None:
    config = load_config(BACKBONES[family][head][1])
    config["model"]["pretrained"] = False
    model = build_model(config["model"]).eval()

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))

    assert output.logits.shape == (1, 17)
    expected_embedding = (
        1536 if head in {"hbp", "hbp_linear"} else model.pool.output_dim
    )
    assert output.embedding.shape == (1, expected_embedding)
