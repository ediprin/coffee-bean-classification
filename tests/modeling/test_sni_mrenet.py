from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.sni_ontology import (
    SNI_CLASSES,
    SNI_GROUPS,
    SNI_GROUP_SIZES,
    validate_sni_classes,
)
from bilinear_lmmd.modeling.models import (
    HierarchicalBilinearPooling,
    ProjectedHierarchicalGAP,
    build_model,
)


def _small_model(config_path: str):
    cfg = load_config(config_path)
    cfg["model"].update(
        {
            "backbone": "resnet18",
            "pretrained": False,
            "out_indices": [1, 2, 3, 4],
            "sni_feature_dim": 16,
            "projection_dim": 8,
        }
    )
    return build_model(cfg["model"])


def test_sni_ontology_is_contiguous_and_matches_imagefolder_order():
    assert len(SNI_CLASSES) == 21
    assert SNI_GROUP_SIZES == (12, 3, 3, 3)
    assert tuple(name for group in SNI_GROUPS.values() for name in group) == SNI_CLASSES
    assert tuple(sorted(SNI_CLASSES)) == SNI_CLASSES
    validate_sni_classes(list(SNI_CLASSES))
    with pytest.raises(ValueError):
        validate_sni_classes(list(reversed(SNI_CLASSES)))


def test_sni_ontology_models_emit_normalized_21_class_distribution():
    gap = _small_model("configs/sni/SNIB2_efficientnetv2_mre_gap.yaml").eval()
    hbp = _small_model("configs/sni/SNIB3_efficientnetv2_mrenet.yaml").eval()

    assert isinstance(gap.bean_pool, ProjectedHierarchicalGAP)
    assert isinstance(hbp.bean_pool, HierarchicalBilinearPooling)
    assert sum(parameter.numel() for parameter in gap.parameters()) == sum(
        parameter.numel() for parameter in hbp.parameters()
    )

    images = torch.randn(2, 3, 64, 64)
    for model in (gap, hbp):
        output = model(images)
        assert output.logits.shape == (2, 21)
        torch.testing.assert_close(
            output.logits.exp().sum(dim=1),
            torch.ones(2),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        loss = torch.nn.functional.cross_entropy(
            output.logits, torch.tensor([0, 20])
        )
        loss.backward()
        assert model.router.weight.grad is not None
        assert model.expert_classifiers[0].weight.grad is not None


def test_sni_configs_form_the_frozen_b0_b3_ablation():
    expected = {
        "SNIB0_efficientnetv2_gap.yaml": "gap",
        "SNIB1_efficientnetv2_multiresolution_flat.yaml": (
            "sni_multiresolution_flat"
        ),
        "SNIB2_efficientnetv2_mre_gap.yaml": "sni_mre_ontology_gap",
        "SNIB3_efficientnetv2_mrenet.yaml": "sni_mrenet",
    }
    for filename, head in expected.items():
        cfg = load_config(f"configs/sni/{filename}")
        assert cfg["model"]["backbone"] == "tf_efficientnetv2_b0.in1k"
        assert cfg["model"]["num_classes"] == 21
        assert cfg["model"]["head"] == head
        assert cfg["adaptation"]["method"] == "source_only"
        assert cfg["training"]["precision"] == "amp_fp16"
        assert cfg["training"]["channels_last"] is True
        assert cfg["training"]["non_blocking"] is True


def test_sni_second_order_and_control_statistics_stay_fp32_under_autocast():
    hbp = HierarchicalBilinearPooling([8, 8, 8], projection_dim=4).eval()
    gap = ProjectedHierarchicalGAP([8, 8, 8], projection_dim=4).eval()
    features = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 8, 4, 4),
    ]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        hbp_embedding = hbp(features)
        gap_embedding = gap(features)

    assert hbp_embedding.dtype == torch.float32
    assert gap_embedding.dtype == torch.float32
