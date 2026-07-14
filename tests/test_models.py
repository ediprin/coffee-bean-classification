import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.config import load_config
from bilinear_lmmd.models import AdaptationModel, build_model


@pytest.mark.parametrize("head", ["gap", "bilinear", "hbp"])
def test_mobilenetv3_output_shapes(head):
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head=head,
        out_indices=(1, 3, 4),
        projection_dim=32,
        pretrained=False,
        enable_domain_classifier=True,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 96, 96), domain_strength=1.0)
    assert output.logits.shape == (2, 4)
    assert output.embedding.ndim == 2
    assert output.domain_logits.shape == (2, 2)


def test_m0b_and_hbp_have_matched_embedding_classifier_and_capacity():
    bilinear_cfg = load_config("configs/M0b_mobilenetv3_bilinear_source.yaml")
    hbp_cfg = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    bilinear_cfg["model"]["pretrained"] = False
    hbp_cfg["model"]["pretrained"] = False
    bilinear = build_model(bilinear_cfg["model"])
    hbp = build_model(hbp_cfg["model"])

    assert bilinear.pool.output_dim == hbp.pool.output_dim == 1536
    assert bilinear.classifier.in_features == hbp.classifier.in_features == 1536
    bilinear_params = sum(parameter.numel() for parameter in bilinear.parameters())
    hbp_params = sum(parameter.numel() for parameter in hbp.parameters())
    assert abs(bilinear_params - hbp_params) / hbp_params < 0.01
