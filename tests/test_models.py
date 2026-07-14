import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.models import AdaptationModel


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
