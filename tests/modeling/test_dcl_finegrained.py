import torch

from bilinear_lmmd.modeling.dcl_finegrained import DCLFineGrainedModel
from bilinear_lmmd.modeling.models import AdaptationModel


def _dcl() -> DCLFineGrainedModel:
    return DCLFineGrainedModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        out_indices=(4,),
        grid_size=3,
        dropout=0.2,
        pretrained=False,
    )


def test_dcl_forward_and_training_heads_have_expected_shapes():
    model = _dcl()
    images = torch.randn(4, 3, 96, 96)

    inference = model(images)
    training = model.forward_dcl(images)

    assert inference.logits.shape == (4, 4)
    assert inference.embedding.shape == (4, model.output_dim)
    assert training.classification.logits.shape == (4, 4)
    assert training.swap_logits.shape == (4, 2)
    assert training.layout.shape == (4, 9)
    assert model.auxiliary_parameter_count() > 0
    assert model.inference_parameter_count() < sum(
        parameter.numel() for parameter in model.parameters()
    )

    (
        training.classification.logits.sum()
        + training.swap_logits.sum()
        + training.layout.sum()
    ).backward()
    assert model.classifier.weight.grad is not None
    assert model.swap_classifier.weight.grad is not None
    assert model.layout_head.weight.grad is not None


def test_dcl_auxiliary_heads_do_not_change_same_seed_core_initialization():
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (4,),
        "dropout": 0.2,
        "pretrained": False,
    }
    torch.manual_seed(53)
    baseline = AdaptationModel(head="gap", **kwargs)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(53)
    candidate = DCLFineGrainedModel(grid_size=3, **kwargs)

    for baseline_parameter, candidate_parameter in zip(
        baseline.encoder.parameters(),
        candidate.encoder.parameters(),
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    assert torch.equal(baseline.classifier.bias, candidate.classifier.bias)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())
