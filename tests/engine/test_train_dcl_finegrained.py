import pytest
import torch
from torch import nn

from bilinear_lmmd.engine.train_dcl_finegrained import dcl_objective
from bilinear_lmmd.modeling.dcl_finegrained import DCLTrainingOutput
from bilinear_lmmd.modeling.models import ModelOutput


def _output() -> DCLTrainingOutput:
    embedding = torch.randn(4, 8, requires_grad=True)
    return DCLTrainingOutput(
        classification=ModelOutput(
            logits=torch.randn(4, 3, requires_grad=True),
            embedding=embedding,
        ),
        swap_logits=torch.randn(4, 2, requires_grad=True),
        layout=torch.randn(4, 9, requires_grad=True).tanh(),
    )


def test_dcl_objective_combines_paper_losses_and_backpropagates():
    output = _output()
    total, components = dcl_objective(
        output,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        torch.randn(4, 9),
        classification_loss=nn.CrossEntropyLoss(),
        swap_weight=1.0,
        layout_weight=1.0,
    )

    assert set(components) == {
        "classification_ce",
        "swap_ce",
        "layout_l1",
    }
    assert torch.allclose(total, sum(components.values()))
    total.backward()
    assert output.classification.logits.grad is not None
    assert output.swap_logits.grad is not None


def test_dcl_objective_rejects_fake_contrastive_configuration():
    with pytest.raises(ValueError, match="Projection"):
        dcl_objective(
            _output(),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
            torch.randn(4, 9),
            classification_loss=nn.CrossEntropyLoss(),
            swap_weight=1.0,
            layout_weight=1.0,
            contrastive_weight=0.2,
        )
