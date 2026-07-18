import pytest
import torch
from torch import nn
from types import SimpleNamespace

from bilinear_lmmd.modeling.hierarchy import build_parent_hierarchy
from bilinear_lmmd.engine.train import supervised_objective


def test_parent_hierarchy_builds_mapping_in_dataset_order():
    hierarchy = build_parent_hierarchy(
        ["Fine B", "Fine A", "Fine C"],
        {"paired": ["Fine A", "Fine B"], "single": ["Fine C"]},
    )

    assert hierarchy.parent_names == ("paired", "single")
    assert hierarchy.fine_to_parent == (0, 0, 1)


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({"one": ["A"]}, "belum memiliki parent"),
        ({"one": ["A"], "two": ["A", "B"]}, "lebih dari sekali"),
        ({"one": ["A"], "two": ["B", "Ghost"]}, "tidak ditemukan"),
    ],
)
def test_parent_hierarchy_rejects_invalid_partition(groups, message):
    with pytest.raises(ValueError, match=message):
        build_parent_hierarchy(["A", "B"], groups)


def test_supervised_objective_adds_weighted_parent_ce_and_backpropagates():
    fine_logits = torch.randn(3, 4, requires_grad=True)
    parent_logits = torch.randn(3, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 3])
    parent_mapping = torch.tensor([0, 0, 1, 1])
    output = SimpleNamespace(
        logits=fine_logits,
        parent_logits=parent_logits,
        expert_logits=None,
    )
    criterion = nn.CrossEntropyLoss()

    total, components = supervised_objective(
        output,
        labels,
        criterion,
        nn.Identity(),
        auxiliary_weight=0.0,
        diversity_weight=0.0,
        parent_mapping=parent_mapping,
        hierarchy_weight=0.2,
    )
    expected = criterion(fine_logits, labels) + 0.2 * criterion(
        parent_logits, parent_mapping[labels]
    )

    assert torch.allclose(total, expected)
    assert set(components) == {"fused_ce", "parent_ce"}
    total.backward()
    assert fine_logits.grad is not None
    assert parent_logits.grad is not None
