import torch
from torch.nn import functional as F

from bilinear_lmmd.modeling.losses import (
    OntologyMarginalLoss,
    TaxonomyCompatibleContrastiveLoss,
)


def test_singleton_marginal_loss_equals_cross_entropy() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.3, 1.2]], requires_grad=True)
    labels = torch.tensor([0, 2])
    compatibility = F.one_hot(labels, num_classes=3).bool()
    actual = OntologyMarginalLoss()(logits, compatibility)
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_coarse_marginal_loss_equals_summed_softmax_probability() -> None:
    logits = torch.tensor([[1.0, 2.0, -0.5]], requires_grad=True)
    compatibility = torch.tensor([[True, True, False]])
    actual = OntologyMarginalLoss()(logits, compatibility)
    expected = -torch.log(logits.softmax(dim=1)[0, :2].sum())
    assert torch.allclose(actual, expected, atol=1e-7)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_taxonomy_contrastive_ignores_overlapping_non_equal_sets() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]],
        requires_grad=True,
    )
    compatibility = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [True, True, False],  # parent-child overlap: ignored
            [False, False, True],  # definite negative
        ]
    )
    loss = TaxonomyCompatibleContrastiveLoss(temperature=0.2)(
        embeddings, compatibility
    )
    assert loss.isfinite() and loss.item() >= 0.0
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_taxonomy_contrastive_returns_differentiable_zero_without_positives() -> None:
    embeddings = torch.randn(3, 4, requires_grad=True)
    compatibility = torch.eye(3, dtype=torch.bool)
    loss = TaxonomyCompatibleContrastiveLoss()(embeddings, compatibility)
    assert loss.item() == 0.0
    loss.backward()
    assert embeddings.grad is not None
