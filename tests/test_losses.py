import torch

from bilinear_lmmd.losses import LMMDLoss, MMDLoss, NonTargetExpertDiversityLoss


def test_mmd_is_finite_and_near_zero_for_identical_features():
    features = torch.randn(6, 12)
    loss = MMDLoss()(features, features.clone())
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-5


def test_lmmd_backpropagates_to_features():
    source = torch.randn(6, 12, requires_grad=True)
    target = torch.randn(6, 12, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.randn(6, 3)
    loss = LMMDLoss(num_classes=3)(source, target, labels, logits)
    loss.backward()
    assert torch.isfinite(loss)
    assert source.grad is not None
    assert target.grad is not None


def test_non_target_diversity_is_lower_for_complementary_experts():
    identical = torch.tensor([[4.0, 2.0, 0.0]])
    complementary = torch.tensor([[4.0, 0.0, 2.0]])
    criterion = NonTargetExpertDiversityLoss()

    same_loss = criterion(identical, identical)
    complementary_loss = criterion(identical, complementary)
    assert torch.allclose(same_loss, torch.tensor(1.0))
    assert complementary_loss < same_loss


def test_non_target_diversity_backpropagates_to_both_experts():
    first = torch.randn(3, 4, requires_grad=True)
    second = torch.randn(3, 4, requires_grad=True)
    loss = NonTargetExpertDiversityLoss()(first, second)
    loss.backward()

    assert torch.isfinite(loss)
    assert first.grad is not None
    assert second.grad is not None
