import torch

from bilinear_lmmd.losses import LMMDLoss, MMDLoss


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
