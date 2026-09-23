import torch
from torch import nn

from bilinear_lmmd.engine.shared_multiview_sbn import (
    auxiliary_batchnorm_batch_stats,
)


def test_selective_bn_uses_batch_stats_without_mutating_running_buffers():
    bn = nn.BatchNorm2d(2, affine=True, track_running_stats=True)
    bn.train()
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([100.0, -100.0]))
        bn.running_var.copy_(torch.tensor([4.0, 9.0]))
        bn.weight.fill_(1.0)
        bn.bias.zero_()

    running_mean_before = bn.running_mean.clone()
    running_var_before = bn.running_var.clone()
    tracked_before = bn.num_batches_tracked.clone()

    x = torch.randn(8, 2, 4, 4) * 3.0 + torch.tensor([5.0, -7.0]).view(1, 2, 1, 1)
    with auxiliary_batchnorm_batch_stats(bn):
        y = bn(x)
        assert bn.training is True
        assert bn.track_running_stats is False

    assert bn.training is True
    assert bn.track_running_stats is True
    assert torch.equal(bn.running_mean, running_mean_before)
    assert torch.equal(bn.running_var, running_var_before)
    assert torch.equal(bn.num_batches_tracked, tracked_before)

    # If the stale running statistics had been used, these means would be huge.
    channel_means = y.mean(dim=(0, 2, 3))
    assert torch.allclose(channel_means, torch.zeros_like(channel_means), atol=1e-5)


def test_standard_raw_forward_still_updates_running_stats():
    bn = nn.BatchNorm2d(2)
    bn.train()
    before = bn.running_mean.clone()
    _ = bn(torch.randn(8, 2, 4, 4) + 3.0)
    assert not torch.equal(before, bn.running_mean)
