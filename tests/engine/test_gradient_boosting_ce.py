from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.engine.gradient_boosting_ce import GradientBoostingCrossEntropy
from bilinear_lmmd.engine.wr_hbp_gce import (
    preflight_matched_initialization,
    validate_config,
)


def test_gce_matches_manual_top2_without_smoothing() -> None:
    logits = torch.tensor([
        [4.0, 3.0, 1.0, -2.0],
        [0.0, 2.0, 3.0, 1.0],
    ])
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss = GradientBoostingCrossEntropy(
        num_classes=4,
        top_k=2,
        label_smoothing=0.0,
    )(logits, targets)

    restricted = torch.stack([
        torch.tensor([4.0, 3.0, 1.0]),
        torch.tensor([2.0, 3.0, 1.0]),
    ])
    expected = -F.log_softmax(restricted, dim=1)[:, 0].mean()
    torch.testing.assert_close(loss, expected)


def test_gce_all_negatives_equals_cross_entropy_with_same_smoothing() -> None:
    torch.manual_seed(7)
    logits = torch.randn(8, 17, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 16], dtype=torch.long)
    smoothing = 0.1

    observed = GradientBoostingCrossEntropy(
        num_classes=17,
        top_k=16,
        label_smoothing=smoothing,
    )(logits, targets)
    expected = F.cross_entropy(
        logits,
        targets,
        label_smoothing=smoothing,
    )
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_excluded_negative_receives_zero_gradient() -> None:
    logits = torch.tensor(
        [[5.0, 4.0, 3.0, -10.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0], dtype=torch.long)
    loss = GradientBoostingCrossEntropy(
        num_classes=4,
        top_k=2,
        label_smoothing=0.0,
    )(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 3].item() == 0.0
    assert logits.grad[0, 1].abs().item() > 0.0
    assert logits.grad[0, 2].abs().item() > 0.0


def test_wr_hbp_gce_frozen_config_and_initialization() -> None:
    cfg = copy.deepcopy(load_config("configs/wr_hbp_gce/WR_HBP_GCE_V1.yaml"))
    cfg["model"]["pretrained"] = False

    validate_config(cfg)
    preflight = preflight_matched_initialization(cfg)

    assert preflight["matched_model_tensor_equality"] is True
    assert preflight["initial_logit_max_abs_difference"] <= 1.0e-7
    assert len(preflight["initial_model_state_sha256"]) == 64
