from __future__ import annotations

import torch

from bilinear_lmmd.engine.train_pairwise_contrastive import (
    ContrastiveProjectionHead,
    normalized_soft_confusion,
    update_confusion_ema,
)


def test_projection_head_returns_normalized_embedding():
    head = ContrastiveProjectionHead(8, 6, 4)
    result = head(torch.randn(5, 8))
    assert result.shape == (5, 4)
    torch.testing.assert_close(result.norm(dim=1), torch.ones(5))


def test_soft_confusion_is_symmetric_normalized_and_zero_diagonal():
    labels = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor(
        [
            [0.8, 0.2, 0.0],
            [0.6, 0.4, 0.0],
            [0.3, 0.7, 0.0],
            [0.1, 0.8, 0.1],
        ]
    )
    matrix = normalized_soft_confusion(labels, probabilities, 3)
    torch.testing.assert_close(matrix, matrix.t())
    torch.testing.assert_close(matrix.diag(), torch.zeros(3))
    assert matrix.max() == 1.0
    assert torch.all((matrix >= 0.0) & (matrix <= 1.0))


def test_confusion_ema_uses_current_matrix_when_uninitialized():
    current = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    updated = update_confusion_ema(torch.zeros_like(current), current, 0.9)
    torch.testing.assert_close(updated, current)
