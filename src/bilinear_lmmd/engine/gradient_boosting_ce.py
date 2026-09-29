from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GradientBoostingCrossEntropy(nn.Module):
    """Top-k negative cross entropy for fine-grained confusing classes.

    This implements the core GCE idea from Sun et al. (AAAI 2020): for each
    sample, keep the ground-truth logit and only the k highest-scoring negative
    logits in the normalization. Coffee17 keeps the existing label-smoothing
    contract by applying smoothing on that restricted support.

    When top_k == num_classes - 1, the loss is exactly standard cross entropy
    with the same label smoothing (up to floating-point ordering).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        top_k: int,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes harus >= 2.")
        if not 1 <= top_k <= num_classes - 1:
            raise ValueError(
                f"top_k harus 1..{num_classes - 1}, observed {top_k}."
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing harus di [0,1).")
        self.num_classes = int(num_classes)
        self.top_k = int(top_k)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("logits harus [batch, classes].")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Jumlah kelas logits {logits.shape[1]} != {self.num_classes}."
            )
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets harus [batch] dan cocok dengan logits.")
        if targets.dtype != torch.long:
            raise TypeError("targets harus torch.long.")

        true_logits = logits.gather(1, targets[:, None])

        negative_scores = logits.clone()
        negative_scores.scatter_(1, targets[:, None], float("-inf"))
        negative_indices = negative_scores.topk(
            self.top_k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        negative_logits = logits.gather(1, negative_indices)

        restricted_logits = torch.cat((true_logits, negative_logits), dim=1)
        log_probs = F.log_softmax(restricted_logits, dim=1)

        nll = -log_probs[:, 0]
        if self.label_smoothing == 0.0:
            return nll.mean()

        smooth = -log_probs.mean(dim=1)
        return (
            (1.0 - self.label_smoothing) * nll
            + self.label_smoothing * smooth
        ).mean()
