from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax for long-tailed single-label classification.

    Training logits are adjusted by the log class frequency. Raw model logits
    remain unchanged for validation and inference.
    """

    def __init__(self, class_counts: Tensor, label_smoothing: float = 0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() < 2:
            raise ValueError("class_counts harus berupa vektor dengan minimal dua kelas.")
        if torch.any(counts <= 0):
            raise ValueError("Semua class_counts harus lebih besar dari nol.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing harus berada pada [0, 1).")
        self.register_buffer("log_class_counts", counts.log())
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.log_class_counts.numel():
            raise ValueError(
                "Dimensi kelas logits tidak cocok dengan class_counts "
                f"({logits.shape} vs {self.log_class_counts.numel()})."
            )
        adjusted_logits = logits + self.log_class_counts.to(dtype=logits.dtype)
        return F.cross_entropy(
            adjusted_logits,
            labels,
            label_smoothing=self.label_smoothing,
        )


def _multi_rbf_kernel(
    source: Tensor,
    target: Tensor,
    kernel_mul: float,
    kernel_num: int,
    fixed_sigma: float | None,
) -> Tensor:
    total = torch.cat((source, target), dim=0)
    distances = torch.cdist(total, total, p=2).square()
    if fixed_sigma is None:
        count = total.shape[0]
        sigma = distances.detach().sum() / max(count * count - count, 1)
    else:
        sigma = torch.as_tensor(fixed_sigma, device=total.device, dtype=total.dtype)
    sigma = sigma.clamp_min(1e-6) / (kernel_mul ** (kernel_num // 2))
    return sum(torch.exp(-distances / (sigma * (kernel_mul**i))) for i in range(kernel_num))


class MMDLoss(nn.Module):
    def __init__(
        self, kernel_mul: float = 2.0, kernel_num: int = 5, fixed_sigma: float | None = None
    ):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fixed_sigma = fixed_sigma

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        batch_s = source.shape[0]
        kernel = _multi_rbf_kernel(
            source, target, self.kernel_mul, self.kernel_num, self.fixed_sigma
        )
        k_ss = kernel[:batch_s, :batch_s]
        k_tt = kernel[batch_s:, batch_s:]
        k_st = kernel[:batch_s, batch_s:]
        return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()


class NonTargetExpertDiversityLoss(nn.Module):
    """Promote complementary expert distributions outside their main class.

    The shared top class is masked before symmetric KL is measured. Minimizing
    ``exp(-KL)`` rewards diversity without directly asking the experts to
    disagree about the most likely class.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, first_logits: Tensor, second_logits: Tensor) -> Tensor:
        if first_logits.shape != second_logits.shape:
            raise ValueError("Logit kedua expert harus memiliki shape yang sama.")
        if first_logits.ndim != 2 or first_logits.shape[1] < 2:
            raise ValueError("Diversity loss membutuhkan logit [batch, class>=2].")

        first = first_logits.softmax(dim=1)
        second = second_logits.softmax(dim=1)
        shared_top = ((first + second) * 0.5).detach().argmax(dim=1)
        mask = torch.ones_like(first).scatter_(1, shared_top[:, None], 0.0)
        first = first * mask
        second = second * mask
        first = first / first.sum(dim=1, keepdim=True).clamp_min(self.eps)
        second = second / second.sum(dim=1, keepdim=True).clamp_min(self.eps)
        first = first.clamp_min(self.eps)
        second = second.clamp_min(self.eps)
        kl_first_second = (first * (first.log() - second.log())).sum(dim=1)
        kl_second_first = (second * (second.log() - first.log())).sum(dim=1)
        symmetric_kl = 0.5 * (kl_first_second + kl_second_first)
        return torch.exp(-symmetric_kl).mean()


class LMMDLoss(nn.Module):
    """Class-conditional MMD using source labels and target soft pseudo-labels."""

    def __init__(
        self,
        num_classes: int,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fixed_sigma: float | None = None,
        confidence_threshold: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fixed_sigma = fixed_sigma
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_labels: Tensor,
        target_logits: Tensor,
    ) -> Tensor:
        source_weights = F.one_hot(
            source_labels, num_classes=self.num_classes
        ).to(dtype=source.dtype)
        target_weights = F.softmax(target_logits.detach(), dim=1).to(dtype=target.dtype)
        if self.confidence_threshold > 0:
            keep = target_weights.max(dim=1).values >= self.confidence_threshold
            target_weights = target_weights * keep[:, None]

        source_mass = source_weights.sum(dim=0)
        target_mass = target_weights.sum(dim=0)
        active = (source_mass > 0) & (target_mass > 1e-8)
        if not torch.any(active):
            return source.sum() * 0.0

        source_weights = source_weights[:, active] / source_mass[active].clamp_min(1e-8)
        target_weights = target_weights[:, active] / target_mass[active].clamp_min(1e-8)
        class_count = active.sum().to(dtype=source.dtype)
        w_ss = source_weights @ source_weights.T / class_count
        w_tt = target_weights @ target_weights.T / class_count
        w_st = source_weights @ target_weights.T / class_count

        kernel = _multi_rbf_kernel(
            source, target, self.kernel_mul, self.kernel_num, self.fixed_sigma
        )
        batch_s = source.shape[0]
        k_ss = kernel[:batch_s, :batch_s]
        k_tt = kernel[batch_s:, batch_s:]
        k_st = kernel[:batch_s, batch_s:]
        return (w_ss * k_ss).sum() + (w_tt * k_tt).sum() - 2 * (w_st * k_st).sum()
