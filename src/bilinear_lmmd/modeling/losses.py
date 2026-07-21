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


class ConfusionAwareSupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss with optional class-pair weighting.

    Positives are every other embedding with the same label. Different-class
    embeddings remain negatives, but their contribution to the denominator is
    multiplied by ``1 + confusion_strength * confusion[a, b]``. The confusion
    matrix must be computed from training predictions only; a zero matrix
    recovers ordinary supervised contrastive learning.
    """

    def __init__(self, temperature: float = 0.1, confusion_strength: float = 0.0):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("contrastive temperature harus lebih besar dari nol.")
        if confusion_strength < 0.0:
            raise ValueError("confusion_strength tidak boleh negatif.")
        self.temperature = float(temperature)
        self.confusion_strength = float(confusion_strength)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        class_confusion: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings harus berbentuk [sample, feature].")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels harus berbentuk [sample] dan sejajar embeddings.")
        if embeddings.shape[0] < 2:
            raise ValueError("SupCon membutuhkan minimal dua embedding.")

        labels = labels.to(device=embeddings.device, dtype=torch.long)
        normalized = F.normalize(embeddings, p=2, dim=1)
        logits = normalized @ normalized.t() / self.temperature
        sample_count = embeddings.shape[0]
        self_mask = torch.eye(
            sample_count, device=embeddings.device, dtype=torch.bool
        )
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        positive_count = positive_mask.sum(dim=1)
        if torch.any(positive_count == 0):
            raise ValueError(
                "Setiap anchor harus memiliki minimal satu positive sekelas."
            )

        log_weights = torch.zeros_like(logits)
        if self.confusion_strength > 0.0:
            if class_confusion is None:
                raise ValueError(
                    "class_confusion wajib saat confusion_strength > 0."
                )
            if class_confusion.ndim != 2 or class_confusion.shape[0] != class_confusion.shape[1]:
                raise ValueError("class_confusion harus berupa matriks persegi.")
            if labels.max().item() >= class_confusion.shape[0]:
                raise ValueError("Label berada di luar dimensi class_confusion.")
            confusion = class_confusion.to(
                device=embeddings.device, dtype=embeddings.dtype
            ).clamp_min(0.0)
            pair_weights = 1.0 + self.confusion_strength * confusion[
                labels[:, None], labels[None, :]
            ]
            # Positive terms stay unweighted; weighting only prioritizes hard
            # negative class pairs in the partition function.
            pair_weights = torch.where(
                labels[:, None].eq(labels[None, :]),
                torch.ones_like(pair_weights),
                pair_weights,
            )
            log_weights = pair_weights.log()

        denominator_logits = logits + log_weights
        denominator_logits = denominator_logits.masked_fill(self_mask, -torch.inf)
        log_denominator = torch.logsumexp(denominator_logits, dim=1)
        positive_log_probability = logits - log_denominator[:, None]
        per_anchor = -(
            positive_log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count
        )
        return per_anchor.mean()


class KnowledgeDistillationLoss(nn.Module):
    """Blend hard-label CE with temperature-scaled teacher KL divergence."""

    def __init__(
        self,
        temperature: float = 2.0,
        hard_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature harus lebih besar dari nol.")
        if not 0.0 <= hard_weight <= 1.0:
            raise ValueError("hard_weight harus berada pada [0, 1].")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing harus berada pada [0, 1).")
        self.temperature = float(temperature)
        self.hard_weight = float(hard_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "Dimensi logits student dan teacher harus identik "
                f"({student_logits.shape} vs {teacher_logits.shape})."
            )
        hard = F.cross_entropy(
            student_logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        temperature = self.temperature
        soft = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)
        total = self.hard_weight * hard + (1.0 - self.hard_weight) * soft
        return total, {"hard_ce": hard, "soft_kl": soft}


class OntologyMarginalLoss(nn.Module):
    """Exact negative log-likelihood for coarsened/candidate-set labels.

    A boolean compatibility mask identifies the canonical leaves compatible
    with each observed label. The implementation uses two log-sum-exp terms
    and never materializes probabilities, which is stable for extreme logits.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction harus 'none', 'mean', atau 'sum'.")
        self.reduction = reduction

    def forward(self, logits: Tensor, compatibility: Tensor) -> Tensor:
        if logits.ndim != 2 or compatibility.shape != logits.shape:
            raise ValueError(
                "logits dan compatibility harus memiliki shape [batch, canonical_class] "
                f"yang sama ({logits.shape} vs {compatibility.shape})."
            )
        compatibility = compatibility.to(device=logits.device, dtype=torch.bool)
        if not torch.all(compatibility.any(dim=1)):
            raise ValueError("Setiap sampel harus kompatibel dengan minimal satu leaf.")
        compatible_logits = logits.masked_fill(~compatibility, -torch.inf)
        losses = torch.logsumexp(logits, dim=1) - torch.logsumexp(
            compatible_logits, dim=1
        )
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()


class TaxonomyCompatibleContrastiveLoss(nn.Module):
    """Supervised contrastive loss that does not invent fine labels.

    Equal compatible sets are positives. Disjoint sets are definite negatives.
    Partially overlapping or parent-child sets are ignored by default because
    treating them as either positive or negative can collapse real subclasses
    or introduce false negatives. ``nested_positives`` is exposed only as an
    explicit ablation.
    """

    def __init__(self, temperature: float = 0.1, nested_positives: bool = False):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature harus lebih besar dari nol.")
        self.temperature = float(temperature)
        self.nested_positives = bool(nested_positives)

    def forward(self, embeddings: Tensor, compatibility: Tensor) -> Tensor:
        if embeddings.ndim != 2 or compatibility.ndim != 2:
            raise ValueError("embeddings dan compatibility harus berupa matriks.")
        if embeddings.shape[0] != compatibility.shape[0]:
            raise ValueError("Batch embeddings dan compatibility berbeda.")
        batch = embeddings.shape[0]
        if batch < 2:
            return embeddings.sum() * 0.0
        masks = compatibility.to(device=embeddings.device, dtype=torch.bool)
        if not torch.all(masks.any(dim=1)):
            raise ValueError("Set compatibility tidak boleh kosong.")

        overlap = (masks[:, None, :] & masks[None, :, :]).any(dim=2)
        equal = (masks[:, None, :] == masks[None, :, :]).all(dim=2)
        positives = equal
        if self.nested_positives:
            left_subset = ((~masks[:, None, :]) | masks[None, :, :]).all(dim=2)
            right_subset = ((~masks[None, :, :]) | masks[:, None, :]).all(dim=2)
            positives = positives | left_subset | right_subset
        eye = torch.eye(batch, device=embeddings.device, dtype=torch.bool)
        positives = positives & ~eye
        candidates = (positives | ~overlap) & ~eye

        normalized = F.normalize(embeddings, p=2, dim=1)
        similarity = normalized @ normalized.T / self.temperature
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        exp_similarity = similarity.exp() * candidates
        denominator = exp_similarity.sum(dim=1).clamp_min(1e-12)
        log_probability = similarity - denominator.log()[:, None]
        positive_count = positives.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return embeddings.sum() * 0.0
        per_anchor = -(
            (log_probability * positives).sum(dim=1)
            / positive_count.clamp_min(1)
        )
        return per_anchor[valid].mean()


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
