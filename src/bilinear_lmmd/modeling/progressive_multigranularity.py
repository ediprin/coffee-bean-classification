from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ProgressiveModelOutput:
    logits: Tensor
    embedding: Tensor
    domain_logits: Tensor | None = None
    parent_logits: Tensor | None = None
    expert_logits: dict[str, Tensor] | None = None
    gate_weights: Tensor | None = None
    open_set_loss: Tensor | None = None


@dataclass
class ProgressiveBranchOutput:
    logits: Tensor
    pooled: Tensor
    descriptor: Tensor


class BasicConv(nn.Module):
    """PMG-style projection block adapted to arbitrary endpoint channels."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1.0e-5, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1.0e-5, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.layers(feature)


class ProgressiveClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.feature = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.classifier(self.feature(embedding))


class ProgressiveMultiGranularityModel(nn.Module):
    """EfficientNet adaptation of PMG and PMG-V2 training components.

    PMG's three endpoint-specific projection/classification branches and
    combined inference are retained. The optional category-consistency mode is
    consumed by the dedicated trainer. ResNet-specific consistent block
    convolution from PMG-V2 is deliberately not claimed or emulated here.
    """

    BRANCH_NAMES = ("fine", "medium", "coarse")

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        out_indices: tuple[int, int, int],
        feature_dim: int,
        branch_dim: int,
        dropout: float,
        pretrained: bool,
        category_consistency: bool,
    ):
        super().__init__()
        if len(out_indices) != 3:
            raise ValueError("Progressive multi-granularity membutuhkan 3 endpoint.")
        if feature_dim <= 0 or branch_dim <= 0:
            raise ValueError("feature_dim dan branch_dim harus lebih besar dari nol.")
        self.backbone_name = backbone
        self.head = (
            "progressive_multigranularity_consistency"
            if category_consistency
            else "progressive_multigranularity"
        )
        self.category_consistency = bool(category_consistency)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        if len(channels) != 3:
            raise RuntimeError("Encoder tidak menghasilkan tiga feature map.")

        self.projections = nn.ModuleList(
            BasicConv(channel, feature_dim, branch_dim) for channel in channels
        )
        self.branch_classifiers = nn.ModuleList(
            ProgressiveClassifier(branch_dim, feature_dim, num_classes, dropout)
            for _ in channels
        )
        self.concat_classifier = ProgressiveClassifier(
            branch_dim * len(channels),
            feature_dim,
            num_classes,
            dropout,
        )
        self.output_dim = branch_dim * len(channels)

    @staticmethod
    def category_descriptor(feature: Tensor, block_size: int = 7) -> Tensor:
        """PMG-V2-style channel descriptor for same-class consistency.

        The official ResNet implementation average-pools 7x7 regions, applies
        global max pooling, then min-max normalizes channels. Adaptive handling
        keeps the same operation valid for EfficientNet endpoint geometries.
        """

        kernel = min(block_size, feature.shape[-2], feature.shape[-1])
        pooled = F.avg_pool2d(feature, kernel_size=kernel, stride=kernel)
        pooled = F.adaptive_max_pool2d(pooled, output_size=1).flatten(1)
        minimum = pooled.amin(dim=1, keepdim=True)
        maximum = pooled.amax(dim=1, keepdim=True)
        return (pooled - minimum) / (maximum - minimum).clamp_min(1.0e-6)

    def _encode_branch(
        self,
        feature: Tensor,
        branch_index: int,
    ) -> ProgressiveBranchOutput:
        projected = self.projections[branch_index](feature)
        pooled = F.adaptive_max_pool2d(projected, output_size=1).flatten(1)
        logits = self.branch_classifiers[branch_index](pooled)
        descriptor = self.category_descriptor(feature)
        return ProgressiveBranchOutput(logits, pooled, descriptor)

    def forward_branch(self, images: Tensor, branch_index: int) -> ProgressiveBranchOutput:
        if branch_index not in range(3):
            raise ValueError("branch_index harus 0, 1, atau 2.")
        features = self.encoder(images)
        return self._encode_branch(features[branch_index], branch_index)

    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ProgressiveModelOutput:
        del labels
        if domain_strength is not None:
            raise ValueError("Progressive multi-granularity hanya mendukung source_only.")
        features = self.encoder(images)
        branches = [
            self._encode_branch(feature, index)
            for index, feature in enumerate(features)
        ]
        embedding = torch.cat([branch.pooled for branch in branches], dim=1)
        concat_logits = self.concat_classifier(embedding)
        branch_logits = {
            name: branch.logits
            for name, branch in zip(self.BRANCH_NAMES, branches)
        }
        expert_logits = {**branch_logits, "concat": concat_logits}
        # PMG evaluates the sum of the three branch logits and concat logits.
        logits = concat_logits + sum(branch_logits.values())
        return ProgressiveModelOutput(
            logits=logits,
            embedding=embedding,
            expert_logits=expert_logits,
        )


def jigsaw_generator(images: Tensor, parts: int) -> Tensor:
    """Shuffle an n-by-n image grid using one permutation for the batch."""

    if images.ndim != 4:
        raise ValueError("images harus memiliki bentuk [B, C, H, W].")
    if parts <= 0:
        raise ValueError("parts harus lebih besar dari nol.")
    batch, channels, height, width = images.shape
    if height % parts or width % parts:
        raise ValueError(
            f"Ukuran {height}x{width} tidak habis dibagi grid {parts}x{parts}."
        )
    patch_height = height // parts
    patch_width = width // parts
    patches = (
        images.reshape(batch, channels, parts, patch_height, parts, patch_width)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, parts * parts, channels, patch_height, patch_width)
    )
    permutation = torch.randperm(parts * parts, device=images.device)
    shuffled = patches[:, permutation]
    return (
        shuffled.reshape(batch, parts, parts, channels, patch_height, patch_width)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )
