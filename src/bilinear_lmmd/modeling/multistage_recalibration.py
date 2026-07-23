from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class MultistageFusionOutput:
    logits: Tensor
    embedding: Tensor
    domain_logits: Tensor | None = None
    parent_logits: Tensor | None = None
    expert_logits: dict[str, Tensor] | None = None
    gate_weights: Tensor | None = None
    open_set_loss: Tensor | None = None


class StageProjection(nn.Module):
    """Project one endpoint without discarding its spatial arrangement."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.layers(feature)


class AdaptiveStageChannelGate(nn.Module):
    """Predict per-image, per-channel weights over aligned backbone stages."""

    def __init__(self, stages: int, channels: int, hidden_dim: int):
        super().__init__()
        if stages < 2:
            raise ValueError("Adaptive gate membutuhkan sedikitnya dua stage.")
        if channels <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensi channel dan hidden gate harus positif.")
        self.stages = stages
        self.channels = channels
        self.network = nn.Sequential(
            nn.Linear(stages * channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, stages * channels),
        )
        # MSF1 starts as the exact uniform fusion used by MSF0. Learning,
        # rather than random initialization, must create stage preference.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5:
            raise ValueError("features harus berbentuk [B, S, C, H, W].")
        batch, stages, channels, _, _ = features.shape
        if stages != self.stages or channels != self.channels:
            raise ValueError(
                "Jumlah stage/channel feature tidak cocok dengan konfigurasi gate."
            )
        descriptor = features.mean(dim=(-2, -1)).reshape(batch, stages * channels)
        logits = self.network(descriptor).reshape(batch, stages, channels)
        return torch.softmax(logits, dim=1)


class MultistageRecalibrationModel(nn.Module):
    """Single-pass spatial multistage fusion with a controlled adaptive ablation.

    ``fixed`` (MSF0) averages three aligned endpoint maps. ``adaptive`` (MSF1)
    uses a stage-channel gate whose weights sum to one over stages. Both modes
    share the same encoder, projections, fused embedding, and classifier.

    This is deliberately not PMG/E2: it uses an intact image, one forward pass,
    one classifier, and one optimization objective.
    """

    MODES = {"fixed", "adaptive"}

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        out_indices: tuple[int, ...],
        fusion_dim: int,
        target_stage: int,
        gate_hidden_dim: int,
        dropout: float,
        pretrained: bool,
        mode: str,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode harus salah satu dari {sorted(self.MODES)}.")
        if len(out_indices) != 3:
            raise ValueError("Multistage fusion v1 membutuhkan tepat tiga endpoint.")
        if not 0 <= target_stage < len(out_indices):
            raise ValueError("target_stage berada di luar endpoint yang tersedia.")
        if fusion_dim <= 0:
            raise ValueError("fusion_dim harus lebih besar dari nol.")

        self.backbone_name = backbone
        self.head = f"multistage_{mode}"
        self.mode = mode
        self.target_stage = int(target_stage)
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
            StageProjection(channel, fusion_dim) for channel in channels
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.gate = (
            AdaptiveStageChannelGate(
                stages=len(channels),
                channels=fusion_dim,
                hidden_dim=gate_hidden_dim,
            )
            if mode == "adaptive"
            else None
        )
        self.output_dim = fusion_dim

    @staticmethod
    def _align(feature: Tensor, target_size: tuple[int, int]) -> Tensor:
        height, width = feature.shape[-2:]
        target_height, target_width = target_size
        if (height, width) == target_size:
            return feature
        if height >= target_height and width >= target_width:
            return F.adaptive_avg_pool2d(feature, target_size)
        return F.interpolate(
            feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def aligned_features(self, images: Tensor) -> Tensor:
        endpoints = self.encoder(images)
        if len(endpoints) != 3:
            raise RuntimeError("Encoder tidak mengembalikan tiga endpoint.")
        target_size = tuple(endpoints[self.target_stage].shape[-2:])
        aligned = [
            self._align(projection(feature), target_size)
            for projection, feature in zip(self.projections, endpoints)
        ]
        return torch.stack(aligned, dim=1)

    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> MultistageFusionOutput:
        del labels
        if domain_strength is not None:
            raise ValueError("Multistage fusion v1 hanya mendukung source_only.")
        aligned = self.aligned_features(images)
        if self.gate is None:
            weights = aligned.new_full(
                (aligned.shape[0], aligned.shape[1], aligned.shape[2]),
                1.0 / aligned.shape[1],
            )
        else:
            weights = self.gate(aligned)
        fused = (aligned * weights[..., None, None]).sum(dim=1)
        embedding = F.adaptive_avg_pool2d(fused, output_size=1).flatten(1)
        logits = self.classifier(self.dropout(embedding))
        return MultistageFusionOutput(
            logits=logits,
            embedding=embedding,
            gate_weights=weights,
        )
