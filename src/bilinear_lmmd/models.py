from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - actionable runtime message
    raise ImportError("Paket 'timm' diperlukan. Jalankan: pip install -r requirements.txt") from exc


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.strength * grad_output, None


def gradient_reverse(x: Tensor, strength: float = 1.0) -> Tensor:
    return _GradientReverse.apply(x, strength)


class ArcMarginClassifier(nn.Module):
    """ArcFace classifier with ordinary cosine logits at inference time.

    During training, the angular margin is applied only to the ground-truth
    class. Without labels the layer returns scaled cosine logits, so existing
    evaluation and checkpoint-prediction code can use the model unchanged.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.3,
    ):
        super().__init__()
        if scale <= 0:
            raise ValueError("arcface_scale harus lebih besar dari nol.")
        if not 0 <= margin < math.pi / 2:
            raise ValueError("arcface_margin harus berada pada [0, pi/2).")
        self.in_features = in_features
        self.out_features = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_margin = math.cos(self.margin)
        self.sin_margin = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_correction = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embedding: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = F.linear(
            F.normalize(embedding, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.scale
        if labels.ndim != 1 or labels.shape[0] != embedding.shape[0]:
            raise ValueError("Label ArcFace harus berbentuk [batch].")

        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        phi = cosine * self.cos_margin - sine * self.sin_margin
        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.margin_correction,
        )
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


class GAPHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.output_dim = channels

    def forward(self, features: list[Tensor]) -> Tensor:
        return F.adaptive_avg_pool2d(features[-1], 1).flatten(1)


class FactorizedBilinearPooling(nn.Module):
    """Single-layer low-rank bilinear control on the deepest feature.

    ``projection_dim`` controls the bilinear rank. ``output_dim`` can expand
    that representation so the downstream classifier matches the HBP model.
    """

    def __init__(self, channels: int, projection_dim: int, output_dim: int | None = None):
        super().__init__()
        self.left = nn.Conv2d(channels, projection_dim, kernel_size=1, bias=False)
        self.right = nn.Conv2d(channels, projection_dim, kernel_size=1, bias=False)
        self.expansion = (
            nn.Linear(projection_dim, output_dim, bias=False)
            if output_dim is not None and output_dim != projection_dim
            else nn.Identity()
        )
        self.output_dim = output_dim or projection_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        feature = features[-1]
        interaction = (self.left(feature) * self.right(feature)).flatten(2).mean(-1)
        interaction = torch.sign(interaction) * torch.sqrt(torch.abs(interaction) + 1e-8)
        interaction = F.normalize(interaction, p=2, dim=1)
        return F.normalize(self.expansion(interaction), p=2, dim=1)


class HierarchicalBilinearPooling(nn.Module):
    """Low-dimensional HBP over three backbone depths.

    Each feature is projected to ``projection_dim`` channels and resized to the
    deepest spatial resolution. Pairwise multiplicative interactions (1-2,
    1-3, 2-3) are pooled, signed-square-rooted, L2-normalized, and concatenated.
    """

    def __init__(self, channels: list[int], projection_dim: int):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("HBP membutuhkan tepat tiga feature map.")
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, projection_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(projection_dim),
                    nn.ReLU(inplace=True),
                )
                for channel in channels
            ]
        )
        self.output_dim = projection_dim * 3

    @staticmethod
    def _normalize(x: Tensor) -> Tensor:
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8)
        return F.normalize(x, p=2, dim=1)

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != 3:
            raise ValueError(f"HBP menerima 3 feature map, didapat {len(features)}.")
        target_size = features[-1].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.adaptive_avg_pool2d(feature, target_size)
            projected.append(feature)

        pairwise = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            interaction = (projected[left] * projected[right]).flatten(2).mean(-1)
            pairwise.append(self._normalize(interaction))
        return torch.cat(pairwise, dim=1)


class SPPFAttention(nn.Module):
    """SPPF with channel-spatial attention and a residual connection.

    This follows Hong et al. (2026), equations 8-12: three sequential 5x5
    max-pooling operations preserve spatial resolution; their outputs are
    fused, recalibrated by channel and spatial attention, then added to the
    original feature map.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels SPPF-Attention harus lebih besar dari nol.")
        if reduction <= 0:
            raise ValueError("reduction SPPF-Attention harus lebih besar dari nol.")
        hidden_channels = max(1, channels // 2)
        attention_hidden = max(1, channels // reduction)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, attention_hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, feature: Tensor) -> Tensor:
        base = self.reduce(feature)
        pooled_1 = self.pool(base)
        pooled_2 = self.pool(pooled_1)
        pooled_3 = self.pool(pooled_2)
        fused = self.fuse(torch.cat((base, pooled_1, pooled_2, pooled_3), dim=1))

        channel_attention = torch.sigmoid(
            self.channel_mlp(F.adaptive_avg_pool2d(fused, 1))
        )
        channel_refined = fused * channel_attention
        spatial_descriptor = torch.cat(
            (
                channel_refined.mean(dim=1, keepdim=True),
                channel_refined.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_attention = torch.sigmoid(self.spatial(spatial_descriptor))
        return channel_refined * spatial_attention + feature


class SPPFAttentionHBP(nn.Module):
    """Refine only the deepest feature with SPPF-Attention before HBP."""

    def __init__(
        self,
        channels: list[int],
        projection_dim: int,
        attention_reduction: int = 16,
    ):
        super().__init__()
        self.hbp = HierarchicalBilinearPooling(channels, projection_dim)
        # Keep M1's HBP, classifier, and subsequent RNG stream identical at
        # the same seed. Only the new refinement branch receives extra weights.
        rng_state = torch.random.get_rng_state()
        self.attention = SPPFAttention(channels[-1], attention_reduction)
        torch.random.set_rng_state(rng_state)
        self.output_dim = self.hbp.output_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != 3:
            raise ValueError(
                f"SPPF-Attention-HBP menerima 3 feature map, didapat {len(features)}."
            )
        refined = list(features)
        refined[-1] = self.attention(refined[-1])
        return self.hbp(refined)


class SpatiallyPreservedHBP(HierarchicalBilinearPooling):
    """HBP with a fixed interaction grid that preserves intermediate detail.

    The baseline HBP aligns every projected feature map to the deepest feature
    resolution.  MobileNetV3 endpoints ``(1, 3, 4)`` have reductions
    ``(4, 16, 32)``, so a 224 px input is reduced from ``(56, 14, 7)`` to
    ``7 x 7`` before the pairwise products.  This controlled variant aligns the
    maps to a configurable intermediate grid instead.  It changes no trainable
    parameter and retains the same pairwise products and normalization as HBP.
    """

    def __init__(
        self,
        channels: list[int],
        projection_dim: int,
        spatial_size: int = 14,
    ):
        super().__init__(channels, projection_dim)
        if spatial_size <= 0:
            raise ValueError("hbp_spatial_size harus lebih besar dari nol.")
        self.spatial_size = int(spatial_size)

    def _align(self, feature: Tensor) -> Tensor:
        target_size = (self.spatial_size, self.spatial_size)
        current_size = feature.shape[-2:]
        if current_size == target_size:
            return feature
        if (
            current_size[0] >= self.spatial_size
            and current_size[1] >= self.spatial_size
        ):
            return F.adaptive_avg_pool2d(feature, target_size)
        return F.interpolate(
            feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != 3:
            raise ValueError(f"SP-HBP menerima 3 feature map, didapat {len(features)}.")
        projected = [
            self._align(projection(feature))
            for projection, feature in zip(self.projections, features)
        ]

        pairwise = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            interaction = (projected[left] * projected[right]).flatten(2).mean(-1)
            pairwise.append(self._normalize(interaction))
        return torch.cat(pairwise, dim=1)


class LocalMaxExpert(nn.Module):
    """Compact local-detail expert over an intermediate feature map."""

    def __init__(self, channels: int, output_dim: int):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("moe_local_dim harus lebih besar dari nol.")
        self.projector = nn.Sequential(
            nn.Conv2d(channels, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = output_dim

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.projector(feature)
        return F.adaptive_max_pool2d(feature, 1).flatten(1)


class CBAMAttentionPaths(nn.Module):
    """Channel-then-spatial attention paths used by the paper's equations 8-9."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        average = F.adaptive_avg_pool2d(feature, 1)
        maximum = F.adaptive_max_pool2d(feature, 1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(average) + self.channel_mlp(maximum)
        )
        channel_feature = channel_attention * feature

        spatial_descriptor = torch.cat(
            (
                channel_feature.mean(dim=1, keepdim=True),
                channel_feature.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_attention = torch.sigmoid(self.spatial(spatial_descriptor))
        spatial_feature = spatial_attention * channel_feature
        return channel_feature, spatial_feature


class FixedFusionCBAM(nn.Module):
    """Parameter-free 50/50 fusion control for the LGF attention paths."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.paths = CBAMAttentionPaths(channels, reduction)

    def forward(self, feature: Tensor) -> Tensor:
        channel_feature, spatial_feature = self.paths(feature)
        return 0.5 * (channel_feature + spatial_feature)


class LGFCBAM(nn.Module):
    """Learnable gated fusion of channel and spatial CBAM feature maps.

    This follows equations 8-11/pseudocode in Techie-Menson et al. (2026):
    spatial attention refines the channel-attended map, and a per-sample
    softmax gate fuses both paths. The last gate layer starts at zero so the
    module initially matches the fixed 50/50 control.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.paths = CBAMAttentionPaths(channels, reduction)
        hidden = max(2, channels // reduction)
        # Preserve the global RNG stream so a same-seed fixed-fusion control
        # receives identical encoder, attention-path, and classifier weights.
        rng_state = torch.random.get_rng_state()
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        torch.random.set_rng_state(rng_state)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self.last_gate_weights: Tensor | None = None

    def forward(self, feature: Tensor) -> Tensor:
        channel_feature, spatial_feature = self.paths(feature)
        descriptors = torch.cat(
            (
                F.adaptive_avg_pool2d(channel_feature, 1).flatten(1),
                F.adaptive_avg_pool2d(spatial_feature, 1).flatten(1),
            ),
            dim=1,
        )
        gates = torch.softmax(self.gate_mlp(descriptors), dim=1)
        self.last_gate_weights = gates.detach()
        alpha = gates[:, 0].view(-1, 1, 1, 1)
        beta = gates[:, 1].view(-1, 1, 1, 1)
        return alpha * channel_feature + beta * spatial_feature


class AttentionGAP(nn.Module):
    """Refine the deepest feature map, then use ordinary global average pooling."""

    def __init__(
        self,
        channels: int,
        attention: str,
        reduction: int = 16,
    ):
        super().__init__()
        attention_classes = {
            "fixed": FixedFusionCBAM,
            "lgf": LGFCBAM,
        }
        if attention not in attention_classes:
            raise ValueError("attention harus 'fixed' atau 'lgf'.")
        module = attention_classes[attention]
        self.attention = module(channels, reduction)
        self.output_dim = channels

    def forward(self, features: list[Tensor]) -> Tensor:
        refined = self.attention(features[-1])
        return F.adaptive_avg_pool2d(refined, 1).flatten(1)


class ProjectedHBP(nn.Module):
    """HBP followed by a learnable nonlinear projection capacity control."""

    def __init__(self, channels: list[int], projection_dim: int, output_dim: int):
        super().__init__()
        self.hbp = HierarchicalBilinearPooling(channels, projection_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.hbp.output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        return self.projector(self.hbp(features))


class GAPHBPFeatureFusion(nn.Module):
    """Fuse first-order GAP and second-order HBP representations.

    Each branch is projected and normalized independently before concatenation,
    allowing the final classifier to learn class-dependent branch weights.
    """

    def __init__(
        self,
        channels: list[int],
        projection_dim: int,
        hbp_output_dim: int,
        gap_output_dim: int,
    ):
        super().__init__()
        self.hbp = HierarchicalBilinearPooling(channels, projection_dim)
        self.gap = GAPHead(channels[-1])
        self.hbp_projector = nn.Sequential(
            nn.Linear(self.hbp.output_dim, hbp_output_dim),
            nn.LayerNorm(hbp_output_dim),
            nn.GELU(),
        )
        self.gap_projector = nn.Sequential(
            nn.Linear(self.gap.output_dim, gap_output_dim),
            nn.LayerNorm(gap_output_dim),
            nn.GELU(),
        )
        self.output_dim = hbp_output_dim + gap_output_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        hbp = self.hbp_projector(self.hbp(features))
        gap = self.gap_projector(self.gap(features))
        return torch.cat((hbp, gap), dim=1)


class ResidualHBPControl(nn.Module):
    """Preserve the full HBP vector and append a small HBP-only branch."""

    def __init__(
        self, channels: list[int], projection_dim: int, auxiliary_dim: int
    ):
        super().__init__()
        self.hbp = HierarchicalBilinearPooling(channels, projection_dim)
        self.auxiliary_projector = nn.Sequential(
            nn.Linear(self.hbp.output_dim, auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
        )
        self.output_dim = self.hbp.output_dim + auxiliary_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        hbp = self.hbp(features)
        auxiliary = F.normalize(self.auxiliary_projector(hbp), p=2, dim=1)
        return torch.cat((hbp, auxiliary), dim=1)


class ResidualGAPHBPFusion(nn.Module):
    """Preserve the full HBP vector and append a normalized GAP residual."""

    def __init__(
        self, channels: list[int], projection_dim: int, gap_output_dim: int
    ):
        super().__init__()
        self.hbp = HierarchicalBilinearPooling(channels, projection_dim)
        self.gap = GAPHead(channels[-1])
        self.gap_projector = nn.Sequential(
            nn.Linear(self.gap.output_dim, gap_output_dim),
            nn.LayerNorm(gap_output_dim),
            nn.GELU(),
        )
        self.output_dim = self.hbp.output_dim + gap_output_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        hbp = self.hbp(features)
        gap = F.normalize(self.gap_projector(self.gap(features)), p=2, dim=1)
        return torch.cat((hbp, gap), dim=1)


@dataclass
class ModelOutput:
    logits: Tensor
    embedding: Tensor
    domain_logits: Tensor | None = None
    parent_logits: Tensor | None = None
    expert_logits: dict[str, Tensor] | None = None
    gate_weights: Tensor | None = None


class AdaptationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        head: str = "hbp",
        out_indices: tuple[int, ...] = (1, 3, 4),
        projection_dim: int = 512,
        hbp_spatial_size: int = 14,
        bilinear_output_dim: int | None = None,
        hbp_mlp_dim: int = 672,
        fusion_hbp_dim: int = 512,
        fusion_gap_dim: int = 256,
        residual_control_dim: int = 80,
        residual_gap_dim: int = 128,
        attention_reduction: int = 16,
        moe_local_dim: int = 256,
        moe_gate_hidden: int = 32,
        moe_hbp_prior: float = 0.8,
        dropout: float = 0.2,
        classifier: str = "linear",
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.3,
        pretrained: bool = True,
        enable_domain_classifier: bool = False,
        hierarchy_num_parents: int = 0,
    ):
        super().__init__()
        supported_heads = {
            "gap",
            "bilinear",
            "hbp",
            "hbp_moe",
            "sp_hbp",
            "sppf_attention_hbp",
            "hbp_mlp",
            "gap_hbp_fusion",
            "hbp_residual_control",
            "gap_hbp_residual",
            "gap_fixed_cbam",
            "gap_lgf_cbam",
        }
        if head not in supported_heads:
            raise ValueError(f"head harus salah satu dari {sorted(supported_heads)}.")
        hbp_heads = {
            "hbp",
            "hbp_moe",
            "sp_hbp",
            "sppf_attention_hbp",
            "hbp_mlp",
            "gap_hbp_fusion",
            "hbp_residual_control",
            "gap_hbp_residual",
        }
        if head in hbp_heads and len(out_indices) != 3:
            raise ValueError("out_indices untuk head berbasis HBP harus tepat 3 indeks.")

        self.backbone_name = backbone
        self.head = head
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        if head in {"hbp", "hbp_moe"}:
            self.pool = HierarchicalBilinearPooling(channels, projection_dim)
        elif head == "sppf_attention_hbp":
            self.pool = SPPFAttentionHBP(
                channels,
                projection_dim,
                attention_reduction,
            )
        elif head == "sp_hbp":
            self.pool = SpatiallyPreservedHBP(
                channels,
                projection_dim,
                hbp_spatial_size,
            )
        elif head == "gap_fixed_cbam":
            self.pool = AttentionGAP(
                channels[-1],
                attention="fixed",
                reduction=attention_reduction,
            )
        elif head == "gap_lgf_cbam":
            self.pool = AttentionGAP(
                channels[-1],
                attention="lgf",
                reduction=attention_reduction,
            )
        elif head == "hbp_mlp":
            self.pool = ProjectedHBP(channels, projection_dim, hbp_mlp_dim)
        elif head == "gap_hbp_fusion":
            self.pool = GAPHBPFeatureFusion(
                channels,
                projection_dim,
                fusion_hbp_dim,
                fusion_gap_dim,
            )
        elif head == "hbp_residual_control":
            self.pool = ResidualHBPControl(
                channels, projection_dim, residual_control_dim
            )
        elif head == "gap_hbp_residual":
            self.pool = ResidualGAPHBPFusion(
                channels, projection_dim, residual_gap_dim
            )
        elif head == "bilinear":
            self.pool = FactorizedBilinearPooling(
                channels[-1], projection_dim, bilinear_output_dim
            )
        else:
            self.pool = GAPHead(channels[-1])
        self.dropout = nn.Dropout(dropout)
        classifier = classifier.lower()
        if classifier == "linear":
            self.classifier = nn.Linear(self.pool.output_dim, num_classes)
        elif classifier == "arcface":
            self.classifier = ArcMarginClassifier(
                self.pool.output_dim,
                num_classes,
                scale=arcface_scale,
                margin=arcface_margin,
            )
        else:
            raise ValueError("model.classifier harus 'linear' atau 'arcface'.")
        self.classifier_type = classifier

        if hierarchy_num_parents < 0:
            raise ValueError("hierarchy_num_parents tidak boleh negatif.")
        if hierarchy_num_parents and classifier != "linear":
            raise ValueError("Hierarchical supervision saat ini membutuhkan classifier linear.")
        self.parent_classifier: nn.Linear | None = None
        if hierarchy_num_parents:
            # Auxiliary-head initialization must not alter the global RNG state,
            # so same-seed M1 and H1 retain identical backbone/HBP/fine-head
            # initialization and data-loader shuffling.
            rng_state = torch.random.get_rng_state()
            self.parent_classifier = nn.Linear(
                self.pool.output_dim, hierarchy_num_parents
            )
            torch.random.set_rng_state(rng_state)

        self.local_expert: LocalMaxExpert | None = None
        self.local_classifier: nn.Linear | None = None
        self.expert_gate: nn.Sequential | None = None
        if head == "hbp_moe":
            if classifier != "linear":
                raise ValueError("hbp_moe saat ini hanya mendukung classifier linear.")
            if not 0.0 < moe_hbp_prior < 1.0:
                raise ValueError("moe_hbp_prior harus berada di antara 0 dan 1.")
            if moe_gate_hidden <= 0:
                raise ValueError("moe_gate_hidden harus lebih besar dari nol.")
            self.local_expert = LocalMaxExpert(channels[1], moe_local_dim)
            self.local_classifier = nn.Linear(moe_local_dim, num_classes)
            self.expert_gate = nn.Sequential(
                nn.Linear(num_classes * 2, moe_gate_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(moe_gate_hidden, 2),
            )
            nn.init.zeros_(self.expert_gate[-1].weight)
            nn.init.constant_(
                self.expert_gate[-1].bias[0], math.log(moe_hbp_prior)
            )
            nn.init.constant_(
                self.expert_gate[-1].bias[1], math.log(1.0 - moe_hbp_prior)
            )
        hidden = min(1024, max(128, self.pool.output_dim // 2))
        self.domain_classifier = (
            nn.Sequential(
                nn.Linear(self.pool.output_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
            if enable_domain_classifier
            else None
        )

    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        features = self.encoder(x)
        embedding = self.pool(features)
        if self.classifier_type == "arcface":
            logits = self.classifier(embedding, labels)
            classifier_embedding = embedding
        else:
            classifier_embedding = self.dropout(embedding)
            logits = self.classifier(classifier_embedding)
        parent_logits = (
            self.parent_classifier(classifier_embedding)
            if self.parent_classifier is not None
            else None
        )
        expert_logits = None
        gate_weights = None
        if self.head == "hbp_moe":
            if self.local_expert is None or self.local_classifier is None or self.expert_gate is None:
                raise RuntimeError("Komponen hbp_moe belum diinisialisasi.")
            local_embedding = self.local_expert(features[1])
            local_logits = self.local_classifier(self.dropout(local_embedding))
            gate_weights = torch.softmax(
                self.expert_gate(torch.cat((logits, local_logits), dim=1)), dim=1
            )
            expert_logits = {
                "hbp_global": logits,
                "local_gmp": local_logits,
            }
            logits = (
                gate_weights[:, :1] * logits
                + gate_weights[:, 1:] * local_logits
            )
        domain_logits = None
        if domain_strength is not None:
            if self.domain_classifier is None:
                raise RuntimeError(
                    "DANN membutuhkan model.enable_domain_classifier=true."
                )
            domain_logits = self.domain_classifier(
                gradient_reverse(embedding, domain_strength)
            )
        return ModelOutput(
            logits=logits,
            embedding=embedding,
            domain_logits=domain_logits,
            parent_logits=parent_logits,
            expert_logits=expert_logits,
            gate_weights=gate_weights,
        )


def build_model(cfg: dict) -> AdaptationModel:
    return AdaptationModel(
        backbone=cfg["backbone"],
        num_classes=int(cfg["num_classes"]),
        head=cfg.get("head", "hbp"),
        out_indices=tuple(cfg.get("out_indices", (1, 3, 4))),
        projection_dim=int(cfg.get("projection_dim", 512)),
        hbp_spatial_size=int(cfg.get("hbp_spatial_size", 14)),
        bilinear_output_dim=(
            int(cfg["bilinear_output_dim"])
            if cfg.get("bilinear_output_dim") is not None
            else None
        ),
        hbp_mlp_dim=int(cfg.get("hbp_mlp_dim", 672)),
        fusion_hbp_dim=int(cfg.get("fusion_hbp_dim", 512)),
        fusion_gap_dim=int(cfg.get("fusion_gap_dim", 256)),
        residual_control_dim=int(cfg.get("residual_control_dim", 80)),
        residual_gap_dim=int(cfg.get("residual_gap_dim", 128)),
        attention_reduction=int(cfg.get("attention_reduction", 16)),
        moe_local_dim=int(cfg.get("moe_local_dim", 256)),
        moe_gate_hidden=int(cfg.get("moe_gate_hidden", 32)),
        moe_hbp_prior=float(cfg.get("moe_hbp_prior", 0.8)),
        dropout=float(cfg.get("dropout", 0.2)),
        classifier=str(cfg.get("classifier", "linear")),
        arcface_scale=float(cfg.get("arcface_scale", 30.0)),
        arcface_margin=float(cfg.get("arcface_margin", 0.3)),
        pretrained=bool(cfg.get("pretrained", True)),
        enable_domain_classifier=bool(cfg.get("enable_domain_classifier", False)),
        hierarchy_num_parents=int(cfg.get("hierarchy_num_parents", 0)),
    )
