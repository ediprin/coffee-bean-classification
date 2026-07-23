from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F

from bilinear_lmmd.modeling.dsconv import (
    replace_spatial_convolutions_with_dsconv,
)

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


class AdversarialReciprocalPointClassifier(nn.Module):
    """ARPLoss classifier without the optional confusing-sample GAN.

    This follows the official ARPL implementation: one learnable reciprocal
    point per class, logits equal to normalized squared-L2 distance minus the
    dot product, and a learnable-radius margin regularizer for known samples.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        regularization_weight: float = 0.1,
        margin: float = 1.0,
    ):
        super().__init__()
        if in_features <= 0 or num_classes < 2:
            raise ValueError("Dimensi ARPL dan jumlah kelas harus valid.")
        if regularization_weight < 0.0:
            raise ValueError("arpl_weight tidak boleh negatif.")
        if margin < 0.0:
            raise ValueError("arpl_margin tidak boleh negatif.")
        self.in_features = int(in_features)
        self.out_features = int(num_classes)
        self.regularization_weight = float(regularization_weight)
        self.margin = float(margin)
        self.reciprocal_points = nn.Parameter(
            0.1 * torch.randn(num_classes, in_features)
        )
        self.radius = nn.Parameter(torch.zeros(1))

    def squared_l2(self, embedding: Tensor) -> Tensor:
        feature_norm = embedding.square().sum(dim=1, keepdim=True)
        center_norm = self.reciprocal_points.square().sum(
            dim=1, keepdim=True
        ).t()
        distance = (
            feature_norm
            - 2.0 * embedding @ self.reciprocal_points.t()
            + center_norm
        )
        return distance / float(embedding.shape[1])

    def forward(self, embedding: Tensor) -> Tensor:
        distance = self.squared_l2(embedding)
        dot_product = embedding @ self.reciprocal_points.t()
        return distance - dot_product

    def open_space_regularization(
        self,
        embedding: Tensor,
        labels: Tensor,
    ) -> Tensor:
        target_points = self.reciprocal_points[labels]
        target_distance = (embedding - target_points).square().mean(dim=1)
        target = torch.ones_like(target_distance)
        radius = self.radius.expand_as(target_distance)
        margin_loss = F.margin_ranking_loss(
            radius,
            target_distance,
            target,
            margin=self.margin,
        )
        return self.regularization_weight * margin_loss


class GAPHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.output_dim = channels

    def forward(self, features: list[Tensor]) -> Tensor:
        return F.adaptive_avg_pool2d(features[-1], 1).flatten(1)


class FactorizedBilinearConvClassifier(nn.Module):
    """Position-wise 1x1 Conv-FBN classifier from Li et al. (ICCV 2017).

    For each class, the layer adds a conventional linear response to a
    symmetric low-rank quadratic response.  DropFactor drops complete rank
    factors during training and scales all factors by the keep probability at
    inference, matching the paper rather than inverted-dropout semantics.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        rank: int = 20,
        keep_prob: float = 0.5,
        *,
        quadratic: bool = True,
    ):
        super().__init__()
        if channels <= 0 or num_classes <= 0:
            raise ValueError("Channel dan jumlah kelas FB Conv harus positif.")
        if rank <= 0:
            raise ValueError("fb_rank harus lebih besar dari nol.")
        if not 0.0 < keep_prob <= 1.0:
            raise ValueError("fb_dropfactor_keep_prob harus berada di (0, 1].")
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.keep_prob = float(keep_prob)
        self.quadratic = bool(quadratic)
        self.linear = nn.Conv2d(channels, num_classes, kernel_size=1, bias=True)
        self.factors = nn.Parameter(torch.empty(num_classes, rank, channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()
        # Keeps the summed rank paths O(1) at initialization. The original
        # source release is unavailable; this adaptation choice is documented
        # explicitly in the protocol rather than attributed to the paper.
        nn.init.normal_(
            self.factors,
            mean=0.0,
            std=1.0 / math.sqrt(self.channels * self.rank),
        )

    def _dropfactor(self, values: Tensor) -> Tensor:
        if self.training:
            mask = torch.empty(
                (1, self.num_classes, self.rank, 1, 1),
                device=values.device,
                dtype=values.dtype,
            ).bernoulli_(self.keep_prob)
            return values * mask
        return values * self.keep_prob

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                "FB Conv mengharapkan feature map BCHW dengan "
                f"C={self.channels}; diterima {tuple(feature.shape)}."
            )
        bounded = torch.tanh(feature)
        linear_map = self.linear(bounded)
        projected = torch.einsum("bchw,orc->borhw", bounded, self.factors)
        factor_response = projected.square() if self.quadratic else projected
        interaction_map = self._dropfactor(factor_response).sum(dim=2)
        score_map = linear_map + interaction_map
        return F.adaptive_avg_pool2d(score_map, 1).flatten(1)


class MatrixPowerNormalizedCovariancePooling(nn.Module):
    """Compact iSQRT-COV / fast MPN-COV pooling.

    The forward path follows the official fast MPN-COV implementation by Li
    et al. (CVPR 2018): channel reduction, centered covariance pooling,
    trace-normalized Newton--Schulz matrix square root, trace
    post-compensation, and upper-triangular vectorization.  The reduction
    dimension is configurable so the representation remains practical on the
    small Coffee17 dataset.
    """

    def __init__(
        self,
        channels: int,
        reduction_dim: int = 128,
        iterations: int = 5,
        epsilon: float = 1.0e-5,
    ):
        super().__init__()
        if reduction_dim <= 1:
            raise ValueError("mpncov_reduction_dim harus lebih besar dari satu.")
        if iterations <= 0:
            raise ValueError("mpncov_iterations harus lebih besar dari nol.")
        if epsilon <= 0.0:
            raise ValueError("mpncov_epsilon harus lebih besar dari nol.")
        self.reduction_dim = int(reduction_dim)
        self.iterations = int(iterations)
        self.epsilon = float(epsilon)
        self.reduction = nn.Sequential(
            nn.Conv2d(channels, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )
        rows, columns = torch.triu_indices(reduction_dim, reduction_dim)
        self.register_buffer("triu_rows", rows, persistent=False)
        self.register_buffer("triu_columns", columns, persistent=False)
        self.output_dim = reduction_dim * (reduction_dim + 1) // 2
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.reduction[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.ones_(self.reduction[1].weight)
        nn.init.zeros_(self.reduction[1].bias)

    def _matrix_square_root(self, covariance: Tensor) -> Tensor:
        batch, channels, _ = covariance.shape
        identity = torch.eye(
            channels, device=covariance.device, dtype=covariance.dtype
        ).expand(batch, -1, -1)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(
            self.epsilon
        )
        y = covariance / trace[:, None, None]
        z = identity
        three_identity = 3.0 * identity
        for _ in range(self.iterations):
            update = 0.5 * (three_identity - z.bmm(y))
            y = y.bmm(update)
            z = update.bmm(z)
        result = y * torch.sqrt(trace)[:, None, None]
        # Suppress tiny numerical asymmetry before upper-triangle extraction.
        return 0.5 * (result + result.transpose(1, 2))

    def forward(self, features: list[Tensor]) -> Tensor:
        feature = self.reduction(features[-1])
        output_dtype = feature.dtype
        # Matrix iterations are intentionally float32 under mixed precision.
        matrix_dtype = (
            torch.float32
            if feature.dtype in {torch.float16, torch.bfloat16}
            else feature.dtype
        )
        feature = feature.to(dtype=matrix_dtype)
        batch, channels, height, width = feature.shape
        spatial = height * width
        flattened = feature.reshape(batch, channels, spatial)
        centered = flattened - flattened.mean(dim=2, keepdim=True)
        covariance = centered.bmm(centered.transpose(1, 2)) / float(spatial)
        square_root = self._matrix_square_root(covariance)
        vector = square_root[:, self.triu_rows, self.triu_columns]
        return vector.to(dtype=output_dtype)


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
            # Keep the second-order product and its normalization in FP32.
            # The learned projections may still use autocast/Tensor Cores.
            projected.append(feature.float())

        pairwise = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            interaction = (projected[left] * projected[right]).flatten(2).mean(-1)
            pairwise.append(self._normalize(interaction))
        return torch.cat(pairwise, dim=1)


class ProjectedHierarchicalGAP(nn.Module):
    """Capacity-matched first-order control for normalized HBP.

    The three selected feature maps use exactly the same learned
    ``Conv-BatchNorm-ReLU`` projections and deepest-grid alignment as
    :class:`HierarchicalBilinearPooling`.  Each projected stage is pooled and
    normalized independently, then concatenated without a multiplicative
    interaction.  HBP and this control therefore have identical trainable
    parameter counts and embedding dimensions; their only representational
    difference is first-order pooling versus pairwise second-order products.
    """

    def __init__(self, channels: list[int], projection_dim: int):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Hierarchical GAP membutuhkan tepat tiga feature map.")
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
            raise ValueError(
                f"Hierarchical GAP menerima 3 feature map, didapat {len(features)}."
            )
        target_size = features[-1].shape[-2:]
        pooled = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.adaptive_avg_pool2d(feature, target_size)
            # Match HBP's numerical policy: projections may use autocast, but
            # pooling and normalization remain FP32 for a fair ablation.
            pooled.append(self._normalize(feature.float().flatten(2).mean(-1)))
        return torch.cat(pooled, dim=1)


class LinearProjectionHBP(nn.Module):
    """Yu-style linear-projection, low-dimensional cross-stage HBP.

    Yu et al. formulate each cross-layer interaction with independent linear
    projections before the Hadamard product.  The legacy
    :class:`HierarchicalBilinearPooling` in this repository adds BatchNorm and
    ReLU to those projections.  This variant deliberately uses bare 1x1
    convolutions so negative projected interactions are retained.  The
    learnable biases start at zero, matching the released Caffe implementation
    without consuming an additional random initialization stream.

    Feature maps from modern backbones can have different spatial resolutions.
    They are therefore still aligned to the deepest selected grid; this is an
    explicit cross-stage adaptation and not a literal reproduction of the
    same-resolution VGG layers used in the original paper.
    """

    def __init__(self, channels: list[int], projection_dim: int):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Linear HBP membutuhkan tepat tiga feature map.")
        if projection_dim <= 0:
            raise ValueError("projection_dim Linear HBP harus lebih besar dari nol.")
        projections = []
        for channel in channels:
            projection = nn.Conv2d(
                channel,
                projection_dim,
                kernel_size=1,
                bias=False,
            )
            projection.bias = nn.Parameter(torch.zeros(projection_dim))
            projections.append(projection)
        self.projections = nn.ModuleList(projections)
        self.output_dim = projection_dim * 3

    @staticmethod
    def _normalize(x: Tensor) -> Tensor:
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8)
        return F.normalize(x, p=2, dim=1)

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != 3:
            raise ValueError(
                f"Linear HBP menerima 3 feature map, didapat {len(features)}."
            )
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


class SPPFAttentionGAP(nn.Module):
    """Refine the deepest feature with SPPF-Attention before ordinary GAP."""

    def __init__(self, channels: int, attention_reduction: int = 16):
        super().__init__()
        # Preserve the RNG stream used by the GAP baseline so the backbone and
        # classifier remain paired at an identical seed. Only this new branch
        # receives additional randomly initialized weights.
        rng_state = torch.random.get_rng_state()
        self.attention = SPPFAttention(channels, attention_reduction)
        torch.random.set_rng_state(rng_state)
        self.output_dim = channels

    def forward(self, features: list[Tensor]) -> Tensor:
        if not features:
            raise ValueError("SPPF-Attention-GAP membutuhkan feature map.")
        refined = self.attention(features[-1])
        return F.adaptive_avg_pool2d(refined, 1).flatten(1)


class MultiScaleDefectExtraction(nn.Module):
    """Residual 3x3/5x5 feature extractor inspired by Chang and Liu (2024).

    The paper uses parallel standard convolutions to retain fine defect lines
    and coarser bean contours.  This adaptation applies the same causal
    operator to the deepest feature map of a modern pretrained backbone and
    adds a residual connection so the experiment measures refinement rather
    than replacing the pretrained representation outright.
    """

    def __init__(self, channels: int, branch_channels: int = 16):
        super().__init__()
        if channels <= 0 or branch_channels <= 0:
            raise ValueError("Channel MDE harus lebih besar dari nol.")
        self.channels = int(channels)
        self.branch_channels = int(branch_channels)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                branch_channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, feature: Tensor) -> Tensor:
        fine = self.branch_3x3(feature)
        coarse = self.branch_5x5(feature)
        return feature + self.fuse(torch.cat((fine, coarse), dim=1))


class MultiScaleDefectGAP(nn.Module):
    """Chang-Liu multiscale refinement followed by ordinary GAP."""

    def __init__(self, channels: int, branch_channels: int = 16):
        super().__init__()
        # Do not shift the baseline classifier initialization or the global
        # RNG stream.  At a paired seed, only this refinement is additional.
        rng_state = torch.random.get_rng_state()
        self.refinement = MultiScaleDefectExtraction(channels, branch_channels)
        torch.random.set_rng_state(rng_state)
        self.output_dim = channels

    def forward(self, features: list[Tensor]) -> Tensor:
        if not features:
            raise ValueError("MDE-GAP membutuhkan feature map.")
        refined = self.refinement(features[-1])
        return F.adaptive_avg_pool2d(refined, 1).flatten(1)


class CapacityResidualGAP(nn.Module):
    """Pointwise capacity control for the MDE 3x3/5x5 spatial operator."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        rng_state = torch.random.get_rng_state()
        self.refinement = PointwiseResidualCapacity(channels, hidden_channels)
        torch.random.set_rng_state(rng_state)
        self.output_dim = channels

    def forward(self, features: list[Tensor]) -> Tensor:
        if not features:
            raise ValueError("Capacity-Residual-GAP membutuhkan feature map.")
        refined = self.refinement(features[-1])
        return F.adaptive_avg_pool2d(refined, 1).flatten(1)


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


class PointwiseResidualCapacity(nn.Module):
    """Parameter-matched deep refinement without pooling or attention."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("Channel capacity control harus lebih besar dari nol.")
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.refine(feature) + feature


class CapacityResidualHBP(nn.Module):
    """Capacity control that refines the deepest feature before ordinary HBP."""

    def __init__(
        self,
        channels: list[int],
        projection_dim: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.hbp = HierarchicalBilinearPooling(channels, projection_dim)
        rng_state = torch.random.get_rng_state()
        self.refinement = PointwiseResidualCapacity(
            channels[-1], hidden_channels
        )
        torch.random.set_rng_state(rng_state)
        self.output_dim = self.hbp.output_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != 3:
            raise ValueError(
                f"Capacity-Residual-HBP menerima 3 feature map, didapat {len(features)}."
            )
        refined = list(features)
        refined[-1] = self.refinement(refined[-1])
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
    open_set_loss: Tensor | None = None


class DecoupledMobileNetV3Encoder(nn.Module):
    """Share MobileNetV3 through block 4 and duplicate blocks 5-6.

    The shared branch exposes block-1 and block-4 features. The GAP and HBP
    branches then receive independent copies of the final MobileNetV3 stages,
    allowing their pooling objectives to shape different late representations.
    """

    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        if backbone != "mobilenetv3_large_100":
            raise ValueError(
                "Decoupled GAP-HBP saat ini dikunci ke mobilenetv3_large_100."
            )
        base = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 3, 4),
        )
        if len(base.blocks) != 7:
            raise RuntimeError("Struktur MobileNetV3 tidak sesuai branch point terkunci.")
        self.stem = nn.Sequential(base.conv_stem, base.bn1, base.act1)
        self.shared_blocks = nn.ModuleList(list(base.blocks[:5]))
        original_late = nn.Sequential(*list(base.blocks[5:]))
        self.gap_late = original_late
        self.hbp_late = copy.deepcopy(original_late)
        self.channels = list(base.feature_info.channels())

    def shared_parameters(self):
        yield from self.stem.parameters()
        yield from self.shared_blocks.parameters()

    def forward(self, images: Tensor) -> tuple[list[Tensor], Tensor]:
        feature = self.stem(images)
        middle: Tensor | None = None
        late_shared: Tensor | None = None
        for index, block in enumerate(self.shared_blocks):
            feature = block(feature)
            if index == 1:
                middle = feature
            elif index == 4:
                late_shared = feature
        if middle is None or late_shared is None:
            raise RuntimeError("Feature shared branch tidak lengkap.")
        gap_final = self.gap_late(late_shared)
        hbp_final = self.hbp_late(late_shared)
        return [middle, late_shared, hbp_final], gap_final


class DecoupledGAPHBPModel(nn.Module):
    """Shared-early, branched-late GAP-HBP with decoupled fusion gradients."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        projection_dim: int,
        dropout: float,
        pretrained: bool,
        fusion: str,
        gate_hidden: int,
        fusion_detach: bool,
    ):
        super().__init__()
        if fusion not in {"fixed", "learned"}:
            raise ValueError("dual_fusion harus 'fixed' atau 'learned'.")
        if gate_hidden <= 0:
            raise ValueError("dual_gate_hidden harus lebih besar dari nol.")
        self.backbone_name = backbone
        self.head = f"decoupled_gap_hbp_{fusion}"
        self.fusion = fusion
        self.fusion_detach = bool(fusion_detach)
        self.encoder = DecoupledMobileNetV3Encoder(backbone, pretrained)
        channels = self.encoder.channels
        self.gap_pool = GAPHead(channels[-1])
        self.hbp_pool = HierarchicalBilinearPooling(channels, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gap_classifier = nn.Linear(self.gap_pool.output_dim, num_classes)
        self.hbp_classifier = nn.Linear(self.hbp_pool.output_dim, num_classes)
        self.gate: nn.Sequential | None = None
        if fusion == "learned":
            # Gate initialization must not change branch initialization or the
            # subsequent DataLoader RNG stream relative to the fixed control.
            rng_state = torch.random.get_rng_state()
            self.gate = nn.Sequential(
                nn.Linear(num_classes * 2, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 2),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
            torch.random.set_rng_state(rng_state)

    def shared_parameters_for_audit(self) -> list[nn.Parameter]:
        return list(self.encoder.shared_parameters())

    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        if domain_strength is not None:
            raise ValueError("Decoupled GAP-HBP saat ini hanya untuk source_only.")
        hbp_features, gap_feature = self.encoder(x)
        gap_embedding = self.gap_pool([gap_feature])
        hbp_embedding = self.hbp_pool(hbp_features)
        gap_logits = self.gap_classifier(self.dropout(gap_embedding))
        hbp_logits = self.hbp_classifier(self.dropout(hbp_embedding))

        if self.gate is None:
            gate_weights = torch.full(
                (x.shape[0], 2),
                0.5,
                device=x.device,
                dtype=gap_logits.dtype,
            )
        else:
            gate_input = torch.cat((gap_logits.detach(), hbp_logits.detach()), dim=1)
            gate_weights = torch.softmax(self.gate(gate_input), dim=1)
        fusion_gap = gap_logits.detach() if self.fusion_detach else gap_logits
        fusion_hbp = hbp_logits.detach() if self.fusion_detach else hbp_logits
        logits = (
            gate_weights[:, :1] * fusion_gap
            + gate_weights[:, 1:] * fusion_hbp
        )
        return ModelOutput(
            logits=logits,
            embedding=torch.cat((gap_embedding, hbp_embedding), dim=1),
            expert_logits={"gap": gap_logits, "hbp": hbp_logits},
            gate_weights=gate_weights,
        )


class AdaptationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        head: str = "hbp",
        out_indices: tuple[int, ...] = (1, 3, 4),
        projection_dim: int = 512,
        mpncov_reduction_dim: int = 128,
        mpncov_iterations: int = 5,
        mpncov_epsilon: float = 1.0e-5,
        fb_rank: int = 20,
        fb_dropfactor_keep_prob: float = 0.5,
        hbp_spatial_size: int = 14,
        bilinear_output_dim: int | None = None,
        hbp_mlp_dim: int = 672,
        fusion_hbp_dim: int = 512,
        fusion_gap_dim: int = 256,
        residual_control_dim: int = 80,
        residual_gap_dim: int = 128,
        attention_reduction: int = 16,
        capacity_hidden_dim: int = 1259,
        mde_branch_channels: int = 16,
        mde_control_hidden_dim: int = 287,
        moe_local_dim: int = 256,
        moe_gate_hidden: int = 32,
        moe_hbp_prior: float = 0.8,
        dropout: float = 0.2,
        classifier: str = "linear",
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.3,
        arpl_weight: float = 0.1,
        arpl_margin: float = 1.0,
        dsconv_enabled: bool = False,
        dsconv_bits: int = 4,
        dsconv_block_size: int = 128,
        dsconv_stage_prefixes: tuple[str, ...] = (
            "blocks.0",
            "blocks.1",
            "blocks.2",
        ),
        pretrained: bool = True,
        enable_domain_classifier: bool = False,
        hierarchy_num_parents: int = 0,
    ):
        super().__init__()
        supported_heads = {
            "gap",
            "mpncov",
            "factorized_linear_conv_control",
            "factorized_bilinear_conv",
            "hierarchical_gap",
            "sppf_attention_gap",
            "multiscale_defect_gap",
            "capacity_residual_gap",
            "bilinear",
            "hbp",
            "hbp_linear",
            "hbp_moe",
            "sp_hbp",
            "sppf_attention_hbp",
            "capacity_residual_hbp",
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
            "hbp_linear",
            "hbp_moe",
            "sp_hbp",
            "sppf_attention_hbp",
            "capacity_residual_hbp",
            "hbp_mlp",
            "gap_hbp_fusion",
            "hbp_residual_control",
            "gap_hbp_residual",
        }
        hierarchical_heads = hbp_heads | {"hierarchical_gap"}
        if head in hierarchical_heads and len(out_indices) != 3:
            raise ValueError(
                "out_indices untuk head hierarkis harus tepat 3 indeks."
            )

        self.backbone_name = backbone
        self.head = head
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.dsconv_replaced_layers: list[str] = []
        if dsconv_enabled:
            self.dsconv_replaced_layers = replace_spatial_convolutions_with_dsconv(
                self.encoder,
                stage_prefixes=dsconv_stage_prefixes,
                bits=dsconv_bits,
                block_size=dsconv_block_size,
            )
        channels = list(self.encoder.feature_info.channels())
        if head in {"hbp", "hbp_moe"}:
            self.pool = HierarchicalBilinearPooling(channels, projection_dim)
        elif head == "mpncov":
            self.pool = MatrixPowerNormalizedCovariancePooling(
                channels[-1],
                reduction_dim=mpncov_reduction_dim,
                iterations=mpncov_iterations,
                epsilon=mpncov_epsilon,
            )
        elif head == "hierarchical_gap":
            self.pool = ProjectedHierarchicalGAP(channels, projection_dim)
        elif head == "hbp_linear":
            self.pool = LinearProjectionHBP(
                channels,
                projection_dim,
            )
        elif head == "sppf_attention_gap":
            self.pool = SPPFAttentionGAP(
                channels[-1],
                attention_reduction,
            )
        elif head == "multiscale_defect_gap":
            self.pool = MultiScaleDefectGAP(
                channels[-1],
                mde_branch_channels,
            )
        elif head == "capacity_residual_gap":
            self.pool = CapacityResidualGAP(
                channels[-1],
                mde_control_hidden_dim,
            )
        elif head == "sppf_attention_hbp":
            self.pool = SPPFAttentionHBP(
                channels,
                projection_dim,
                attention_reduction,
            )
        elif head == "capacity_residual_hbp":
            self.pool = CapacityResidualHBP(
                channels,
                projection_dim,
                capacity_hidden_dim,
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
        self.direct_classifier: FactorizedBilinearConvClassifier | None = None
        if head in {
            "factorized_linear_conv_control",
            "factorized_bilinear_conv",
        }:
            if classifier != "linear":
                raise ValueError("Conv-FBN hanya mendukung classifier linear langsung.")
            self.direct_classifier = FactorizedBilinearConvClassifier(
                channels[-1],
                num_classes,
                rank=fb_rank,
                keep_prob=fb_dropfactor_keep_prob,
                quadratic=head == "factorized_bilinear_conv",
            )
            self.classifier = nn.Identity()
            self.classifier_type = "direct"
        elif classifier == "linear":
            self.classifier = nn.Linear(self.pool.output_dim, num_classes)
        elif classifier == "arcface":
            self.classifier = ArcMarginClassifier(
                self.pool.output_dim,
                num_classes,
                scale=arcface_scale,
                margin=arcface_margin,
            )
        elif classifier == "arpl":
            self.classifier = AdversarialReciprocalPointClassifier(
                self.pool.output_dim,
                num_classes,
                regularization_weight=arpl_weight,
                margin=arpl_margin,
            )
        else:
            raise ValueError(
                "model.classifier harus 'linear', 'arcface', atau 'arpl'."
            )
        if self.direct_classifier is None:
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
        open_set_loss = None
        if self.direct_classifier is not None:
            classifier_embedding = embedding
            logits = self.direct_classifier(features[-1])
        elif self.classifier_type == "arcface":
            logits = self.classifier(embedding, labels)
            classifier_embedding = embedding
        else:
            classifier_embedding = self.dropout(embedding)
            logits = self.classifier(classifier_embedding)
            if self.classifier_type == "arpl" and labels is not None:
                open_set_loss = self.classifier.open_space_regularization(
                    classifier_embedding,
                    labels,
                )
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
            open_set_loss=open_set_loss,
        )

    def slow_start_parameters(self):
        """Parameters requiring the paper's FB-layer slow-start schedule."""
        if self.direct_classifier is None:
            return ()
        return tuple(self.direct_classifier.parameters())


def build_model(cfg: dict) -> nn.Module:
    head = cfg.get("head", "hbp")
    if head in {
        "swin_gap",
        "swin_hsfpn",
        "swin_sam",
        "swin_hssam",
    }:
        from bilinear_lmmd.modeling.swin_hssam import SwinHSSAMClassifier

        if str(cfg.get("classifier", "linear")) != "linear":
            raise ValueError("Reproduksi Swin-HSSAM hanya mendukung classifier linear.")
        return SwinHSSAMClassifier(
            backbone=cfg["backbone"],
            num_classes=int(cfg["num_classes"]),
            out_indices=tuple(cfg.get("out_indices", (1, 2, 3))),
            hsfpn_channels=int(cfg.get("hsfpn_channels", 256)),
            sam_hidden_dim=int(cfg.get("sam_hidden_dim", 128)),
            attention_reduction=int(cfg.get("attention_reduction", 16)),
            dropout=float(cfg.get("dropout", 0.2)),
            pretrained=bool(cfg.get("pretrained", True)),
            use_hsfpn=head in {"swin_hsfpn", "swin_hssam"},
            use_sam=head in {"swin_sam", "swin_hssam"},
        )
    if head in {
        "sni_multiresolution_flat",
        "sni_flat_residual_gap",
        "sni_flat_residual_hbp",
        "sni_mre_ontology_gap",
        "sni_mrenet",
    }:
        from bilinear_lmmd.modeling.sni_mrenet import (
            SNIMultiResolutionExpertModel,
        )

        if str(cfg.get("classifier", "linear")) != "linear":
            raise ValueError("SNI-MRENet hanya mendukung classifier linear.")
        mode = {
            "sni_multiresolution_flat": "flat",
            "sni_flat_residual_gap": "flat_residual_gap",
            "sni_flat_residual_hbp": "flat_residual_hbp",
            "sni_mre_ontology_gap": "ontology_gap",
            "sni_mrenet": "ontology_hbp",
        }[head]
        return SNIMultiResolutionExpertModel(
            backbone=cfg["backbone"],
            num_classes=int(cfg["num_classes"]),
            out_indices=tuple(cfg.get("out_indices", (1, 2, 3, 4))),
            feature_dim=int(cfg.get("sni_feature_dim", 128)),
            projection_dim=int(cfg.get("projection_dim", 128)),
            dropout=float(cfg.get("dropout", 0.2)),
            pretrained=bool(cfg.get("pretrained", True)),
            mode=mode,
        )
    if head in {
        "progressive_multigranularity",
        "progressive_multigranularity_consistency",
    }:
        from bilinear_lmmd.modeling.progressive_multigranularity import (
            ProgressiveMultiGranularityModel,
        )

        if str(cfg.get("classifier", "linear")) != "linear":
            raise ValueError(
                "Progressive multi-granularity hanya mendukung classifier linear."
            )
        return ProgressiveMultiGranularityModel(
            backbone=cfg["backbone"],
            num_classes=int(cfg["num_classes"]),
            out_indices=tuple(cfg.get("out_indices", (1, 3, 4))),
            feature_dim=int(cfg.get("pmg_feature_dim", 256)),
            branch_dim=int(cfg.get("pmg_branch_dim", 512)),
            dropout=float(cfg.get("dropout", 0.2)),
            pretrained=bool(cfg.get("pretrained", True)),
            category_consistency=head.endswith("_consistency"),
        )
    if head in {"decoupled_gap_hbp_fixed", "decoupled_gap_hbp_learned"}:
        if str(cfg.get("classifier", "linear")) != "linear":
            raise ValueError("Decoupled GAP-HBP hanya mendukung classifier linear.")
        return DecoupledGAPHBPModel(
            backbone=cfg["backbone"],
            num_classes=int(cfg["num_classes"]),
            projection_dim=int(cfg.get("projection_dim", 512)),
            dropout=float(cfg.get("dropout", 0.2)),
            pretrained=bool(cfg.get("pretrained", True)),
            fusion=("fixed" if head.endswith("fixed") else "learned"),
            gate_hidden=int(cfg.get("dual_gate_hidden", 32)),
            fusion_detach=bool(cfg.get("dual_fusion_detach", True)),
        )
    return AdaptationModel(
        backbone=cfg["backbone"],
        num_classes=int(cfg["num_classes"]),
        head=head,
        out_indices=tuple(cfg.get("out_indices", (1, 3, 4))),
        projection_dim=int(cfg.get("projection_dim", 512)),
        mpncov_reduction_dim=int(cfg.get("mpncov_reduction_dim", 128)),
        mpncov_iterations=int(cfg.get("mpncov_iterations", 5)),
        mpncov_epsilon=float(cfg.get("mpncov_epsilon", 1.0e-5)),
        fb_rank=int(cfg.get("fb_rank", 20)),
        fb_dropfactor_keep_prob=float(
            cfg.get("fb_dropfactor_keep_prob", 0.5)
        ),
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
        capacity_hidden_dim=int(cfg.get("capacity_hidden_dim", 1259)),
        mde_branch_channels=int(cfg.get("mde_branch_channels", 16)),
        mde_control_hidden_dim=int(cfg.get("mde_control_hidden_dim", 287)),
        moe_local_dim=int(cfg.get("moe_local_dim", 256)),
        moe_gate_hidden=int(cfg.get("moe_gate_hidden", 32)),
        moe_hbp_prior=float(cfg.get("moe_hbp_prior", 0.8)),
        dropout=float(cfg.get("dropout", 0.2)),
        classifier=str(cfg.get("classifier", "linear")),
        arcface_scale=float(cfg.get("arcface_scale", 30.0)),
        arcface_margin=float(cfg.get("arcface_margin", 0.3)),
        arpl_weight=float(cfg.get("arpl_weight", 0.1)),
        arpl_margin=float(cfg.get("arpl_margin", 1.0)),
        dsconv_enabled=bool(cfg.get("dsconv_enabled", False)),
        dsconv_bits=int(cfg.get("dsconv_bits", 4)),
        dsconv_block_size=int(cfg.get("dsconv_block_size", 128)),
        dsconv_stage_prefixes=tuple(
            cfg.get(
                "dsconv_stage_prefixes",
                ("blocks.0", "blocks.1", "blocks.2"),
            )
        ),
        pretrained=bool(cfg.get("pretrained", True)),
        enable_domain_classifier=bool(cfg.get("enable_domain_classifier", False)),
        hierarchy_num_parents=int(cfg.get("hierarchy_num_parents", 0)),
    )
