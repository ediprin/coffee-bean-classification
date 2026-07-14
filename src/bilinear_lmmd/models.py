from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class ModelOutput:
    logits: Tensor
    embedding: Tensor
    domain_logits: Tensor | None = None


class AdaptationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        head: str = "hbp",
        out_indices: tuple[int, ...] = (1, 3, 4),
        projection_dim: int = 512,
        bilinear_output_dim: int | None = None,
        dropout: float = 0.2,
        pretrained: bool = True,
        enable_domain_classifier: bool = False,
    ):
        super().__init__()
        if head not in {"gap", "bilinear", "hbp"}:
            raise ValueError("head harus 'gap', 'bilinear', atau 'hbp'.")
        if head == "hbp" and len(out_indices) != 3:
            raise ValueError("out_indices untuk HBP harus berisi tepat 3 indeks.")

        self.backbone_name = backbone
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        if head == "hbp":
            self.pool = HierarchicalBilinearPooling(channels, projection_dim)
        elif head == "bilinear":
            self.pool = FactorizedBilinearPooling(
                channels[-1], projection_dim, bilinear_output_dim
            )
        else:
            self.pool = GAPHead(channels[-1])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.pool.output_dim, num_classes)
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

    def forward(self, x: Tensor, domain_strength: float | None = None) -> ModelOutput:
        embedding = self.pool(self.encoder(x))
        logits = self.classifier(self.dropout(embedding))
        domain_logits = None
        if domain_strength is not None:
            if self.domain_classifier is None:
                raise RuntimeError(
                    "DANN membutuhkan model.enable_domain_classifier=true."
                )
            domain_logits = self.domain_classifier(
                gradient_reverse(embedding, domain_strength)
            )
        return ModelOutput(logits=logits, embedding=embedding, domain_logits=domain_logits)


def build_model(cfg: dict) -> AdaptationModel:
    return AdaptationModel(
        backbone=cfg["backbone"],
        num_classes=int(cfg["num_classes"]),
        head=cfg.get("head", "hbp"),
        out_indices=tuple(cfg.get("out_indices", (1, 3, 4))),
        projection_dim=int(cfg.get("projection_dim", 512)),
        bilinear_output_dim=(
            int(cfg["bilinear_output_dim"])
            if cfg.get("bilinear_output_dim") is not None
            else None
        ),
        dropout=float(cfg.get("dropout", 0.2)),
        pretrained=bool(cfg.get("pretrained", True)),
        enable_domain_classifier=bool(cfg.get("enable_domain_classifier", False)),
    )
