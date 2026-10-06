from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Paket 'timm' diperlukan.") from exc

from bilinear_lmmd.modeling.frequency_screening import haar_dwt2d
from bilinear_lmmd.modeling.models import (
    GAPHead,
    HierarchicalBilinearPooling,
    ModelOutput,
    build_model,
)


REPRESENTATION_CANDIDATES = {
    "B0": {"kind": "anchor_gap", "anchor": "B0", "loss": "ce"},
    "B1": {"kind": "anchor_hbp", "anchor": "B1", "loss": "ce"},
    "R1": {"kind": "mfr_gap", "anchor": "B0", "loss": "ce"},
    "R2": {"kind": "mfr_hbp", "anchor": "B1", "loss": "ce"},
    "R3": {"kind": "lrbp_i", "anchor": "B1", "loss": "lrbp_hinge"},
    "R4": {"kind": "adaptive_multistage", "anchor": "B1", "loss": "ce"},
    "R5": {"kind": "hbp_fusion_loss", "anchor": "B1", "loss": "ce_focal"},
    "W7": {"kind": "dwt_dualdomain", "anchor": "B0", "loss": "ce"},
    "F1": {"kind": "ffc_late", "anchor": "B0", "loss": "ce"},
    "F2": {"kind": "global_filter_late", "anchor": "B0", "loss": "ce"},
}


class MixedFeatureRecalibration(nn.Module):
    """MFR from Huang et al. (CEA 2022).

    The source paper specifies lambda=8, alpha=0.8 and rho=0.2. Its printed
    W5 shape is dimensionally inconsistent with the concatenation in Eq. (2)
    and Fig. 6; this implementation follows the computational graph in Fig. 6,
    mapping the concatenated 2N/lambda average-path descriptor back to N.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        alpha: float = 0.8,
        rho: float = 0.2,
    ):
        super().__init__()
        hidden = max(1, channels // reduction)
        max_hidden = max(1, 2 * channels // reduction)
        self.avg_w1 = nn.Linear(channels, hidden, bias=False)
        self.avg_w2 = nn.Linear(channels, hidden, bias=False)
        self.avg_out = nn.Linear(hidden * 2, channels, bias=False)
        self.max_w3 = nn.Linear(channels, max_hidden, bias=False)
        self.max_out = nn.Linear(max_hidden, channels, bias=False)
        self.alpha = float(alpha)
        self.rho = float(rho)

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=(-2, -1))
        maximum = x.amax(dim=(-2, -1))
        avg_gate = torch.sigmoid(
            self.avg_out(
                F.relu(
                    torch.cat((self.avg_w1(avg), self.avg_w2(avg)), dim=1),
                    inplace=False,
                )
            )
        )
        max_gate = torch.sigmoid(
            self.max_out(F.relu(self.max_w3(maximum), inplace=False))
        )
        avg_gate = avg_gate.unsqueeze(-1).unsqueeze(-1)
        max_gate = max_gate.unsqueeze(-1).unsqueeze(-1)
        return self.alpha * (x * avg_gate) + self.rho * (x * max_gate)


class RecalibratedClassifier(nn.Module):
    """Late MFR adaptation with exact GAP/HBP shared-core initialization."""

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        head: str,
        projection_dim: int,
        dropout: float,
        pretrained: bool,
        reduction: int,
        alpha: float,
        rho: float,
    ):
        super().__init__()
        if head not in {"gap", "hbp"}:
            raise ValueError("MFR head harus gap atau hbp.")
        out_indices = (4,) if head == "gap" else (1, 3, 4)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        self.pool = (
            GAPHead(channels[-1])
            if head == "gap"
            else HierarchicalBilinearPooling(channels, projection_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.pool.output_dim, num_classes)

        rng_state = torch.random.get_rng_state()
        self.recalibration = MixedFeatureRecalibration(
            channels[-1],
            reduction=reduction,
            alpha=alpha,
            rho=rho,
        )
        torch.random.set_rng_state(rng_state)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        features = list(self.encoder(x))
        features[-1] = self.recalibration(features[-1])
        embedding = self.pool(features)
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)


class AdaptiveMultiStageClassifier(nn.Module):
    """MobileNet adaptation of MFSwin's adaptive multi-stage fusion principle."""

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        fusion_dim: int,
        dropout: float,
        pretrained: bool,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 3, 4),
        )
        channels = list(self.encoder.feature_info.channels())
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, fusion_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fusion_dim),
                    nn.ReLU(inplace=True),
                )
                for channel in channels
            ]
        )
        self.stage_gate = nn.Linear(fusion_dim * 3, 3)
        self.refine = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        features = list(self.encoder(x))
        target_hw = features[-1].shape[-2:]
        projected = []
        descriptors = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_hw:
                feature = F.adaptive_avg_pool2d(feature, target_hw)
            projected.append(feature)
            descriptors.append(feature.mean(dim=(-2, -1)))
        weights = torch.softmax(
            self.stage_gate(torch.cat(descriptors, dim=1)),
            dim=1,
        )
        fused = sum(
            feature * weights[:, index].view(-1, 1, 1, 1)
            for index, feature in enumerate(projected)
        )
        fused = self.refine(fused)
        embedding = fused.mean(dim=(-2, -1))
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding, gate_weights=weights)


class LowRankBilinearClassifier(nn.Module):
    """LRBP-I/co-decomposition style classifier adapted to MobileNet features.

    A shared 1x1 projection P reduces C -> m. Each class owns positive and
    negative rank-r/2 factors. Scores use the difference of squared Frobenius
    norms from Kong & Fowlkes Eq. (8), avoiding an explicit m^2 bilinear map.
    """

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        reduced_dim: int = 100,
        rank: int = 8,
        pretrained: bool = True,
    ):
        super().__init__()
        if rank <= 0 or rank % 2:
            raise ValueError("LRBP rank harus positif dan genap.")
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        channels = int(self.encoder.feature_info.channels()[-1])
        self.projection = nn.Conv2d(
            channels, reduced_dim, kernel_size=1, bias=False
        )
        half_rank = rank // 2
        self.u_plus = nn.Parameter(
            0.02 * torch.randn(num_classes, reduced_dim, half_rank)
        )
        self.u_minus = nn.Parameter(
            0.02 * torch.randn(num_classes, reduced_dim, half_rank)
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.reduced_dim = int(reduced_dim)
        self.rank = int(rank)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        feature = self.encoder(x)[-1]
        projected = self.projection(feature).flatten(2)
        positive = torch.einsum("cmr,bms->bcrs", self.u_plus, projected)
        negative = torch.einsum("cmr,bms->bcrs", self.u_minus, projected)
        logits = (
            positive.square().sum(dim=(-2, -1))
            - negative.square().sum(dim=(-2, -1))
            + self.bias
        )
        embedding = projected.mean(dim=2)
        return ModelOutput(logits=logits, embedding=embedding)

    def frobenius_regularizer(self) -> Tensor:
        plus_gram = torch.matmul(self.u_plus.transpose(1, 2), self.u_plus)
        minus_gram = torch.matmul(self.u_minus.transpose(1, 2), self.u_minus)
        cross = torch.matmul(self.u_plus.transpose(1, 2), self.u_minus)
        return (
            plus_gram.square().sum(dim=(-2, -1))
            + minus_gram.square().sum(dim=(-2, -1))
            + cross.square().sum(dim=(-2, -1))
        ).mean()


class DWTDualDomainClassifier(nn.Module):
    """Lightweight DWTFormer-inspired spatial/frequency complementarity arm."""

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        fusion_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        channels = int(self.encoder.feature_info.channels()[-1])
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.frequency = nn.Sequential(
            nn.Conv2d(channels * 4, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        feature = self.encoder(x)[-1]
        spatial = self.spatial(feature)
        bands, _ = haar_dwt2d(feature)
        frequency = self.frequency(torch.cat(bands, dim=1))
        frequency = F.interpolate(
            frequency,
            size=spatial.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fuse(torch.cat((spatial, frequency), dim=1))
        embedding = fused.mean(dim=(-2, -1))
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)


class FourierUnit(nn.Module):
    """Real-compatible Fourier Unit following FFC's FFT/1x1/IFFT pattern."""

    def __init__(self, channels: int):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x, norm="ortho")
        stacked = torch.cat((spectrum.real, spectrum.imag), dim=1)
        stacked = self.spectral(stacked)
        real, imag = stacked.chunk(2, dim=1)
        updated = torch.complex(real, imag)
        return torch.fft.irfft2(updated, s=(height, width), norm="ortho")


class FFCLateBlock(nn.Module):
    """One late Fast-Fourier-Convolution block with local/global channel paths."""

    def __init__(self, channels: int, alpha: float = 0.5):
        super().__init__()
        global_channels = int(round(channels * alpha))
        local_channels = channels - global_channels
        if not local_channels or not global_channels:
            raise ValueError("FFC alpha harus menghasilkan local dan global channels.")
        self.local_channels = local_channels
        self.global_channels = global_channels
        self.local_local = nn.Conv2d(
            local_channels, local_channels, 3, padding=1, bias=False
        )
        self.global_local = nn.Conv2d(
            global_channels, local_channels, 1, bias=False
        )
        self.local_global = nn.Conv2d(
            local_channels, global_channels, 1, bias=False
        )
        self.global_global = FourierUnit(global_channels)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        local, global_ = torch.split(
            x, (self.local_channels, self.global_channels), dim=1
        )
        out_local = self.local_local(local) + self.global_local(global_)
        out_global = self.local_global(local) + self.global_global(global_)
        return self.act(self.norm(torch.cat((out_local, out_global), dim=1)))


class FFCLateClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        alpha: float,
        dropout: float,
        pretrained: bool,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        channels = int(self.encoder.feature_info.channels()[-1])
        self.ffc = FFCLateBlock(channels, alpha=alpha)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        feature = self.ffc(self.encoder(x)[-1])
        embedding = feature.mean(dim=(-2, -1))
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)


class GlobalFilter2D(nn.Module):
    """GFNet-style learnable complex global filter on a fixed 7x7 late grid."""

    def __init__(self, channels: int, height: int = 7, width: int = 7):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.weight = nn.Parameter(
            0.02 * torch.randn(channels, height, width // 2 + 1, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != (self.height, self.width):
            raise RuntimeError(
                f"Global filter mengharapkan {self.height}x{self.width}, "
                f"didapat {tuple(x.shape[-2:])}."
            )
        spectrum = torch.fft.rfft2(x, norm="ortho")
        weight = torch.view_as_complex(self.weight.contiguous())
        filtered = spectrum * weight.unsqueeze(0)
        return torch.fft.irfft2(
            filtered, s=(self.height, self.width), norm="ortho"
        )


class GlobalFilterLateClassifier(nn.Module):
    """Late CNN adaptation of GFNet's learnable FFT-domain global filter."""

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        dropout: float,
        pretrained: bool,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        channels = int(self.encoder.feature_info.channels()[-1])
        self.norm = nn.BatchNorm2d(channels)
        self.global_filter = GlobalFilter2D(channels, 7, 7)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        feature = self.encoder(x)[-1]
        feature = feature + self.global_filter(self.norm(feature))
        embedding = feature.mean(dim=(-2, -1))
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)


def build_representation_screening_model(candidate: str, cfg: dict) -> nn.Module:
    if candidate not in REPRESENTATION_CANDIDATES:
        raise ValueError(
            f"candidate harus salah satu {sorted(REPRESENTATION_CANDIDATES)}."
        )
    model_cfg = cfg["model"]
    rep = cfg.get("representation", {})
    kind = REPRESENTATION_CANDIDATES[candidate]["kind"]

    if kind in {"anchor_gap", "anchor_hbp", "hbp_fusion_loss"}:
        anchor_cfg = dict(model_cfg)
        head = "gap" if kind == "anchor_gap" else "hbp"
        anchor_cfg["head"] = head
        anchor_cfg["out_indices"] = [4] if head == "gap" else [1, 3, 4]
        return build_model(anchor_cfg)

    if kind in {"mfr_gap", "mfr_hbp"}:
        return RecalibratedClassifier(
            backbone=model_cfg["backbone"],
            num_classes=int(model_cfg["num_classes"]),
            head="gap" if kind.endswith("gap") else "hbp",
            projection_dim=int(model_cfg.get("projection_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            reduction=int(rep.get("mfr_reduction", 8)),
            alpha=float(rep.get("mfr_alpha", 0.8)),
            rho=float(rep.get("mfr_rho", 0.2)),
        )

    if kind == "lrbp_i":
        return LowRankBilinearClassifier(
            backbone=model_cfg["backbone"],
            num_classes=int(model_cfg["num_classes"]),
            reduced_dim=int(rep.get("lrbp_reduced_dim", 100)),
            rank=int(rep.get("lrbp_rank", 8)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    if kind == "adaptive_multistage":
        return AdaptiveMultiStageClassifier(
            backbone=model_cfg["backbone"],
            num_classes=int(model_cfg["num_classes"]),
            fusion_dim=int(rep.get("multistage_dim", 384)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    if kind == "dwt_dualdomain":
        return DWTDualDomainClassifier(
            backbone=model_cfg["backbone"],
            num_classes=int(model_cfg["num_classes"]),
            fusion_dim=int(rep.get("dualdomain_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    if kind == "ffc_late":
        return FFCLateClassifier(
            backbone=model_cfg["backbone"],
            num_classes=int(model_cfg["num_classes"]),
            alpha=float(rep.get("ffc_alpha", 0.5)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    if kind == "global_filter_late":
        return GlobalFilterLateClassifier(
            backbone=model_cfg["backbone"],
            num_classes=int(model_cfg["num_classes"]),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    raise RuntimeError(f"Builder belum tersedia untuk {candidate}/{kind}.")


def lrbp_hinge_loss(
    model: nn.Module,
    logits: Tensor,
    labels: Tensor,
    regularization_weight: float,
) -> Tensor:
    if not isinstance(model, LowRankBilinearClassifier):
        raise TypeError("lrbp_hinge_loss membutuhkan LowRankBilinearClassifier.")
    targets = logits.new_full(logits.shape, -1.0)
    targets.scatter_(1, labels.view(-1, 1), 1.0)
    hinge = F.relu(1.0 - targets * logits).mean()
    return hinge + 0.5 * float(regularization_weight) * model.frobenius_regularizer()


def focal_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    gamma: float = 2.0,
) -> Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(gamma) * ce).mean()
