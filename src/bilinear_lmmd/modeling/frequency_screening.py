from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Paket 'timm' diperlukan.") from exc

from bilinear_lmmd.modeling.models import (
    GAPHead,
    HierarchicalBilinearPooling,
    ModelOutput,
    build_model,
)


FREQUENCY_CANDIDATES = {
    "B0": {"head": "gap", "module": None, "anchor": "B0"},
    "B1": {"head": "hbp", "module": None, "anchor": "B1"},
    "H384": {
        "head": "hbp",
        "module": None,
        "anchor": "H384",
        "projection_dim": 384,
    },
    "W1": {"head": "gap", "module": "wca", "anchor": "B0"},
    "W2": {"head": "hbp", "module": "wca", "anchor": "B1"},
    "W3": {"head": "gap", "module": "fda", "anchor": "B0"},
    "W4": {"head": "hbp", "module": "fda", "anchor": "B1"},
    "W5": {"head": "gap", "module": "fca_lf2", "anchor": "B0"},
    "W6": {"head": "hbp", "module": "fca_lf2", "anchor": "B1"},
    # Focused efficiency follow-up. C1/C2 test FReCSA-style frequency
    # regulation on the same late MobileNetV3 feature used by W3. C3/C4
    # use a preregistered compact HBP (p=384) and are compared only with
    # the matched H384 anchor, never directly as a one-factor test vs B1.
    "C1": {"head": "gap", "module": "frsa", "anchor": "B0"},
    "C2": {"head": "gap", "module": "frecsa", "anchor": "B0"},
    "C3": {
        "head": "hbp",
        "module": "fda",
        "anchor": "H384",
        "projection_dim": 384,
    },
    "C4": {
        "head": "hbp",
        "module": "frecsa",
        "anchor": "H384",
        "projection_dim": 384,
    },
}


def _pad_even(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (height, width)


def haar_dwt2d(x: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], tuple[int, int]]:
    """Orthonormal one-level Haar DWT on feature maps.

    Returns LL/LH/HL/HH and the original spatial size. Odd feature grids are
    replicate-padded by one pixel only for the transform; IWT crops back.
    """

    x, original_hw = _pad_even(x)
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (-a - b + c + d) * 0.5
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return (ll, lh, hl, hh), original_hw


def haar_iwt2d(
    bands: tuple[Tensor, Tensor, Tensor, Tensor],
    output_hw: tuple[int, int],
) -> Tensor:
    ll, lh, hl, hh = bands
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    batch, channels, height, width = ll.shape
    out = ll.new_empty(batch, channels, height * 2, width * 2)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out[..., : output_hw[0], : output_hw[1]]


class WaveletChannelAttention(nn.Module):
    """WCANet-style Haar-wavelet channel attention.

    Zhang et al. (KBS 2025) use the four DWT subbands to construct a richer
    channel descriptor while reweighting the original spatial feature map.
    A SENet-style two-layer FC mapper is used for the paper's FC attention
    operator.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        bands, _ = haar_dwt2d(x)
        descriptor = sum(band.mean(dim=(-2, -1)) for band in bands)
        weights = torch.sigmoid(self.fc(descriptor)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class FrequencyDomainAttention(nn.Module):
    """FdaNet-style adaptive weighting of LL/LH/HL/HH followed by Haar IWT.

    Each subband is compressed to one scalar through a learned 1x1 projection
    and GAP. The four descriptors are mapped to four sample-dependent weights.
    """

    def __init__(self, channels: int, hidden: int = 4):
        super().__init__()
        if hidden <= 0:
            raise ValueError("fda_hidden harus > 0.")
        self.compressors = nn.ModuleList(
            [nn.Conv2d(channels, 1, kernel_size=1, bias=False) for _ in range(4)]
        )
        self.fc = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        bands, original_hw = haar_dwt2d(x)
        descriptors = [
            compressor(band).mean(dim=(-2, -1))
            for compressor, band in zip(self.compressors, bands)
        ]
        descriptor = torch.cat(descriptors, dim=1)
        weights = torch.sigmoid(self.fc(descriptor))
        weighted = tuple(
            band * weights[:, index].view(-1, 1, 1, 1)
            for index, band in enumerate(bands)
        )
        return haar_iwt2d(weighted, original_hw)



class FrequencyRegulatedSpatialAttention(nn.Module):
    """Spatial branch of FReCSA (Zhuang et al., ESWA 2025).

    The paper forms a predefined high-pass response as x - AvgPool7x7(x),
    performs local-local interaction with the original feature, applies BN,
    ReLU with fixed bias 0.5, sigmoid, and multiplicative recalibration.

    This is a *late-feature adaptation* for MobileNetV3, not a reproduction
    of the paper's ResNet block-wise placement.
    """

    def __init__(self, channels: int, kernel_size: int = 7, bias: float = 0.5):
        super().__init__()
        if channels <= 0:
            raise ValueError("FRSA channels harus > 0.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("FRSA kernel_size harus ganjil dan > 0.")
        self.low_pass = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.register_buffer(
            "fixed_bias",
            torch.tensor(float(bias)),
            persistent=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        low = self.low_pass(x)
        high = x - low
        similarity = high * x
        activated = F.relu(self.norm(similarity) + self.fixed_bias, inplace=False)
        weights = torch.sigmoid(activated)
        return x * weights


class SimplifiedFrequencyChannelAttention(nn.Module):
    """Simplified channel branch of FReCSA.

    GAP -> BN -> independent per-channel scaling (zero-init) -> sigmoid.
    There are no learned cross-channel FC/Conv connections.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("FReCSA channel attention channels harus > 0.")
        self.norm = nn.BatchNorm2d(channels)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        descriptor = F.adaptive_avg_pool2d(x, 1)
        normalized = self.norm(descriptor)
        weights = torch.sigmoid(normalized * self.scale)
        return x * weights


class FrequencyRegulatedChannelSpatialAttention(nn.Module):
    """Full FReCSA channel -> frequency-regulated spatial recalibration."""

    def __init__(self, channels: int, kernel_size: int = 7, bias: float = 0.5):
        super().__init__()
        self.channel = SimplifiedFrequencyChannelAttention(channels)
        self.spatial = FrequencyRegulatedSpatialAttention(
            channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.channel(x))

def _dct_basis(
    height: int,
    width: int,
    u: int,
    v: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    basis_y = torch.cos(math.pi * (2.0 * y + 1.0) * float(u) / (2.0 * height))
    basis_x = torch.cos(math.pi * (2.0 * x + 1.0) * float(v) / (2.0 * width))
    alpha_u = math.sqrt(1.0 / height) if u == 0 else math.sqrt(2.0 / height)
    alpha_v = math.sqrt(1.0 / width) if v == 0 else math.sqrt(2.0 / width)
    return (alpha_u * alpha_v) * basis_y[:, None] * basis_x[None, :]


class FcaLowFrequencyAttention(nn.Module):
    """FcaNet-LF with the paper-validated K=2 low-frequency setting.

    The channels are split into two groups and compressed with DCT (0,0) and
    (0,1), respectively. This intentionally avoids dataset-specific TS/NAS
    frequency searching on the reused Coffee17 development folds.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 2:
            raise ValueError("FcaNet membutuhkan minimal dua channel.")
        hidden = max(1, channels // reduction)
        self.channels = int(channels)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        split = channels // 2
        groups = (x[:, :split], x[:, split:])
        indices = ((0, 0), (0, 1))
        descriptors = []
        for group, (u, v) in zip(groups, indices):
            basis = _dct_basis(
                height,
                width,
                u,
                v,
                device=x.device,
                dtype=x.dtype,
            )
            descriptors.append((group * basis).sum(dim=(-2, -1)))
        descriptor = torch.cat(descriptors, dim=1)
        if descriptor.shape[1] != self.channels:
            raise RuntimeError("Dimensi descriptor FcaNet tidak cocok.")
        weights = torch.sigmoid(self.fc(descriptor)).view(batch, channels, 1, 1)
        return x * weights


class FrequencyRecalibratedClassifier(nn.Module):
    """MobileNetV3 classifier with a paper-grounded late frequency module.

    For HBP candidates only the deepest selected feature is recalibrated.
    Shallow/mid HBP stages remain unchanged, making W2/W4/W6 a one-factor
    adaptation relative to B1 rather than an all-stage architecture search.

    Shared encoder/pool/classifier modules are constructed before the
    candidate-only frequency module. The RNG state is restored after
    candidate-module construction, preserving the baseline RNG stream.
    """

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        head: str,
        module: str,
        projection_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        reduction: int = 16,
        fda_hidden: int = 4,
    ):
        super().__init__()
        if head not in {"gap", "hbp"}:
            raise ValueError("head frequency screening harus gap atau hbp.")
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
        self.head = head
        self.module_name = module

        rng_state = torch.random.get_rng_state()
        if module == "wca":
            self.feature_module = WaveletChannelAttention(
                channels[-1], reduction=reduction
            )
        elif module == "fda":
            self.feature_module = FrequencyDomainAttention(
                channels[-1], hidden=fda_hidden
            )
        elif module == "fca_lf2":
            self.feature_module = FcaLowFrequencyAttention(
                channels[-1], reduction=reduction
            )
        elif module == "frsa":
            self.feature_module = FrequencyRegulatedSpatialAttention(
                channels[-1], kernel_size=7, bias=0.5
            )
        elif module == "frecsa":
            self.feature_module = FrequencyRegulatedChannelSpatialAttention(
                channels[-1], kernel_size=7, bias=0.5
            )
        else:
            raise ValueError(f"Frequency module tidak dikenal: {module}")
        torch.random.set_rng_state(rng_state)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> ModelOutput:
        del labels
        features = list(self.encoder(x))
        features[-1] = self.feature_module(features[-1])
        embedding = self.pool(features)
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)


def build_frequency_screening_model(candidate: str, cfg: dict) -> nn.Module:
    if candidate not in FREQUENCY_CANDIDATES:
        raise ValueError(
            f"candidate harus salah satu {sorted(FREQUENCY_CANDIDATES)}."
        )
    spec = FREQUENCY_CANDIDATES[candidate]
    model_cfg = cfg["model"]
    projection_dim = int(spec.get("projection_dim", model_cfg.get("projection_dim", 512)))
    if spec["module"] is None:
        anchor_cfg = dict(model_cfg)
        anchor_cfg["head"] = spec["head"]
        anchor_cfg["out_indices"] = [4] if spec["head"] == "gap" else [1, 3, 4]
        anchor_cfg["projection_dim"] = projection_dim
        return build_model(anchor_cfg)
    frequency_cfg = cfg.get("frequency", {})
    return FrequencyRecalibratedClassifier(
        backbone=model_cfg["backbone"],
        num_classes=int(model_cfg["num_classes"]),
        head=spec["head"],
        module=spec["module"],
        projection_dim=projection_dim,
        dropout=float(model_cfg.get("dropout", 0.2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        reduction=int(frequency_cfg.get("attention_reduction", 16)),
        fda_hidden=int(frequency_cfg.get("fda_hidden", 4)),
    )


def shared_core_state(model: nn.Module) -> dict[str, Tensor]:
    """Return only encoder/pool/classifier tensors for matched-init checks."""

    allowed = ("encoder.", "pool.", "classifier.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(allowed)
    }


def assert_shared_core_equal(anchor: nn.Module, candidate: nn.Module) -> None:
    left = shared_core_state(anchor)
    right = shared_core_state(candidate)
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())
        missing_right = sorted(left.keys() - right.keys())
        raise RuntimeError(
            "Shared-core keys berbeda. "
            f"anchor_missing={missing_left}, candidate_missing={missing_right}"
        )
    for key in left:
        if not torch.equal(left[key], right[key]):
            raise RuntimeError(f"Shared-core initialization berbeda pada {key}.")
