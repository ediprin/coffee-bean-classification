from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Paket 'timm' diperlukan.") from exc

from bilinear_lmmd.data.preprocessing.wav1 import (
    haar_dwt2,
    soft_threshold,
    visushrink_threshold,
)
from bilinear_lmmd.modeling.models import HierarchicalBilinearPooling, ModelOutput


def luminance_l1_visushrink_details(raw_rgb: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Return thresholded L1 Haar LH/HL/HH maps from RGB input in [0, 1].

    The branch deliberately keeps only one luminance wavelet decomposition.
    RGB remains untouched in the main MobileNetV3 path.  This follows the
    Coffee17 cue audit: detail survivability after VisuShrink carried the
    coherent W0 association, while the three RGB retained-ratio cues were
    highly redundant.
    """

    if raw_rgb.ndim != 4 or raw_rgb.shape[1] != 3:
        raise ValueError("Wavelet branch membutuhkan BCHW RGB.")
    if not torch.is_floating_point(raw_rgb):
        raise TypeError("Wavelet branch membutuhkan tensor floating point.")
    if float(raw_rgb.detach().min()) < -1e-6 or float(raw_rgb.detach().max()) > 1.0 + 1e-6:
        raise ValueError("Wavelet branch membutuhkan RGB [0,1].")

    weights = raw_rgb.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    luminance = (raw_rgb * weights).sum(dim=1, keepdim=True)

    bands, _ = haar_dwt2(luminance)
    details = bands[:, :, 1:]
    threshold = visushrink_threshold(details, eps=eps)
    retained = soft_threshold(details, threshold)

    # B x 1 x 3 x H x W -> B x 3 x H x W (LH, HL, HH)
    return retained[:, 0]


class WaveletResidualHBPModel(nn.Module):
    """Raw-RGB HBP with a tiny L1-wavelet residual injected at the shallow stage.

    Core initialization order intentionally matches AdaptationModel(HBP):
    encoder -> HBP -> dropout -> linear classifier.  The wavelet branch is
    constructed only afterwards, so same-seed control/candidate core weights
    can be verified tensor-for-tensor before training.

    Candidate representation:
        F_s' = F_s + tanh(alpha) * P(D_L1)
        z    = HBP(F_s', F_m, F_d)

    alpha is initialized to exactly zero.  Therefore the candidate starts from
    the same functional RGB-HBP core while retaining a learnable route for the
    explicit wavelet detail signal.
    """

    def __init__(
        self,
        *,
        backbone: str = "mobilenetv3_large_100",
        num_classes: int = 17,
        out_indices: tuple[int, ...] = (1, 3, 4),
        projection_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        branch_hidden_channels: int = 16,
        wavelet_eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if len(out_indices) != 3:
            raise ValueError("WaveletResidualHBP membutuhkan tepat 3 out_indices.")
        if branch_hidden_channels <= 0:
            raise ValueError("branch_hidden_channels harus positif.")
        if wavelet_eps <= 0:
            raise ValueError("wavelet_eps harus positif.")

        self.backbone_name = backbone
        self.head = "wavelet_residual_hbp"
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        self.pool = HierarchicalBilinearPooling(channels, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.pool.output_dim, num_classes)
        self.classifier_type = "linear"

        # Extra candidate-only branch. Constructed after the shared core.
        self.wavelet_branch = nn.Sequential(
            nn.Conv2d(
                3,
                branch_hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                branch_hidden_channels,
                channels[0],
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
        )
        self.wavelet_gate = nn.Parameter(torch.zeros(1))
        self.wavelet_eps = float(wavelet_eps)

    def gate_value(self) -> Tensor:
        return torch.tanh(self.wavelet_gate)

    def forward(
        self,
        x: Tensor,
        *,
        raw_rgb: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        if domain_strength is not None:
            raise ValueError("WaveletResidualHBP hanya untuk source_only.")
        if labels is not None and labels.ndim != 1:
            raise ValueError("labels harus 1D.")

        features = list(self.encoder(x))

        with torch.no_grad():
            details = luminance_l1_visushrink_details(
                raw_rgb,
                eps=self.wavelet_eps,
            )
        residual = self.wavelet_branch(details)
        if residual.shape[-2:] != features[0].shape[-2:]:
            residual = F.interpolate(
                residual,
                size=features[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        features[0] = features[0] + self.gate_value() * residual

        embedding = self.pool(features)
        classifier_embedding = self.dropout(embedding)
        logits = self.classifier(classifier_embedding)
        return ModelOutput(
            logits=logits,
            embedding=embedding,
        )


def shared_hbp_core_state(model: nn.Module) -> dict[str, Tensor]:
    """Clone the shared RGB-HBP core for exact initialization checks."""

    prefixes = ("encoder.", "pool.", "classifier.")
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def assert_shared_hbp_core_equal(control: nn.Module, candidate: nn.Module) -> None:
    left = shared_hbp_core_state(control)
    right = shared_hbp_core_state(candidate)
    if left.keys() != right.keys():
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise RuntimeError(
            "Shared core keys berbeda. "
            f"control_missing={missing_left}, candidate_missing={missing_right}"
        )
    mismatched = [
        key
        for key in left
        if not torch.equal(left[key], right[key])
    ]
    if mismatched:
        raise RuntimeError(
            "Shared RGB-HBP initialization tidak identik: "
            + ", ".join(mismatched[:10])
        )
