from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bilinear_lmmd.modeling.models import ModelOutput
from bilinear_lmmd.modeling.wavelet_residual_hbp import (
    WaveletResidualHBPModel,
    luminance_l1_visushrink_details,
)


class WaveletCLAHEContrastResidualHBPModel(WaveletResidualHBPModel):
    """WR-HBP plus a zero-gated CLAHE-derived luminance-contrast residual.

    Shared WR-HBP construction is preserved exactly by calling the parent
    constructor first.  The candidate-only contrast branch is initialized
    afterwards and the CPU RNG state is restored, so same-seed WR-HBP and
    WRC-HBP share identical RGB encoder, HBP, classifier, wavelet branch, and
    wavelet gate initialization.

    Candidate representation:
        F_s' = F_s
             + tanh(alpha_w) * P_w(D_L1)
             + tanh(alpha_c) * P_c(Delta_L)

    Delta_L is supplied by the runtime as the signed Rec.709 luminance
    difference between the frozen C0 CLAHE image and the raw RGB image.
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
        wavelet_hidden_channels: int = 16,
        contrast_hidden_channels: int = 8,
        wavelet_eps: float = 1.0e-8,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            out_indices=out_indices,
            projection_dim=projection_dim,
            dropout=dropout,
            pretrained=pretrained,
            branch_hidden_channels=wavelet_hidden_channels,
            wavelet_eps=wavelet_eps,
        )
        if contrast_hidden_channels <= 0:
            raise ValueError("contrast_hidden_channels harus positif.")

        shallow_channels = int(self.encoder.feature_info.channels()[0])

        rng_state = torch.random.get_rng_state()
        self.contrast_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(
                1,
                contrast_hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(contrast_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                contrast_hidden_channels,
                shallow_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(shallow_channels),
        )
        self.contrast_gate = nn.Parameter(torch.zeros(1))
        torch.random.set_rng_state(rng_state)
        self.head = "wavelet_clahe_contrast_residual_hbp"

    def contrast_gate_value(self) -> Tensor:
        return torch.tanh(self.contrast_gate)

    def forward(
        self,
        x: Tensor,
        *,
        raw_rgb: Tensor,
        contrast_residual: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        if domain_strength is not None:
            raise ValueError("WRC-HBP hanya untuk source_only.")
        if labels is not None and labels.ndim != 1:
            raise ValueError("labels harus 1D.")
        if contrast_residual.ndim != 4 or contrast_residual.shape[1] != 1:
            raise ValueError("contrast_residual harus Bx1xHxW.")
        if contrast_residual.shape[0] != x.shape[0]:
            raise ValueError("Batch contrast_residual berbeda dari input.")

        features = list(self.encoder(x))

        with torch.no_grad():
            details = luminance_l1_visushrink_details(
                raw_rgb,
                eps=self.wavelet_eps,
            )
        wavelet_residual = self.wavelet_branch(details)
        if wavelet_residual.shape[-2:] != features[0].shape[-2:]:
            wavelet_residual = F.interpolate(
                wavelet_residual,
                size=features[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        contrast_feature = self.contrast_branch(contrast_residual)
        if contrast_feature.shape[-2:] != features[0].shape[-2:]:
            contrast_feature = F.interpolate(
                contrast_feature,
                size=features[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        features[0] = (
            features[0]
            + self.gate_value() * wavelet_residual
            + self.contrast_gate_value() * contrast_feature
        )

        embedding = self.pool(features)
        classifier_embedding = self.dropout(embedding)
        logits = self.classifier(classifier_embedding)
        return ModelOutput(
            logits=logits,
            embedding=embedding,
        )


def shared_wr_state(model: nn.Module) -> dict[str, Tensor]:
    """State shared by WR-HBP and WRC-HBP, excluding contrast-only modules."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("contrast_branch.")
        and key != "contrast_gate"
    }


def assert_shared_wr_equal(control: nn.Module, candidate: nn.Module) -> None:
    left = shared_wr_state(control)
    right = shared_wr_state(candidate)
    if left.keys() != right.keys():
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise RuntimeError(
            "Shared WR-HBP keys berbeda. "
            f"control_missing={missing_left}, candidate_missing={missing_right}"
        )

    mismatched = [
        key
        for key in left
        if not torch.equal(left[key], right[key])
    ]
    if mismatched:
        raise RuntimeError(
            "Shared WR-HBP initialization tidak identik: "
            + ", ".join(mismatched[:10])
        )
