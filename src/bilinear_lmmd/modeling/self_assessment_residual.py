from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bilinear_lmmd.modeling.models import ModelOutput
from bilinear_lmmd.modeling.wavelet_residual_hbp import (
    WaveletResidualHBPModel,
    luminance_l1_visushrink_details,
)


class TopKSelfAssessmentResidual(nn.Module):
    """Frozen WR-HBP plus a localized top-k reassessment residual.

    The WR-HBP base is frozen. The reassessment head receives the existing
    mid-level spatial feature map and the base model's current top-k classes.
    It can only add residual logits to those top-k classes; every other logit
    is preserved exactly.

    The final residual layer is zero-initialized, so the candidate starts
    functionally identical to the frozen WR-HBP checkpoint.
    """

    def __init__(
        self,
        base: WaveletResidualHBPModel,
        *,
        num_classes: int = 17,
        top_k: int = 5,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        feature_index: int = 1,
        attention_scale: str = "sqrt_dim_cosine",
    ) -> None:
        super().__init__()
        if not 1 <= top_k < num_classes:
            raise ValueError("top_k harus berada di 1..num_classes-1.")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim/hidden_dim harus positif.")
        if feature_index not in (0, 1, 2):
            raise ValueError("feature_index harus 0, 1, atau 2.")

        self.base = base
        self.num_classes = int(num_classes)
        self.top_k = int(top_k)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_index = int(feature_index)
        if attention_scale != "sqrt_dim_cosine":
            raise ValueError(
                "attention_scale harus 'sqrt_dim_cosine' pada SAR V2."
            )
        self.attention_scale = attention_scale
        self.attention_logit_scale = math.sqrt(float(self.embedding_dim))

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

        channels = list(self.base.encoder.feature_info.channels())
        spatial_channels = int(channels[self.feature_index])

        self.spatial_projection = nn.Conv2d(
            spatial_channels,
            self.embedding_dim,
            kernel_size=1,
            bias=False,
        )
        self.spatial_norm = nn.LayerNorm(self.embedding_dim)
        self.class_embedding = nn.Embedding(
            self.num_classes,
            self.embedding_dim,
        )
        self.joint = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + 1, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

        # Exact baseline at initialization. Only this layer learns on the first
        # optimizer step; upstream reassessment parameters receive gradients
        # after the residual path departs from zero.
        nn.init.zeros_(self.joint[-1].weight)
        nn.init.zeros_(self.joint[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen base must keep BN/dropout in evaluation mode.
        self.base.eval()
        return self

    @torch.no_grad()
    def _frozen_base_features(
        self,
        normalized: Tensor,
        raw_rgb: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        features = list(self.base.encoder(normalized))

        details = luminance_l1_visushrink_details(
            raw_rgb,
            eps=self.base.wavelet_eps,
        )
        residual = self.base.wavelet_branch(details)
        if residual.shape[-2:] != features[0].shape[-2:]:
            residual = F.interpolate(
                residual,
                size=features[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        features[0] = features[0] + self.base.gate_value() * residual

        embedding = self.base.pool(features)
        base_logits = self.base.classifier(self.base.dropout(embedding))
        return features, base_logits

    def forward(
        self,
        normalized: Tensor,
        *,
        raw_rgb: Tensor,
    ) -> ModelOutput:
        features, base_logits = self._frozen_base_features(normalized, raw_rgb)

        top_values, top_indices = torch.topk(
            base_logits.detach(),
            k=self.top_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        top_probabilities = torch.softmax(base_logits.detach(), dim=1).gather(
            1, top_indices
        )

        spatial = self.spatial_projection(features[self.feature_index].detach())
        b, d, h, w = spatial.shape
        spatial = spatial.flatten(2).transpose(1, 2)
        spatial = self.spatial_norm(spatial)
        spatial = F.normalize(spatial, dim=-1)

        class_vectors = self.class_embedding(top_indices)
        class_vectors = F.normalize(class_vectors, dim=-1)

        attention_logits = torch.einsum(
            "bld,bkd->bkl",
            spatial,
            class_vectors,
        ) * self.attention_logit_scale
        attention = torch.softmax(attention_logits, dim=-1)
        attended = torch.einsum(
            "bkl,bld->bkd",
            attention,
            spatial,
        )

        joint = torch.cat(
            (
                attended,
                class_vectors,
                top_probabilities.unsqueeze(-1),
            ),
            dim=-1,
        )
        top_residual = self.joint(joint).squeeze(-1)

        full_residual = torch.zeros_like(base_logits)
        full_residual.scatter_(1, top_indices, top_residual)
        logits = base_logits + full_residual

        return ModelOutput(
            logits=logits,
            embedding=attended.flatten(1),
            expert_logits={
                "base": base_logits,
                "residual": full_residual,
            },
            gate_weights=attention,
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.") and parameter.requires_grad
        )
