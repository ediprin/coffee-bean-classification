from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bilinear_lmmd.modeling.models import ModelOutput


@dataclass(frozen=True)
class DCLTrainingOutput:
    """Outputs used only by destruction-construction training."""

    classification: ModelOutput
    swap_logits: Tensor
    layout: Tensor


class DCLFineGrainedModel(nn.Module):
    """EfficientNet classifier with training-only DCL auxiliary heads.

    The design follows Chen et al. (CVPR 2019): the ordinary classification
    path uses final-stage GAP, while a binary swap classifier and a 1x1 region
    alignment head supervise region-confused images during training. The two
    auxiliary heads are not executed by :meth:`forward`, so deployment remains
    the ordinary backbone-GAP-linear path.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        out_indices: tuple[int, ...],
        grid_size: int,
        dropout: float,
        pretrained: bool,
    ):
        super().__init__()
        if not out_indices:
            raise ValueError("DCL membutuhkan minimal satu output feature stage.")
        if grid_size <= 1:
            raise ValueError("dcl_grid_size harus lebih besar dari satu.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout harus berada pada [0, 1).")

        self.backbone_name = backbone
        self.head = "dcl_gap"
        self.grid_size = int(grid_size)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = int(self.encoder.feature_info.channels()[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = channels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)

        # Auxiliary initialization must not perturb the same-seed backbone and
        # classifier initialization relative to the ordinary GAP control.
        rng_state = torch.random.get_rng_state()
        self.swap_classifier = nn.Linear(channels, 2)
        self.layout_head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        torch.random.set_rng_state(rng_state)

    def _features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.encoder(images)[-1]
        embedding = self.pool(feature).flatten(1)
        return feature, embedding

    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        del labels
        if domain_strength is not None:
            raise ValueError("DCL Coffee17 hanya mendukung source_only.")
        _, embedding = self._features(images)
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)

    def forward_dcl(self, images: Tensor) -> DCLTrainingOutput:
        """Run the shared encoder plus both training-only DCL heads."""

        feature, embedding = self._features(images)
        logits = self.classifier(self.dropout(embedding))
        swap_logits = self.swap_classifier(self.dropout(embedding))
        layout_map = torch.tanh(self.layout_head(feature))
        if layout_map.shape[-2:] != (self.grid_size, self.grid_size):
            layout_map = F.interpolate(
                layout_map,
                size=(self.grid_size, self.grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return DCLTrainingOutput(
            classification=ModelOutput(logits=logits, embedding=embedding),
            swap_logits=swap_logits,
            layout=layout_map.flatten(1),
        )

    def inference_parameter_count(self) -> int:
        """Parameters executed by the deployment path."""

        modules = (self.encoder, self.classifier)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    def auxiliary_parameter_count(self) -> int:
        modules = (self.swap_classifier, self.layout_head)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())
