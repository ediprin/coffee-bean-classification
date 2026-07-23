from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import timm

from bilinear_lmmd.modeling.models import ModelOutput


def _to_nchw(feature: Tensor, expected_channels: int) -> Tensor:
    """Normalize timm feature maps without guessing ambiguous layouts."""
    if feature.ndim != 4:
        raise ValueError(
            f"Feature Swin harus 4D, diterima {tuple(feature.shape)}."
        )
    if feature.shape[1] == expected_channels:
        return feature
    if feature.shape[-1] == expected_channels:
        return feature.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        "Channel feature Swin tidak cocok: "
        f"shape={tuple(feature.shape)}, expected={expected_channels}."
    )


class ChannelAttention(nn.Module):
    """Max/average channel screening shown in Fig. 7 of Jiao et al."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("Channel dan reduction HS-FPN harus positif.")
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, feature: Tensor) -> Tensor:
        maximum = self.mlp(F.adaptive_max_pool2d(feature, 1))
        average = self.mlp(F.adaptive_avg_pool2d(feature, 1))
        return feature * torch.sigmoid(maximum + average)


class SelectFeatureFusion(nn.Module):
    """Top-down selective feature fusion (SFF) from the paper diagram."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, high: Tensor, low: Tensor) -> Tensor:
        high = F.interpolate(high, size=low.shape[-2:], mode="nearest")
        return self.refine(high + low)


class HighLevelScreeningFPN(nn.Module):
    """Paper-faithful three-stage HS-FPN for Swin stages S3--S5.

    The authors' released ``EnhancedSwinTransformer`` accidentally replaces
    an intermediate fusion and therefore drops the middle feature map. This
    implementation follows Fig. 7 and Sec. 3.4 instead: every selected stage
    participates in one ordinary top-down path.
    """

    def __init__(
        self,
        input_channels: list[int],
        output_channels: int = 256,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        if len(input_channels) != 3:
            raise ValueError("HS-FPN Jiao membutuhkan tepat tiga stage Swin.")
        if output_channels <= 0:
            raise ValueError("hsfpn_channels harus lebih besar dari nol.")
        self.input_channels = tuple(int(value) for value in input_channels)
        self.output_channels = int(output_channels)
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    ChannelAttention(channels, attention_reduction),
                    nn.Conv2d(
                        channels,
                        output_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                )
                for channels in input_channels
            ]
        )
        self.fusion = nn.ModuleList(
            [
                SelectFeatureFusion(output_channels)
                for _ in range(len(input_channels) - 1)
            ]
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != len(self.input_channels):
            raise ValueError(
                f"HS-FPN menerima 3 feature map, didapat {len(features)}."
            )
        selected = [
            layer(_to_nchw(feature, channels))
            for feature, channels, layer in zip(
                features, self.input_channels, self.lateral
            )
        ]
        result = selected[-1]
        fusion_index = 0
        for low in reversed(selected[:-1]):
            result = self.fusion[fusion_index](result, low)
            fusion_index += 1
        return result


class ControlledDepthwiseSeparableConvolution(nn.Module):
    """CDSC reconstruction from Fig. 9 and the accompanying prose."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            groups=input_channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(input_channels)
        self.pointwise = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, bias=False
        )
        self.pointwise_norm = nn.BatchNorm2d(output_channels)

    def forward(self, feature: Tensor) -> Tensor:
        refined = self.depthwise_norm(F.gelu(self.depthwise(feature)))
        refined = refined + feature
        return self.pointwise_norm(self.pointwise(refined))


class SelectiveAttentionModule(nn.Module):
    """SAM discriminability enhancement reconstructed from Fig. 8.

    The released code returns a vector and omits the element-wise multiplication
    drawn in the paper. Here CDSC features are spatially pooled before the
    three FC layers, expanded back to the input channels, and multiplied with
    the original feature map as specified by the diagram. No sigmoid is added
    because neither the diagram nor released ``exp_channel`` layer uses one.
    """

    def __init__(self, channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if channels <= 0 or hidden_dim <= 0:
            raise ValueError("Channel dan hidden_dim SAM harus positif.")
        self.channels = int(channels)
        self.cdsc = ControlledDepthwiseSeparableConvolution(
            channels, hidden_dim
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.expand_channels = nn.Linear(hidden_dim, channels)

    def forward(self, feature: Tensor) -> Tensor:
        descriptor = F.adaptive_avg_pool2d(self.cdsc(feature), 1).flatten(1)
        descriptor = F.relu(self.fc1(descriptor), inplace=True)
        descriptor = F.relu(self.fc2(descriptor), inplace=True)
        descriptor = F.relu(self.fc3(descriptor), inplace=True)
        weights = self.expand_channels(descriptor).view(
            feature.shape[0], self.channels, 1, 1
        )
        return feature * weights


class SwinHSSAMClassifier(nn.Module):
    """Controlled Swin-T / HS-FPN / SAM classification factorial."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        out_indices: tuple[int, ...] = (1, 2, 3),
        hsfpn_channels: int = 256,
        sam_hidden_dim: int = 128,
        attention_reduction: int = 16,
        dropout: float = 0.2,
        pretrained: bool = True,
        use_hsfpn: bool = True,
        use_sam: bool = True,
    ) -> None:
        super().__init__()
        if len(out_indices) != 3:
            raise ValueError("Swin-HSSAM membutuhkan tepat tiga out_indices.")
        self.backbone_name = backbone
        self.head = "swin_hssam"
        self.use_hsfpn = bool(use_hsfpn)
        self.use_sam = bool(use_sam)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        self.feature_channels = tuple(channels)
        self.hsfpn: HighLevelScreeningFPN | None = None
        if self.use_hsfpn:
            self.hsfpn = HighLevelScreeningFPN(
                channels,
                output_channels=hsfpn_channels,
                attention_reduction=attention_reduction,
            )
            final_channels = hsfpn_channels
        else:
            final_channels = channels[-1]
        self.sam: SelectiveAttentionModule | None = None
        if self.use_sam:
            self.sam = SelectiveAttentionModule(
                final_channels, hidden_dim=sam_hidden_dim
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_channels, num_classes)
        self.embedding_dim = int(final_channels)

    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        del labels
        if domain_strength is not None:
            raise ValueError("Swin-HSSAM reproduction hanya mendukung source_only.")
        features = self.encoder(x)
        if self.hsfpn is not None:
            feature = self.hsfpn(features)
        else:
            feature = _to_nchw(features[-1], self.feature_channels[-1])
        if self.sam is not None:
            feature = self.sam(feature)
        embedding = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        logits = self.classifier(self.dropout(embedding))
        return ModelOutput(logits=logits, embedding=embedding)
