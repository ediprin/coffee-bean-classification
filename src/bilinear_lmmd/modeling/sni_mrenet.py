from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import timm

from bilinear_lmmd.data.sni_ontology import SNI_CLASSES, SNI_GROUP_SIZES
from bilinear_lmmd.modeling.models import (
    HierarchicalBilinearPooling,
    ModelOutput,
    ProjectedHierarchicalGAP,
)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class MultiResolutionFusion(nn.Module):
    """Lightweight top-down fusion over four backbone resolutions."""

    def __init__(self, channels: list[int], feature_dim: int) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("SNI multi-resolution fusion membutuhkan 4 feature map.")
        if feature_dim <= 0:
            raise ValueError("sni_feature_dim harus lebih besar dari nol.")
        self.lateral = nn.ModuleList(
            [
                ConvNormAct(channel, feature_dim, kernel_size=1)
                for channel in channels
            ]
        )
        self.refine = nn.ModuleList(
            [
                ConvNormAct(
                    feature_dim,
                    feature_dim,
                    kernel_size=3,
                    groups=feature_dim,
                )
                for _ in channels
            ]
        )
        self.output_dim = int(feature_dim)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != 4:
            raise ValueError(
                f"SNI fusion menerima 4 feature map, didapat {len(features)}."
            )
        lateral = [layer(feature) for layer, feature in zip(self.lateral, features)]
        fused: list[Tensor | None] = [None] * len(lateral)
        top_down: Tensor | None = None
        for index in reversed(range(len(lateral))):
            current = lateral[index]
            if top_down is not None:
                top_down = F.interpolate(
                    top_down,
                    size=current.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                current = current + top_down
            current = self.refine[index](current)
            fused[index] = current
            top_down = current
        return [feature for feature in fused if feature is not None]


class SNIMultiResolutionExpertModel(nn.Module):
    """SNI-MRENet and its controlled first-order ablations.

    ``flat`` uses the fused multi-resolution GAP embedding with a single
    21-class classifier. ``ontology_gap`` and ``ontology_hbp`` factor the
    normalized leaf probability into a four-way router and group-conditional
    classifiers. Their bean interaction modules are parameter- and
    dimension-matched; only first-order versus multiplicative pooling differs.
    """

    MODES = {
        "flat",
        "flat_residual_gap",
        "flat_residual_hbp",
        "ontology_gap",
        "ontology_hbp",
    }

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        out_indices: tuple[int, ...],
        feature_dim: int,
        projection_dim: int,
        dropout: float,
        pretrained: bool,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Mode SNI-MRE harus salah satu dari {sorted(self.MODES)}.")
        if num_classes != len(SNI_CLASSES):
            raise ValueError(
                f"SNI-MRENet dikunci ke {len(SNI_CLASSES)} kelas, didapat {num_classes}."
            )
        if len(out_indices) != 4:
            raise ValueError("SNI-MRENet membutuhkan tepat 4 out_indices.")
        if projection_dim <= 0:
            raise ValueError("projection_dim harus lebih besar dari nol.")

        self.backbone_name = backbone
        self.mode = mode
        self.head = {
            "flat": "sni_multiresolution_flat",
            "flat_residual_gap": "sni_flat_residual_gap",
            "flat_residual_hbp": "sni_flat_residual_hbp",
            "ontology_gap": "sni_mre_ontology_gap",
            "ontology_hbp": "sni_mrenet",
        }[mode]
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self.encoder.feature_info.channels())
        self.fusion = MultiResolutionFusion(channels, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.global_dim = feature_dim * 4

        self.flat_classifier: nn.Linear | None = None
        self.bean_residual_classifier: nn.Linear | None = None
        self.residual_dropout: nn.Dropout | None = None
        self.router: nn.Linear | None = None
        self.bean_pool: nn.Module | None = None
        self.expert_classifiers = nn.ModuleList()

        if mode.startswith("flat"):
            self.flat_classifier = nn.Linear(self.global_dim, num_classes)
            self.embedding_dim = self.global_dim
            if mode != "flat":
                pool_type = (
                    ProjectedHierarchicalGAP
                    if mode == "flat_residual_gap"
                    else HierarchicalBilinearPooling
                )
                self.bean_pool = pool_type([feature_dim] * 3, projection_dim)
                self.bean_residual_classifier = nn.Linear(
                    self.bean_pool.output_dim,
                    SNI_GROUP_SIZES[0],
                )
                # A warm-started residual model must initially reproduce SNIB1
                # exactly. Learning, rather than random initialization, decides
                # whether the selective correction should depart from zero.
                nn.init.zeros_(self.bean_residual_classifier.weight)
                nn.init.zeros_(self.bean_residual_classifier.bias)
                self.residual_dropout = nn.Dropout(dropout)
                self.embedding_dim += self.bean_pool.output_dim
        else:
            self.router = nn.Linear(self.global_dim, len(SNI_GROUP_SIZES))
            pool_type = (
                ProjectedHierarchicalGAP
                if mode == "ontology_gap"
                else HierarchicalBilinearPooling
            )
            self.bean_pool = pool_type([feature_dim] * 3, projection_dim)
            bean_dim = self.global_dim + self.bean_pool.output_dim
            self.expert_classifiers = nn.ModuleList(
                [
                    nn.Linear(bean_dim, SNI_GROUP_SIZES[0]),
                    *[
                        nn.Linear(self.global_dim, group_size)
                        for group_size in SNI_GROUP_SIZES[1:]
                    ],
                ]
            )
            self.embedding_dim = bean_dim

    @staticmethod
    def _global_embedding(features: list[Tensor]) -> Tensor:
        return torch.cat(
            [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in features],
            dim=1,
        )

    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
        domain_strength: float | None = None,
    ) -> ModelOutput:
        del labels
        if domain_strength is not None:
            raise ValueError("SNI-MRENet v1 hanya mendukung source_only.")
        features = self.fusion(self.encoder(x))
        global_embedding = self._global_embedding(features)

        if self.mode.startswith("flat"):
            if self.flat_classifier is None:
                raise RuntimeError("Flat classifier belum diinisialisasi.")
            logits = self.flat_classifier(self.dropout(global_embedding))
            embedding = global_embedding
            if self.mode != "flat":
                if (
                    self.bean_pool is None
                    or self.bean_residual_classifier is None
                    or self.residual_dropout is None
                ):
                    raise RuntimeError("Selective residual head belum diinisialisasi.")
                interaction = self.bean_pool(features[:3])
                bean_residual = self.bean_residual_classifier(
                    self.residual_dropout(interaction)
                )
                # SNI_CLASSES is contiguous: the first 12 labels are physical
                # bean conditions. Residual evidence must not alter material or
                # foreign-object logits (indices 12..20).
                logits = logits + F.pad(
                    bean_residual,
                    (0, len(SNI_CLASSES) - SNI_GROUP_SIZES[0]),
                )
                embedding = torch.cat((global_embedding, interaction), dim=1)
            return ModelOutput(logits=logits, embedding=embedding)

        if self.router is None or self.bean_pool is None:
            raise RuntimeError("Ontology router belum diinisialisasi.")
        interaction = self.bean_pool(features[:3])
        embedding = torch.cat((global_embedding, interaction), dim=1)
        router_log_probability = F.log_softmax(
            self.router(self.dropout(global_embedding)), dim=1
        )
        local_embeddings = [embedding, global_embedding, global_embedding, global_embedding]
        joint_log_probability = []
        for group_index, (classifier, local_embedding) in enumerate(
            zip(self.expert_classifiers, local_embeddings)
        ):
            conditional = F.log_softmax(
                classifier(self.dropout(local_embedding)), dim=1
            )
            joint_log_probability.append(
                router_log_probability[:, group_index : group_index + 1]
                + conditional
            )
        logits = torch.cat(joint_log_probability, dim=1)
        return ModelOutput(logits=logits, embedding=embedding)
