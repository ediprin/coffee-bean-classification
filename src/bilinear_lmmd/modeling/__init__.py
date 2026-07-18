"""Neural-network architectures, objectives, and label structures."""

from bilinear_lmmd.modeling.losses import (
    LMMDLoss,
    MMDLoss,
    OntologyMarginalLoss,
    TaxonomyCompatibleContrastiveLoss,
)
from bilinear_lmmd.modeling.ontology import OntologySpec, load_ontology
from bilinear_lmmd.modeling.models import AdaptationModel, build_model

__all__ = [
    "AdaptationModel",
    "LMMDLoss",
    "MMDLoss",
    "OntologyMarginalLoss",
    "OntologySpec",
    "TaxonomyCompatibleContrastiveLoss",
    "build_model",
    "load_ontology",
]
