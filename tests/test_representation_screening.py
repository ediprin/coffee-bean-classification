from __future__ import annotations

import torch

from bilinear_lmmd.modeling.frequency_screening import assert_shared_core_equal
from bilinear_lmmd.modeling.representation_screening import (
    LowRankBilinearClassifier,
    build_representation_screening_model,
    lrbp_hinge_loss,
)


def _cfg() -> dict:
    return {
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": False,
            "classifier": "linear",
            "num_classes": 17,
            "projection_dim": 64,
            "dropout": 0.0,
        },
        "representation": {
            "mfr_reduction": 8,
            "mfr_alpha": 0.8,
            "mfr_rho": 0.2,
            "lrbp_reduced_dim": 32,
            "lrbp_rank": 8,
            "lrbp_regularization": 5e-4,
            "multistage_dim": 64,
            "dualdomain_dim": 64,
            "ffc_alpha": 0.5,
        },
    }


def test_mfr_shared_core_matches_gap_and_hbp() -> None:
    cfg = _cfg()
    for candidate, anchor in (("R1", "B0"), ("R2", "B1")):
        torch.manual_seed(42)
        baseline = build_representation_screening_model(anchor, cfg)
        torch.manual_seed(42)
        model = build_representation_screening_model(candidate, cfg)
        assert_shared_core_equal(baseline, model)


def test_representation_candidates_forward() -> None:
    cfg = _cfg()
    images = torch.randn(2, 3, 224, 224)
    for candidate in ("R1", "R2", "R4", "R5", "W7", "F1", "F2"):
        torch.manual_seed(42)
        model = build_representation_screening_model(candidate, cfg)
        output = model(images)
        assert output.logits.shape == (2, 17)
        assert output.embedding.ndim == 2


def test_lrbp_forward_and_hinge_loss() -> None:
    cfg = _cfg()
    torch.manual_seed(42)
    model = build_representation_screening_model("R3", cfg)
    assert isinstance(model, LowRankBilinearClassifier)
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 3], dtype=torch.long)
    output = model(images)
    assert output.logits.shape == (2, 17)
    loss = lrbp_hinge_loss(
        model,
        output.logits,
        labels,
        regularization_weight=5e-4,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
