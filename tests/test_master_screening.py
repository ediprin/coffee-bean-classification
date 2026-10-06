from __future__ import annotations

import torch

from bilinear_lmmd.experiments.run_master_screening_v1 import (
    MASTER_CANDIDATES,
    build_master_model,
    preflight_master,
)


def _cfg() -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "data": {
            "root": "unused",
            "source": "source",
            "train_split": "train",
            "val_split": "val",
            "image_size": 224,
            "batch_size": 32,
            "workers": 0,
            "rotation_angles": [0, 45, 90, 135, 180, 225, 270],
            "augmentation_mode": "preprocessing_study",
            "object_crop": False,
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": False,
            "head": "hbp",
            "classifier": "linear",
            "num_classes": 17,
            "out_indices": [1, 3, 4],
            "projection_dim": 64,
            "dropout": 0.0,
        },
        "frequency": {
            "attention_reduction": 16,
            "fda_hidden": 4,
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
            "fusion_loss_ce_weight": 0.7,
            "fusion_loss_focal_weight": 0.3,
            "fusion_loss_gamma": 2.0,
        },
        "wavelet_residual": {
            "domain": "luminance",
            "level": "L1",
            "bands": ["LH", "HL", "HH"],
            "threshold": "visushrink_soft",
            "injection": "shallow_residual",
            "hidden_channels": 16,
            "gate": "tanh_zero_init",
            "eps": 1e-8,
        },
        "adaptation": {"method": "source_only"},
        "training": {
            "epochs": 50,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "classification_loss": "cross_entropy",
            "label_smoothing": 0.1,
            "scheduler": "cosine",
            "ema_decay": 0.0,
        },
        "evaluation": {
            "hard_groups": {
                "sour_black": ["Partial Black", "Partial Sour", "Full Sour"],
                "shape_withered": ["Withered", "Immature", "Cut"],
                "insect_damage": ["Slight Insect Damage", "Severe Insect Damage"],
            }
        },
    }


def test_master_candidate_registry_is_complete() -> None:
    assert set(MASTER_CANDIDATES) == {
        "B0", "B1", "B2",
        "R1", "R2", "R3", "R4", "R5",
        "W1", "W2", "W3", "W4", "W5", "W6", "W7",
        "F1", "F2",
    }


def test_master_build_representative_candidates() -> None:
    cfg = _cfg()
    images = torch.randn(1, 3, 224, 224)
    for candidate in ("B0", "B1", "B2", "R1", "R3", "R4", "W3", "W7", "F1", "F2"):
        torch.manual_seed(42)
        model = build_master_model(candidate, cfg)
        if candidate == "B2":
            output = model(images, raw_rgb=images)
        else:
            output = model(images)
        assert output.logits.shape == (1, 17)


def test_master_preflight_for_matched_candidates() -> None:
    cfg = _cfg()
    for candidate in ("B0", "B1", "R1", "R2", "R5", "W1", "W2", "W3", "W4"):
        payload = preflight_master(cfg, candidate)
        assert payload["candidate"] == candidate
