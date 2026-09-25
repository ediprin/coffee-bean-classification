import copy

import torch

from bilinear_lmmd.engine.mvfd_sbn import (
    confidence_aware_teacher,
    validate_mvfd_sbn_config,
)


def _base_config():
    return {
        "seed": 42,
        "data": {
            "augmentation_mode": "preprocessing_study",
            "object_crop": False,
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "head": "gap",
            "classifier": "linear",
            "out_indices": [4],
        },
        "adaptation": {"method": "source_only"},
        "training": {
            "epochs": 50,
            "classification_loss": "cross_entropy",
            "ema_decay": 0.0,
        },
        "preprocessing": {
            "enabled": True,
            "code": "R0",
            "method": "raw",
        },
        "feature_distillation": {
            "method": "ca_mvfd_sbn",
            "views": ["R0", "C0", "F0", "W0"],
            "primary_view": "R0",
            "teacher_views": ["C0", "F0", "W0"],
            "teacher_aggregation": "confidence_aware_ce",
            "teacher_stop_gradient": True,
            "selective_batch_norm": True,
            "aux_ce_weight_each": 0.05,
            "feature_distill_weight": 0.007,
            "feature_distill_loss": "squared_l2",
            "auxiliary_dropout": False,
            "frontends": {
                arm: {"enabled": True, "code": arm, "method": "raw"}
                for arm in ("R0", "C0", "F0", "W0")
            },
        },
    }


def test_validator_accepts_ca_mvfd_sbn():
    validate_mvfd_sbn_config(_base_config())


def test_validator_rejects_equal_mean_for_ca_mvfd():
    cfg = _base_config()
    cfg["feature_distillation"]["teacher_aggregation"] = "equal_mean"
    try:
        validate_mvfd_sbn_config(cfg)
    except ValueError:
        return
    raise AssertionError("CA-MVFD harus menolak equal_mean.")


def test_confidence_weights_sum_to_one_and_prefer_lower_ce():
    embeddings = {
        "C0": torch.tensor([[1.0, 0.0]]),
        "F0": torch.tensor([[0.0, 1.0]]),
        "W0": torch.tensor([[2.0, 2.0]]),
    }
    logits = {
        "C0": torch.tensor([[6.0, 0.0]]),
        "F0": torch.tensor([[1.0, 0.0]]),
        "W0": torch.tensor([[-2.0, 2.0]]),
    }
    labels = torch.tensor([0])

    teacher, weights = confidence_aware_teacher(embeddings, logits, labels)

    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1.0e-6)
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
    expected = sum(
        weights[:, index : index + 1] * embeddings[arm]
        for index, arm in enumerate(("C0", "F0", "W0"))
    )
    assert torch.allclose(teacher, expected)


def test_equal_confidence_produces_equal_weights():
    embeddings = {
        arm: torch.randn(2, 4) for arm in ("C0", "F0", "W0")
    }
    logits = {
        arm: torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        for arm in ("C0", "F0", "W0")
    }
    labels = torch.tensor([0, 1])

    _, weights = confidence_aware_teacher(embeddings, logits, labels)
    assert torch.allclose(
        weights,
        torch.full_like(weights, 1.0 / 3.0),
        atol=1.0e-6,
    )
