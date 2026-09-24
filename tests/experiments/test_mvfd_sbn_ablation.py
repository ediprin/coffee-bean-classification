import copy

from bilinear_lmmd.engine.mvfd_sbn import validate_mvfd_sbn_config


def _config(method: str, weight: float) -> dict:
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
            "method": method,
            "views": ["R0", "C0", "F0", "W0"],
            "primary_view": "R0",
            "teacher_views": ["C0", "F0", "W0"],
            "teacher_aggregation": "equal_mean",
            "teacher_stop_gradient": True,
            "selective_batch_norm": True,
            "aux_ce_weight_each": 0.05,
            "feature_distill_weight": weight,
            "feature_distill_loss": "squared_l2",
            "auxiliary_dropout": False,
            "frontends": {
                arm: {"enabled": True, "code": arm, "method": "raw"}
                for arm in ("R0", "C0", "F0", "W0")
            },
        },
    }


def test_validator_accepts_frozen_mvfd_weight():
    validate_mvfd_sbn_config(_config("mvfd_sbn", 0.007))


def test_validator_accepts_zero_feature_causal_control():
    validate_mvfd_sbn_config(_config("auxce_sbn_control", 0.0))


def test_validator_rejects_nonzero_feature_control():
    cfg = _config("auxce_sbn_control", 0.007)
    try:
        validate_mvfd_sbn_config(cfg)
    except ValueError:
        return
    raise AssertionError("Control dengan lambda_feat != 0 harus ditolak.")
