import copy

import torch

from bilinear_lmmd.engine.at_sbn import AuxiliaryTrainingModel
from bilinear_lmmd.engine.ml_mvfd_sbn import (
    spatial_squared_l2_feature_distillation,
    validate_ml_mvfd_sbn_config,
)


def _model_cfg():
    return {
        "backbone": "mobilenetv3_large_100",
        "pretrained": False,
        "head": "gap",
        "classifier": "linear",
        "num_classes": 17,
        "out_indices": [3, 4],
        "dropout": 0.2,
    }


def _full_cfg():
    return {
        "seed": 42,
        "data": {
            "augmentation_mode": "preprocessing_study",
            "object_crop": False,
        },
        "model": _model_cfg(),
        "adaptation": {"method": "source_only"},
        "training": {
            "epochs": 50,
            "classification_loss": "cross_entropy",
            "ema_decay": 0.0,
        },
        "preprocessing": {"code": "R0"},
        "feature_distillation": {
            "method": "ml_mvfd_sbn",
            "views": ["R0", "C0", "F0", "W0"],
            "primary_view": "R0",
            "teacher_views": ["C0", "F0", "W0"],
            "teacher_aggregation": "equal_mean",
            "teacher_stop_gradient": True,
            "selective_batch_norm": True,
            "aux_ce_weight_each": 0.05,
            "feature_distill_weight": 0.007,
            "feature_distill_loss": "squared_l2",
            "mid_feature_distill_weight": 0.007,
            "mid_feature_distill_loss": "spatial_squared_l2",
            "mid_feature_position": 0,
            "mid_backbone_out_index": 3,
            "auxiliary_dropout": False,
            "frontends": {
                "R0": {"code": "R0"},
                "C0": {"code": "C0"},
                "F0": {"code": "F0"},
                "W0": {"code": "W0"},
            },
        },
    }


def test_spatial_feature_distillation_detaches_teacher():
    student = torch.randn(2, 8, 7, 7, requires_grad=True)
    teacher = torch.randn(2, 8, 7, 7, requires_grad=True)
    loss = spatial_squared_l2_feature_distillation(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_spatial_feature_distillation_zero_for_identical_map():
    feature = torch.randn(2, 8, 7, 7)
    loss = spatial_squared_l2_feature_distillation(feature, feature)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_multilevel_config_is_frozen():
    cfg = _full_cfg()
    validate_ml_mvfd_sbn_config(cfg)

    broken = copy.deepcopy(cfg)
    broken["model"]["out_indices"] = [4]
    try:
        validate_ml_mvfd_sbn_config(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("out_indices [4] harus ditolak untuk ML-MVFD-SBN.")


def test_auxiliary_wrapper_exposes_same_final_embedding_shape():
    torch.manual_seed(42)
    model = AuxiliaryTrainingModel(_model_cfg())
    x = torch.randn(2, 3, 224, 224)
    features = model.base.encoder(x)
    assert len(features) == 2
    embedding = model.base.pool(features)
    assert embedding.shape[0] == 2
    assert embedding.shape[1] == model.base.classifier.in_features
