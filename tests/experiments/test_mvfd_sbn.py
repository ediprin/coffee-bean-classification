import copy

import torch

from bilinear_lmmd.engine.at_sbn import AuxiliaryTrainingModel
from bilinear_lmmd.engine.mvfd_sbn import (
    equal_mean_teacher,
    squared_l2_feature_distillation,
)


def _cfg():
    return {
        "backbone": "mobilenetv3_large_100",
        "pretrained": False,
        "head": "gap",
        "classifier": "linear",
        "num_classes": 17,
        "out_indices": [4],
        "dropout": 0.2,
    }


def test_equal_mean_teacher_is_exact_arithmetic_mean():
    embeddings = {
        "C0": torch.tensor([[1.0, 3.0]]),
        "F0": torch.tensor([[3.0, 5.0]]),
        "W0": torch.tensor([[5.0, 7.0]]),
    }
    teacher = equal_mean_teacher(embeddings)
    expected = torch.tensor([[3.0, 5.0]])
    assert torch.allclose(teacher, expected)


def test_feature_distillation_only_backpropagates_to_student():
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8, requires_grad=True)
    loss = squared_l2_feature_distillation(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_zero_feature_distillation_for_identical_features():
    feature = torch.randn(5, 11)
    loss = squared_l2_feature_distillation(feature, feature)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_wrapper_keeps_primary_initialization_stable():
    torch.manual_seed(42)
    first = AuxiliaryTrainingModel(copy.deepcopy(_cfg()))
    first_sha = first.primary_initial_sha256

    torch.manual_seed(42)
    second = AuxiliaryTrainingModel(copy.deepcopy(_cfg()))
    assert second.primary_initial_sha256 == first_sha
