import copy

import torch
from torch import nn

from bilinear_lmmd.engine.at_sbn import (
    AuxiliaryTrainingModel,
    classifier_merge_loss,
    soft_target_cross_entropy,
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


def test_auxiliary_head_creation_preserves_global_rng_stream():
    torch.manual_seed(42)
    before = torch.random.get_rng_state().clone()

    # Establish the state immediately after an ordinary base-model build.
    from bilinear_lmmd.modeling.models import build_model

    _ = build_model(copy.deepcopy(_cfg()))
    expected_after_base = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(before)
    wrapped = AuxiliaryTrainingModel(copy.deepcopy(_cfg()))
    actual_after_wrapper = torch.random.get_rng_state().clone()

    assert wrapped.primary_initial_sha256
    assert torch.equal(actual_after_wrapper, expected_after_base)


def test_auxiliary_classifiers_are_separate_from_primary():
    torch.manual_seed(42)
    model = AuxiliaryTrainingModel(_cfg())
    primary = model.base.classifier
    assert isinstance(primary, nn.Linear)
    for arm, head in model.auxiliary_classifiers.items():
        assert head is not primary
        assert head.weight.data_ptr() != primary.weight.data_ptr()
        assert arm in {"C0", "F0", "W0"}


def test_soft_target_cross_entropy_has_no_teacher_gradient():
    student = torch.randn(4, 17, requires_grad=True)
    teacher = torch.randn(4, 17, requires_grad=True)
    loss = soft_target_cross_entropy(student, teacher, temperature=1.0)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_merge_loss_zero_when_aux_heads_copy_primary():
    torch.manual_seed(42)
    model = AuxiliaryTrainingModel(_cfg())
    primary = model.base.classifier
    with torch.no_grad():
        for head in model.auxiliary_classifiers.values():
            head.weight.copy_(primary.weight)
            head.bias.copy_(primary.bias)
    loss, distances = classifier_merge_loss(model)
    assert torch.allclose(loss, torch.zeros_like(loss))
    assert all(value == 0.0 for value in distances.values())
