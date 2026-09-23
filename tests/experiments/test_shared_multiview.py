import torch
from torch import nn

from bilinear_lmmd.engine.shared_multiview import (
    auxiliary_batchnorm_eval,
    validate_shared_multiview_config,
)


def _minimal_config():
    return {
        "seed": 42,
        "data": {
            "augmentation_mode": "preprocessing_study",
            "object_crop": False,
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "head": "gap",
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
        "multiview": {
            "method": "shared_mvce",
            "views": ["R0", "C0", "F0", "W0"],
            "raw_weight": 0.5,
            "auxiliary_weight": 0.5,
            "raw_bn_updates_only": True,
            "frontends": {
                arm: {"enabled": True, "code": arm, "method": "raw"}
                for arm in ("R0", "C0", "F0", "W0")
            },
        },
    }


def test_shared_multiview_contract_accepts_frozen_v1():
    validate_shared_multiview_config(_minimal_config())


def test_auxiliary_batchnorm_eval_restores_state_and_keeps_affine_gradients():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
    model.train()
    bn = model[0]
    before = bn.running_mean.clone()

    raw = torch.randn(8, 3)
    model(raw).sum().backward()
    assert not torch.equal(before, bn.running_mean)

    model.zero_grad(set_to_none=True)
    running_before_aux = bn.running_mean.clone()
    with auxiliary_batchnorm_eval(model):
        assert bn.training is False
        model(torch.randn(8, 3)).sum().backward()
    assert bn.training is True
    assert torch.equal(running_before_aux, bn.running_mean)
    assert bn.weight.grad is not None
