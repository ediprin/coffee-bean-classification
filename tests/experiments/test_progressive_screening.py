from __future__ import annotations

import torch
from torch import nn

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.engine.train_progressive import progressive_objective
from bilinear_lmmd.experiments.run_progressive_multigranularity import (
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_progressive_gate_requires_all_metrics():
    assert screening_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.00, 0.02, 0.01))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, -0.001, 0.01))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.01, -0.011))["decision"] == "FAIL"


def test_dynamic_consistency_balance_matches_official_ratio():
    logits = torch.zeros(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1])
    descriptor = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    total, classification, consistency = progressive_objective(
        logits,
        targets,
        descriptor,
        pair_batch_size=2,
        classification_loss=nn.CrossEntropyLoss(),
        consistency_weight=0.1,
    )
    assert consistency is not None and consistency > 0
    torch.testing.assert_close(total.detach(), classification.detach() * 1.1)
    total.backward()
    assert logits.grad is not None
    assert descriptor.grad is not None


def test_e2_e3_configs_are_controlled_against_efficientnet_baselines():
    gap = load_config("configs/backbones/BE2G_efficientnetv2_gap_source.yaml")
    hbp = load_config("configs/backbones/BE2H_efficientnetv2_hbp_source.yaml")
    e2 = load_config(
        "configs/finegrained/E2_efficientnetv2_progressive_multigranularity.yaml"
    )
    e3 = load_config(
        "configs/finegrained/E3_efficientnetv2_progressive_consistency.yaml"
    )
    for candidate in (e2, e3):
        assert candidate["data"]["image_size"] == gap["data"]["image_size"]
        assert candidate["model"]["backbone"] == gap["model"]["backbone"]
        assert candidate["model"]["out_indices"] == hbp["model"]["out_indices"]
        for key in ("epochs", "lr", "weight_decay", "label_smoothing"):
            assert candidate["training"][key] == gap["training"][key]
    assert e2["model"]["head"] == "progressive_multigranularity"
    assert e3["model"]["head"] == "progressive_multigranularity_consistency"
