from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from bilinear_lmmd.run_cbd_kd_confirmation import aggregate_kd_results
from bilinear_lmmd.train_cbd_distillation import FrozenStackingTeacher


class LookupModel(nn.Module):
    def __init__(self, logits: np.ndarray):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, indices):
        return SimpleNamespace(logits=self.values[indices.long().flatten()])


def test_frozen_stacking_teacher_matches_sklearn_decision_function():
    gap_logits = np.asarray(
        [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 3.0]] * 6
    )
    hbp_logits = np.asarray(
        [[2.0, 0.0, 1.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]] * 6
    )
    gap_log_prob = torch.log_softmax(torch.tensor(gap_logits), 1).numpy()
    hbp_log_prob = torch.log_softmax(torch.tensor(hbp_logits), 1).numpy()
    features = np.concatenate((gap_log_prob, hbp_log_prob), axis=1)
    labels = np.arange(18) % 3
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    ).fit(features, labels)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    teacher = FrozenStackingTeacher(
        LookupModel(gap_logits),
        LookupModel(hbp_logits),
        scaler.mean_,
        scaler.scale_,
        classifier.coef_,
        classifier.intercept_,
    )
    indices = torch.arange(18).view(-1, 1)
    actual = teacher(indices).detach().numpy()
    expected = pipeline.decision_function(features)
    assert actual == pytest.approx(expected, abs=1e-5)
    assert not teacher.training
    assert not teacher.gap_model.training


def _result_row(values: dict[str, float]) -> dict:
    return {
        name: {
            "accuracy": value,
            "macro_f1": value,
            "defect_f1": value,
            "worst_f1": value,
            "worst_class": "Black",
        }
        for name, value in values.items()
    }


def test_kd_aggregate_requires_fusion_specific_gain():
    row = _result_row(
        {
            "GAP_RAW": 0.90,
            "GAP_KD_CONTROL": 0.905,
            "STACKING_KD": 0.91,
            "STACKING_TEACHER": 0.92,
        }
    )
    aggregate = aggregate_kd_results({42: row, 123: row, 2026: row})
    assert aggregate["decision"]["status"] == "PASS"
    assert aggregate["teacher_gain_preserved"] == pytest.approx(0.5)


def test_one_seed_kd_is_only_screening():
    row = _result_row(
        {
            "GAP_RAW": 0.90,
            "GAP_KD_CONTROL": 0.905,
            "STACKING_KD": 0.91,
            "STACKING_TEACHER": 0.92,
        }
    )
    assert aggregate_kd_results({42: row})["decision"]["status"] == "SCREEN_ONLY"
