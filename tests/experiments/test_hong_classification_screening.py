from __future__ import annotations

from bilinear_lmmd.experiments.run_hong_classification_screening import (
    MODEL_CONFIGS,
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_hong_factorial_has_three_candidates() -> None:
    assert tuple(MODEL_CONFIGS) == ("HCD1", "HCS1", "HCDS1")
    assert all(path.is_file() for path in MODEL_CONFIGS.values())


def test_hong_gate_requires_macro_hard_and_worst_preservation() -> None:
    assert screening_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.01, -0.001, 0.05))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.02, -0.011))["decision"] == "FAIL"
