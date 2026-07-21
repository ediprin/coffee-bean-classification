from __future__ import annotations

from bilinear_lmmd.experiments.run_factorized_bilinear_conv_screening import (
    MODEL_CONFIGS,
    SCREENING_SEEDS,
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_fbconv_screening_protocol_is_locked() -> None:
    assert tuple(MODEL_CONFIGS) == ("FB0", "FB1")
    assert SCREENING_SEEDS == (42,)
    assert all(path.is_file() for path in MODEL_CONFIGS.values())


def test_fbconv_gate_requires_macro_hard_and_worst_preservation() -> None:
    assert screening_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.00, 0.02, 0.03))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, -0.001, 0.03))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.02, -0.011))["decision"] == "FAIL"
