from bilinear_lmmd.experiments.run_covariance_pooling_screening import (
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_covariance_gate_requires_all_frozen_criteria() -> None:
    passed = screening_decision(_summary(0.01, 0.02, -0.01))
    assert passed["decision"] == "PASS"
    assert all(passed["criteria"].values())

    assert screening_decision(_summary(0.00, 0.02, 0.03))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.00, 0.03))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.02, -0.01001))["decision"] == "FAIL"
