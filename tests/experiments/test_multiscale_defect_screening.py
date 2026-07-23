from __future__ import annotations

from bilinear_lmmd.experiments.run_multiscale_defect_screening import (
    MODEL_CONFIGS,
    REQUIRED_COMPARISONS,
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_mde_screening_has_candidate_and_capacity_control() -> None:
    assert tuple(MODEL_CONFIGS) == ("MDE0", "MDE1")
    assert all(path.is_file() for path in MODEL_CONFIGS.values())
    assert REQUIRED_COMPARISONS == ("BE2G_vs_MDE1", "MDE0_vs_MDE1")


def test_mde_gate_requires_macro_hard_and_worst_preservation() -> None:
    assert screening_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.01, -0.001, 0.05))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.02, -0.011))["decision"] == "FAIL"
