from __future__ import annotations

import pytest

from bilinear_lmmd.experiments.run_multistage_recalibration_screening import (
    MODEL_CONFIGS,
    REQUIRED_COMPARISONS,
    SCREENING_SEEDS,
    _bottom_three_f1,
    screening_decision,
)


def _summary(macro: float, hard: float, bottom3: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "bottom3_class_f1": {"delta_mean": bottom3},
    }


def test_multistage_screening_protocol_is_locked() -> None:
    assert tuple(MODEL_CONFIGS) == ("MSF0", "MSF1")
    assert SCREENING_SEEDS == (42,)
    assert REQUIRED_COMPARISONS == ("MSF0_vs_MSF1", "BE2G_vs_MSF1")
    assert all(path.is_file() for path in MODEL_CONFIGS.values())


def test_multistage_gate_requires_macro_hard_and_bottom3() -> None:
    assert screening_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.00, 0.02, 0.03))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, -0.001, 0.03))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.02, -0.011))["decision"] == "FAIL"


def test_bottom_three_f1_is_mean_of_three_lowest_classes() -> None:
    report = {
        "per_class": {
            "a": {"f1": 0.8},
            "b": {"f1": 0.2},
            "c": {"f1": 0.4},
            "d": {"f1": 0.6},
        }
    }
    assert _bottom_three_f1(report) == pytest.approx(0.4)
