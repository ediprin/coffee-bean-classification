from __future__ import annotations

import inspect

from bilinear_lmmd.experiments.run_jiao_swin_hssam_screening import (
    FACTORIAL_COMPARISONS,
    MODEL_CONFIGS,
    STAGE_MODELS,
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_jiao_runner_is_fail_fast_before_factorial():
    assert STAGE_MODELS["screen"] == ("SJ0", "SJFULL")
    assert set(STAGE_MODELS["ablation"]) == set(MODEL_CONFIGS) - {
        "SJ0",
        "SJFULL",
    }
    assert len(MODEL_CONFIGS) == 8
    assert len(FACTORIAL_COMPARISONS) == 7


def test_jiao_decision_requires_macro_hard_and_worst_preservation():
    assert screening_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.01, -0.001, 0.1))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.02, -0.011))["decision"] == "FAIL"


def test_jiao_runner_defaults_to_checkpoint_every_epoch():
    parameters = inspect.signature(
        __import__(
            "bilinear_lmmd.experiments.run_jiao_swin_hssam_screening",
            fromlist=["run_jiao_swin_hssam_screening"],
        ).run_jiao_swin_hssam_screening
    ).parameters

    assert parameters["hf_namespace"].default == "sni-jiao-hssam-v1"
    assert parameters["hf_sync_every"].default == 1
