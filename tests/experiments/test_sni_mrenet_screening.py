from __future__ import annotations

from bilinear_lmmd.experiments.run_sni_mrenet_screening import (
    COMPARISONS,
    MODEL_CONFIGS,
    STAGE_COMPARISONS,
    STAGE_MODELS,
    _audit_models,
    screening_decision,
)


def test_sni_runner_has_only_the_frozen_b0_b3_sequence():
    assert tuple(MODEL_CONFIGS) == ("SNIB0", "SNIB1", "SNIB2", "SNIB3")
    assert COMPARISONS == (
        ("SNIB0", "SNIB1"),
        ("SNIB1", "SNIB2"),
        ("SNIB2", "SNIB3"),
        ("SNIB0", "SNIB3"),
    )
    assert STAGE_MODELS["backbone"] == ("SNIB0", "SNIB1")
    assert STAGE_MODELS["ontology"] == ("SNIB2",)
    assert STAGE_MODELS["bilinear"] == ("SNIB3",)
    assert STAGE_COMPARISONS["bilinear"] == (
        ("SNIB2", "SNIB3"),
        ("SNIB0", "SNIB3"),
    )


def test_sni_gap_and_hbp_ontology_models_are_capacity_matched():
    audits = _audit_models()
    assert audits["SNIB2"]["parameters"] == audits["SNIB3"]["parameters"]


def test_sni_stage_requires_macro_hard_and_worst_preservation():
    passing = {
        "macro_f1": {"delta_mean": 0.01},
        "hard_class_f1": {"delta_mean": 0.02},
        "worst_class_f1": {"delta_mean": -0.01},
    }
    assert screening_decision(passing)["decision"] == "PASS"
    passing["hard_class_f1"]["delta_mean"] = 0.0
    assert screening_decision(passing)["decision"] == "FAIL"
