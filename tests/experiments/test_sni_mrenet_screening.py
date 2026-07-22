from __future__ import annotations

from bilinear_lmmd.experiments.run_sni_mrenet_screening import (
    COMPARISONS,
    MODEL_CONFIGS,
    _audit_models,
)


def test_sni_runner_has_only_the_frozen_b0_b3_sequence():
    assert tuple(MODEL_CONFIGS) == ("SNIB0", "SNIB1", "SNIB2", "SNIB3")
    assert COMPARISONS == (
        ("SNIB0", "SNIB1"),
        ("SNIB1", "SNIB2"),
        ("SNIB2", "SNIB3"),
        ("SNIB0", "SNIB3"),
    )


def test_sni_gap_and_hbp_ontology_models_are_capacity_matched():
    audits = _audit_models()
    assert audits["SNIB2"]["parameters"] == audits["SNIB3"]["parameters"]
