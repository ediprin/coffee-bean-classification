from bilinear_lmmd.experiments.run_osr_hbp_screening import compare_hbp_to_gap


def _summary(near: dict, medium: dict) -> dict:
    return {
        "splits": {
            "near": {"primary_balanced": {"msp": near}},
            "medium": {"primary_balanced": {"msp": medium}},
        }
    }


def test_hbp_gate_passes_only_tier_meeting_all_frozen_criteria() -> None:
    baseline_metrics = {
        "known_macro_f1": 0.85,
        "oscr": 0.65,
        "auroc": 0.75,
        "fpr95": 0.90,
    }
    passing = {
        "known_macro_f1": 0.84,
        "oscr": 0.66,
        "auroc": 0.77,
        "fpr95": 0.85,
    }
    failing = {
        "known_macro_f1": 0.86,
        "oscr": 0.66,
        "auroc": 0.76,
        "fpr95": 0.85,
    }

    result = compare_hbp_to_gap(
        _summary(baseline_metrics, baseline_metrics),
        _summary(passing, failing),
    )

    assert result["tiers"]["near"]["decision"] == "PASS"
    assert result["tiers"]["medium"]["decision"] == "FAIL"
    assert result["passed_tiers"] == ["near"]
    assert result["decision"] == "PASS"
