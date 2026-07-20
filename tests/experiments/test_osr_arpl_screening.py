from bilinear_lmmd.experiments.run_osr_arpl_screening import compare_arpl_to_gap


def _summary(near: dict, medium: dict, score: str) -> dict:
    return {
        "splits": {
            "near": {"primary_balanced": {score: near}},
            "medium": {"primary_balanced": {score: medium}},
        }
    }


def test_arpl_gate_compares_mls_candidate_to_frozen_msp_baseline() -> None:
    baseline = {
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

    result = compare_arpl_to_gap(
        _summary(baseline, baseline, "msp"),
        _summary(passing, failing, "mls"),
    )

    assert result["candidate_score"] == "maximum ARPL logit (MLS)"
    assert result["tiers"]["near"]["decision"] == "PASS"
    assert result["tiers"]["medium"]["decision"] == "FAIL"
    assert result["decision"] == "PASS"
