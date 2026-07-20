from bilinear_lmmd.experiments.run_osr_vim_screening import (
    aggregate_vim_comparison,
)


def _metrics(auroc: float, oscr: float) -> dict:
    return {
        "known_macro_f1": 0.85,
        "auroc": auroc,
        "oscr": oscr,
        "fpr95": 0.90,
    }


def _row(seed: int, near_gain: float, medium_gain: float) -> dict:
    splits = {}
    for tier, gain in (
        ("near", near_gain),
        ("medium", medium_gain),
        ("far", 0.0),
    ):
        splits[tier] = {
            "msp": _metrics(0.70, 0.60),
            "vim": _metrics(0.70 + gain, 0.60 + gain / 2),
        }
    return {"seed": seed, "splits": splits}


def test_vim_gate_uses_mean_gain_and_seed_consistency() -> None:
    result = aggregate_vim_comparison(
        [
            _row(42, 0.03, 0.01),
            _row(123, 0.04, -0.01),
            _row(2026, -0.005, 0.01),
        ]
    )

    assert result["tiers"]["near"]["decision"] == "PASS"
    assert result["tiers"]["medium"]["decision"] == "FAIL"
    assert result["passed_primary_tiers"] == ["near"]
    assert result["decision"] == "PASS"
