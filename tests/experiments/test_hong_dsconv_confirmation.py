from __future__ import annotations

from bilinear_lmmd.experiments.run_hong_dsconv_confirmation import (
    CONFIRMATION_SEEDS,
    confirmation_decision,
)


def _summary(
    macro: float,
    macro_seeds: int,
    hard: float,
    hard_seeds: int,
    worst: float,
) -> dict:
    return {
        "macro_f1": {"delta_mean": macro, "improved_seeds": macro_seeds},
        "hard_class_f1": {"delta_mean": hard, "improved_seeds": hard_seeds},
        "worst_class_f1": {"delta_mean": worst, "improved_seeds": 1},
    }


def test_confirmation_seeds_are_locked() -> None:
    assert CONFIRMATION_SEEDS == (42, 123, 2026)


def test_confirmation_requires_mean_and_seed_consistency() -> None:
    passing = _summary(0.01, 2, 0.02, 2, -0.01)
    assert confirmation_decision(passing)["decision"] == "PASS"

    inconsistent = _summary(0.01, 1, 0.02, 2, 0.03)
    assert confirmation_decision(inconsistent)["decision"] == "FAIL"

    harmed_worst = _summary(0.01, 3, 0.02, 3, -0.011)
    assert confirmation_decision(harmed_worst)["decision"] == "FAIL"
