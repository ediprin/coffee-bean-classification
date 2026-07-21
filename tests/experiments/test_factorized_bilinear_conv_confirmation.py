from __future__ import annotations

import json

import pytest

from bilinear_lmmd.experiments.run_factorized_bilinear_conv_confirmation import (
    CONFIRMATION_SEEDS,
    REQUIRED_COMPARISONS,
    _require_passed_screening,
    confirmation_decision,
)


def _summary(
    macro: float,
    hard: float,
    worst: float,
    macro_wins: int = 2,
    hard_wins: int = 2,
) -> dict:
    return {
        "macro_f1": {"delta_mean": macro, "improved_seeds": macro_wins},
        "hard_class_f1": {"delta_mean": hard, "improved_seeds": hard_wins},
        "worst_class_f1": {"delta_mean": worst, "improved_seeds": 2},
    }


def test_fbconv_confirmation_protocol_is_locked() -> None:
    assert CONFIRMATION_SEEDS == (42, 123, 2026)
    assert REQUIRED_COMPARISONS == ("BE2G_vs_FB1", "FB0_vs_FB1")


def test_confirmation_requires_positive_mean_two_seed_wins_and_worst() -> None:
    assert confirmation_decision(_summary(0.01, 0.02, -0.01))["decision"] == "PASS"
    assert confirmation_decision(_summary(0.01, 0.02, -0.01, 1, 2))["decision"] == "FAIL"
    assert confirmation_decision(_summary(0.01, 0.02, -0.01, 2, 1))["decision"] == "FAIL"
    assert confirmation_decision(_summary(-0.001, 0.02, 0.03))["decision"] == "FAIL"
    assert confirmation_decision(_summary(0.01, 0.02, -0.011))["decision"] == "FAIL"


def test_confirmation_requires_passed_seed42_report(tmp_path) -> None:
    report_dir = tmp_path / "val_reports"
    report_dir.mkdir()
    path = report_dir / "fbconv_screening_decision.json"
    path.write_text(
        json.dumps({"final_decision": "PASS", "seeds": [42]}),
        encoding="utf-8",
    )
    assert _require_passed_screening(tmp_path)["final_decision"] == "PASS"

    path.write_text(
        json.dumps({"final_decision": "FAIL", "seeds": [42]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="belum berstatus PASS"):
        _require_passed_screening(tmp_path)
