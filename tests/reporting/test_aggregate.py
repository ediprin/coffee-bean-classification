import pytest

from bilinear_lmmd.reporting.aggregate_ablation import _paired_summary


def test_paired_summary_counts_consistent_improvements():
    result = _paired_summary([0.70, 0.72, 0.71], [0.74, 0.73, 0.75])
    assert result["delta_mean"] == pytest.approx(0.03)
    assert result["improved_seeds"] == 3
    assert result["total_seeds"] == 3
