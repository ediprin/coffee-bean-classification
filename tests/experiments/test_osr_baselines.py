import json

import pytest

from bilinear_lmmd.experiments.run_osr_baselines import (
    AGGREGATE_METRICS,
    aggregate_osr_summaries,
)


def _summary(seed: int, value: float) -> dict:
    metrics = {name: value for name in AGGREGATE_METRICS}
    return {
        "protocol_id": "test-osr-v1",
        "seed": seed,
        "splits": {
            tier: {"primary_balanced": {"msp": metrics}}
            for tier in ("near", "medium", "far")
        },
    }


def test_aggregate_osr_summaries_writes_mean_std_and_seed_values(tmp_path) -> None:
    result = aggregate_osr_summaries(
        [_summary(42, 0.6), _summary(123, 0.8)],
        tmp_path,
    )

    auroc = result["splits"]["near"]["msp"]["auroc"]
    assert auroc["mean"] == pytest.approx(0.7)
    assert auroc["std"] == pytest.approx(0.02**0.5)
    assert auroc["values"] == {"42": 0.6, "123": 0.8}
    assert (tmp_path / "reports" / "osr_v1_aggregate.csv").is_file()
    saved = json.loads(
        (tmp_path / "reports" / "osr_v1_aggregate.json").read_text()
    )
    assert saved["seeds"] == [42, 123]
