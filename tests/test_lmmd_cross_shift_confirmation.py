import json

import pytest

from bilinear_lmmd.run_lmmd_cross_shift_confirmation import (
    METRICS,
    source_fingerprint,
    summarize_cross_shift,
)


def _source_tree(root, payload=b"same-image"):
    for split in ("train", "val", "test"):
        path = root / "source" / split / "A" / f"{split}.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload + split.encode("utf-8"))


def _write_metrics(path, value, worst=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {metric: value for metric in METRICS}
    if worst is not None:
        metrics["worst_class_f1"] = worst
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_source_fingerprint_detects_identical_and_changed_sources(tmp_path):
    illumination = tmp_path / "illumination"
    sensor = tmp_path / "sensor"
    background = tmp_path / "background"
    _source_tree(illumination)
    _source_tree(sensor)
    _source_tree(background, payload=b"changed-image")

    assert source_fingerprint(illumination) == source_fingerprint(sensor)
    assert source_fingerprint(illumination) != source_fingerprint(background)


def test_cross_shift_summary_requires_all_domains_and_seeds(tmp_path):
    output = tmp_path / "results"
    for shift in ("sensor", "background"):
        for seed in (42, 2026):
            for model, source_value, target_value, worst in (
                ("M1", 0.86, 0.40, 0.10),
                ("M5w01", 0.84, 0.55, 0.20),
            ):
                root = output / "reports" / shift / f"{model}_seed{seed}"
                _write_metrics(root / "source" / "metrics.json", source_value)
                _write_metrics(
                    root / "target" / "metrics.json", target_value, worst=worst
                )

    result = summarize_cross_shift(
        output, domains=["sensor", "background"], seeds=[42, 2026]
    )
    assert result["decision"] == {
        "passed_domains": 2,
        "total_domains": 2,
        "pass": True,
    }
    sensor_delta = result["domain_results"]["sensor"]["aggregate_delta"]
    assert sensor_delta["target"]["macro_f1"]["mean"] == pytest.approx(0.15)
    assert sensor_delta["source_macro_f1"]["mean"] == pytest.approx(-0.02)

    failing = (
        output
        / "reports"
        / "background"
        / "M5w01_seed2026"
        / "target"
        / "metrics.json"
    )
    metrics = json.loads(failing.read_text(encoding="utf-8"))
    metrics["worst_class_f1"] = 0.0
    failing.write_text(json.dumps(metrics), encoding="utf-8")

    result = summarize_cross_shift(
        output, domains=["sensor", "background"], seeds=[42, 2026]
    )
    assert result["decision"]["pass"] is False
    assert result["decision"]["passed_domains"] == 1
