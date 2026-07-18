import json

import pytest

from bilinear_lmmd.experiments.run_lmmd_rescue_confirmation import (
    METRICS,
    summarize_confirmation,
)


def _write_metrics(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({metric: values[metric] for metric in METRICS}),
        encoding="utf-8",
    )


def test_confirmation_requires_every_held_out_seed_to_pass(tmp_path):
    output = tmp_path / "confirmation"
    for seed in (42, 2026):
        baseline_source = {metric: 0.85 for metric in METRICS}
        baseline_target = {metric: 0.50 for metric in METRICS}
        rescue_source = {metric: 0.83 for metric in METRICS}
        rescue_target = {metric: 0.60 for metric in METRICS}
        rescue_target["worst_class_f1"] = 0.20
        for model, source, target in (
            ("M1", baseline_source, baseline_target),
            ("M5w01", rescue_source, rescue_target),
        ):
            _write_metrics(
                output / "reports" / f"{model}_seed{seed}" / "source" / "metrics.json",
                source,
            )
            _write_metrics(
                output / "reports" / f"{model}_seed{seed}" / "target" / "metrics.json",
                target,
            )

    result = summarize_confirmation(output, [42, 2026])
    assert result["decision"] == {
        "passed_seeds": 2,
        "total_seeds": 2,
        "pass": True,
    }
    delta = result["aggregate"]["delta_M5w01_vs_M1"]
    assert delta["target"]["macro_f1"]["mean"] == pytest.approx(0.10)
    assert delta["source_macro_f1"]["mean"] == pytest.approx(-0.02)

    failing_path = (
        output
        / "reports"
        / "M5w01_seed2026"
        / "target"
        / "metrics.json"
    )
    failing = json.loads(failing_path.read_text(encoding="utf-8"))
    failing["worst_class_f1"] = 0.0
    failing_path.write_text(json.dumps(failing), encoding="utf-8")

    result = summarize_confirmation(output, [42, 2026])
    assert result["decision"]["pass"] is False
    assert result["decision"]["passed_seeds"] == 1
