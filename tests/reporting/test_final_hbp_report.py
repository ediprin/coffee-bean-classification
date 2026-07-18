import json

from bilinear_lmmd.experiments.run_final_hbp_report import generate_final_report


def _report(value, class_values):
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "macro_f1": value,
        "worst_class_f1": min(class_values),
        "hard_class_f1": value,
        "hard_groups": {"hard": value},
        "classes": ["a", "b"],
        "per_class": {
            "a": {"f1": class_values[0]},
            "b": {"f1": class_values[1]},
        },
    }


def test_final_report_writes_seed_class_and_markdown_artifacts(tmp_path):
    report_root = tmp_path / "reports"
    for seed, old, new in (
        (42, _report(0.7, [0.6, 0.8]), _report(0.8, [0.7, 0.9])),
        (123, _report(0.8, [0.7, 0.9]), _report(0.85, [0.75, 0.95])),
    ):
        for model, report in (("M0", old), ("M1", new)):
            destination = report_root / f"{model}_seed{seed}"
            destination.mkdir(parents=True)
            (destination / "metrics.json").write_text(json.dumps(report))

    output = tmp_path / "final"
    result = generate_final_report(
        report_root,
        output,
        [42, 123],
        include_benchmark=False,
    )

    assert result["models"]["M1"]["macro_f1"]["mean"] == 0.825
    assert result["paired_comparison"]["summary"]["macro_f1"]["delta_mean"] > 0
    assert (output / "final_summary.json").is_file()
    assert (output / "per_seed.csv").is_file()
    assert (output / "per_class.csv").is_file()
    assert "Macro" not in (output / "FINAL_HBP_REPORT.md").read_text()
