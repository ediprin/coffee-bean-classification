import csv
import json
from pathlib import Path

from bilinear_lmmd.experiments.run_preprocessing_analysis import run_analysis
from bilinear_lmmd.experiments.run_preprocessing_bootstrap import run_bootstrap


def _master(path: Path):
    classes = ["A", "B"]
    rows = []
    for index in range(10):
        actual = classes[index % 2]
        rows.append(
            {
                "identity": f"{actual}/{index}.png",
                "actual": actual,
                "R0_predicted": actual if index not in {2, 7} else classes[1 - index % 2],
                "R0_correct": "",
                "C0_predicted": actual if index != 2 else classes[1 - index % 2],
                "C0_correct": "",
                "F0_predicted": actual if index not in {1, 2, 7} else classes[1 - index % 2],
                "F0_correct": "",
                "W0_predicted": actual if index != 7 else classes[1 - index % 2],
                "W0_correct": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_paired_bootstrap_and_per_class_analysis(tmp_path):
    master = tmp_path / "primary_oof_table.csv"
    _master(master)
    boot = run_bootstrap(master, tmp_path / "bootstrap.json", iterations=200, random_seed=1)
    assert boot["samples"] == 10
    assert set(boot["point_delta_vs_R0"]) == {"C0", "F0", "W0"}
    result = run_analysis(master, tmp_path / "analysis")
    assert result["samples"] == 10
    assert (tmp_path / "analysis/per_class_effects.csv").is_file()
    assert (tmp_path / "analysis/C0_minus_R0_confusion.csv").is_file()
