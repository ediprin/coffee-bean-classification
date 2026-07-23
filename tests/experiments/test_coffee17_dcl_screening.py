import json
from pathlib import Path

import pytest

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_coffee17_dcl_screening import (
    _require_dcl_gate,
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_colab_notebook_is_validation_only_and_persistent():
    notebook_path = (
        Path(__file__).parents[2]
        / "notebooks"
        / "coffee17_dcl_local_detail_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "run_coffee17_dcl_screening" in source
    assert "'--stage', 'dcl'" in source
    assert "'--evaluation-split', 'val'" in source
    assert "'--evaluation-split', 'test'" not in source
    assert "'--artifact-sync-every', '1'" in source
    assert "'--artifact-required'" in source
    assert "SEED = 123" in source


def test_dcl_gate_requires_macro_hard_and_worst():
    assert screening_decision(_summary(0.01, 0.01, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(0.01, -0.01, 0.02))["decision"] == "FAIL"
    assert screening_decision(_summary(0.01, 0.01, -0.011))["decision"] == "FAIL"


def test_contrastive_stage_is_blocked_until_dcl0_passes(tmp_path):
    with pytest.raises(RuntimeError, match="Stage contrastive"):
        _require_dcl_gate(tmp_path, "val")

    report = tmp_path / "val_reports"
    report.mkdir()
    path = report / "dcl_stage_decision.json"
    path.write_text(
        json.dumps({"DCL0_final": {"decision": "FAIL"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tidak lolos"):
        _require_dcl_gate(tmp_path, "val")

    path.write_text(
        json.dumps({"DCL0_final": {"decision": "PASS"}}),
        encoding="utf-8",
    )
    _require_dcl_gate(tmp_path, "val")


def test_dcl_configs_isolate_contrastive_stages():
    dcl0 = load_config("configs/finegrained/DCL0_efficientnetv2_dcl.yaml")
    dcl1 = load_config(
        "configs/finegrained/DCL1_efficientnetv2_dcl_supcon.yaml"
    )
    dcl2 = load_config(
        "configs/finegrained/DCL2_efficientnetv2_dcl_confusion.yaml"
    )

    for cfg in (dcl0, dcl1, dcl2):
        assert cfg["model"]["backbone"] == "tf_efficientnetv2_b0.in1k"
        assert cfg["model"]["head"] == "dcl_gap"
        assert cfg["model"]["dcl_grid_size"] == 7
        assert cfg["adaptation"]["method"] == "source_only"
        assert cfg["training"]["dcl_swap_weight"] == 1.0
        assert cfg["training"]["dcl_layout_weight"] == 1.0

    assert dcl0["training"]["dcl_contrastive_mode"] == "none"
    assert dcl1["training"]["dcl_contrastive_mode"] == "standard"
    assert dcl2["training"]["dcl_contrastive_mode"] == "confusion_aware"
    assert dcl1["training"]["confusion_strength"] == 0.0
    assert dcl2["training"]["confusion_strength"] > 0.0
