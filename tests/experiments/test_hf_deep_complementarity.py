from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from bilinear_lmmd.experiments.run_hf_deep_complementarity import (
    ARMS,
    METRICS,
    validate_config,
)
from bilinear_lmmd.experiments.run_hf_deep_complementarity_summary import summarize


CONFIG = Path("configs/hf_deep_complementarity/HF_DEEP_COMPLEMENTARITY_V1.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_registered_config_validates() -> None:
    validate_config(_config())


def test_validation_tuning_is_rejected() -> None:
    cfg = _config()
    cfg["probe"]["validation_tuning"] = True
    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "validation_tuning" in str(exc)
    else:
        raise AssertionError("validation_tuning=True harus ditolak")


def test_mrmr_scope_is_train_only() -> None:
    cfg = _config()
    cfg["handcrafted"]["selection_fit"] = "all_data"
    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "selection_fit" in str(exc)
    else:
        raise AssertionError("MRMR di luar train fold harus ditolak")


def _fold_result(fold: int, macro_delta: float = 0.01) -> dict:
    deep = {
        "accuracy": 0.80,
        "balanced_accuracy": 0.80,
        "macro_f1": 0.80,
        "hard_class_f1": 0.70,
        "worst_class_f1": 0.50,
    }
    fusion = {
        "accuracy": 0.81,
        "balanced_accuracy": 0.81,
        "macro_f1": deep["macro_f1"] + macro_delta,
        "hard_class_f1": 0.71,
        "worst_class_f1": 0.51,
    }
    hf = {metric: 0.60 for metric in METRICS}
    delta = {metric: fusion[metric] - deep[metric] for metric in METRICS}
    targeted = {
        "HBP_EMB": {"TOTAL": 10},
        "HF20_HBP_EMB": {"TOTAL": 8},
        "HF20": {"TOTAL": 12},
    }
    return {
        "outer_test_accessed": False,
        "hbp_gpu_smoke": {"passed": True},
        "selected_feature_names": [f"f{i}" for i in range(20)],
        "arms": {
            "HF20": hf,
            "HBP_EMB": deep,
            "HF20_HBP_EMB": fusion,
        },
        "delta_fusion_minus_deep": delta,
        "targeted_confusions": targeted,
        "paired_outcomes_fusion_vs_deep": {
            "count": 100,
            "rescue": 3,
            "damage": 1,
            "both_correct": 79,
            "both_wrong": 17,
            "net_correct": 2,
        },
    }


def test_summary_gate_passes_only_frozen_conditions(tmp_path) -> None:
    root = tmp_path / "runs"
    for fold in range(1, 6):
        folder = root / f"fold_{fold}" / "seed42"
        folder.mkdir(parents=True)
        (folder / "fold_result.json").write_text(
            json.dumps(_fold_result(fold), indent=2),
            encoding="utf-8",
        )
    output = tmp_path / "summary.json"
    result = summarize(output_root=root, config_path=CONFIG, output=output)
    assert result["screening_gate"]["decision"] == "SUPPORT_HF_COMPLEMENTARITY"

    # One negative Macro fold is allowed, two are not.
    for fold in (1, 2):
        path = root / f"fold_{fold}" / "seed42" / "fold_result.json"
        bad = _fold_result(fold, macro_delta=-0.01)
        path.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    result = summarize(output_root=root, config_path=CONFIG, output=output)
    assert result["screening_gate"]["decision"] == "NO_CLEAR_HF_COMPLEMENTARITY"


def test_kaggle_notebook_is_valid_and_pins_audited_code() -> None:
    notebook_path = Path(
        "notebooks/Coffee17_HF_Deep_Complementarity_V1_Kaggle.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert (
        'SCIENTIFIC_CODE_COMMIT = "ac6eceee8319a6fd970d9f1de955bf300d3be115"'
        in code
    )
    assert "--shared-hf-cache" in code
    assert "run_hf_deep_complementarity_summary" in code
    assert "coffee17-hf-deep-complementarity-v1-resume-package.zip" in code
    compile(code, str(notebook_path), "exec")
