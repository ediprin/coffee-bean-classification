from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(
    "notebooks/coffee17_multistage_recalibration_colab.ipynb"
)


def test_multistage_notebook_is_resumable_and_validation_only() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "SEEDS = [42]" in source
    assert "EVALUATION_SPLIT = 'val'" in source
    assert "coffee17-multistage-recalibration-v1" in source
    assert "run_multistage_recalibration_screening" in source
    assert "BILINEAR_LMMD_ARTIFACT_REQUIRED" in source
    assert "'--hf-sync-every', str(HF_SYNC_EVERY)" in source
    assert "MSF0" in source and "MSF1" in source
    assert "MSFC" in source
    assert "'--stage', 'capacity'" in source
    assert "screen['final_decision'] == 'PASS'" in source
    assert "multistage_recalibration_capacity_control.json" in source
    assert "bottom3_class_f1" in source
    assert "source/test" not in source
    assert "time.sleep(60)" in source
