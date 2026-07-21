from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/coffee17_hong_classification_colab.ipynb")


def test_hong_classification_notebook_is_valid_and_validation_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_hong_classification_screening" in source
    assert "MODELS = ['HCD1', 'HCS1', 'HCDS1']" in source
    assert "EVALUATION_SPLIT = 'val'" in source
    assert "--evaluation-split" in source
    assert "coffee17-hong-classification" in source
    assert "HEARTBEAT" in source


def test_hong_notebook_uses_persistent_drive_and_remote_baselines() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "/content/drive/MyDrive" in source
    assert "HF_TOKEN" in source
    assert "BE2G_seed" in source
    assert "BE2H_seed" in source
