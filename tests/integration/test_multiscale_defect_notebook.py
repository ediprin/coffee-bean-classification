from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/coffee17_chang_liu_mde_colab.ipynb")


def test_mde_notebook_is_valid_resumable_and_validation_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_multiscale_defect_screening" in source
    assert "SEEDS = [42]" in source
    assert "EVALUATION_SPLIT = 'val'" in source
    assert "coffee17-chang-liu-mde" in source
    assert "HEARTBEAT" in source
    assert "/content/drive/MyDrive" in source
    assert "HF_TOKEN" in source
    assert "for code in ('BE2G', 'BE2H')" in source
    assert "_seed42" in source
    assert "MDE0" in source and "MDE1" in source
