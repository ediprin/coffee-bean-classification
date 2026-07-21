from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/coffee17_hong_dsconv_confirmation_colab.ipynb")


def test_confirmation_notebook_is_valid_and_locked() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "SEEDS = [42, 123, 2026]" in source
    assert "run_hong_dsconv_confirmation" in source
    assert "HCS1" not in source
    assert "HCDS1" not in source
    assert "EVALUATION_SPLIT = 'val'" in source
    assert "--evaluation-split', EVALUATION_SPLIT" in source
    assert "coffee17-hong-classification" in source
