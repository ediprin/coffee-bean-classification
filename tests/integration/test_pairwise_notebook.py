from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/coffee17_confusion_pairwise_colab.ipynb")


def test_pairwise_notebook_is_valid_and_code_compiles():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 5
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec")


def test_pairwise_notebook_preserves_validation_and_resume_guards():
    text = NOTEBOOK.read_text(encoding="utf-8")
    for required in (
        "EVALUATION_SPLIT = 'val'",
        "run_confusion_pair_audit",
        "run_pairwise_contrastive_screening",
        "HF_TOKEN",
        "prepare_clean_grouped_folds",
        "last.pt",
        "HEARTBEAT",
    ):
        assert required in text
