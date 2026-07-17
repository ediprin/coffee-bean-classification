from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/backbone_benchmark_resumable_colab.ipynb")


def test_resumable_notebook_is_valid_and_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 3
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"{NOTEBOOK}:cell-{index}", "exec")


def test_resumable_notebook_contains_recovery_guards() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    for required in (
        "HF_TOKEN",
        "prepare_coffee17",
        "prepare_clean_grouped_folds",
        "--hf-repo",
        "--hf-sync-every",
        "clean_count",
        "last.pt",
    ):
        assert required in text
