from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/sni_mrenet_failfast_colab.ipynb")


def test_sni_mrenet_notebook_is_validation_only_and_staged():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in payload["cells"]
        for line in cell.get("source", [])
    )
    assert "REPO_REF = 'agent/sni-instance-crops'" in source
    assert "--stage', stage" in source
    assert "run_stage('backbone'" in source
    assert "run_stage('ontology'" in source
    assert "run_stage('bilinear'" in source
    assert "EVALUATION_SPLIT = 'val'" in source
    assert "--evaluation-split', EVALUATION_SPLIT" in source
    assert "--split', 'test'" not in source
    assert "SEEDS = [42]" in source
    assert "output_root = Path('/content/drive/MyDrive')" in source
