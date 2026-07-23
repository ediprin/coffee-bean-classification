from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/sni_instance_crop_preparation_colab.ipynb")


def test_sni_preparation_notebook_builds_v2_without_training() -> None:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in data.get("cells", [])
    )
    assert "prepare_sni_classification_v2" in source
    assert "classification-v2" in source
    assert "--min-eval-samples-per-class" in source
    assert "--min-eval-groups-per-class" in source
    assert "Training allowed" in source
    assert "bilinear_lmmd.engine.train" not in source
