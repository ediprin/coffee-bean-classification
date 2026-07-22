import json
from pathlib import Path


NOTEBOOK = Path("notebooks/sni_selective_hbp_diagnostic_colab.ipynb")


def test_selective_hbp_notebook_is_validation_only_and_resumable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_sni_selective_residual_diagnostic" in source
    assert "SNIDG" in source and "SNIDH" in source
    assert "SNIB1_seed42/best.pt" in source
    assert "sni-selective-hbp-diagnostic-v1" in source
    assert "--evaluation-split', 'val'" in source
    assert "test" in source.lower() and "terkunci" in source.lower()
    assert "seed 123" not in source.lower()
    assert "seed 2026" not in source.lower()
