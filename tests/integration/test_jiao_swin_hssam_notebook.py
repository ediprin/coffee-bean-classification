from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/jiao_swin_hssam_failfast_colab.ipynb")


def test_jiao_notebook_is_resumable_validation_only_and_fail_fast():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "DRIVE_RESULT_FOLDER = 'sni-jiao-hssam-v1'" in source
    assert "run_jiao_swin_hssam_screening" in source
    assert "run_stage('screen', ('SJ0', 'SJFULL'))" in source
    assert "screen['final_decision'] == 'PASS'" in source
    assert "--evaluation-split', 'val'" in source
    assert "source/test" not in source
    assert "time.sleep(60)" in source
