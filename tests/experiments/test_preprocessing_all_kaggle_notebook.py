import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "Coffee17_Preprocessing_All_Kaggle.ipynb"


def source():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    chunks = []
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            text = "".join(cell.get("source", []))
            compile(text, str(NOTEBOOK), "exec")
            chunks.append(text)
    return "\n".join(chunks)


def test_all_in_one_kaggle_pipeline_is_kaggle_only_and_complete():
    s = source()
    assert "Path('/kaggle/input')" in s
    assert "Path('/kaggle/working')" in s
    assert "CODE_COMMIT='7de2abb46efd5e71dbab508e2ae4b4e61102a7aa'" in s
    assert "discover_directory_samples(INPUT)" in s
    assert "run_observability_audit" in s
    assert "PASS_PREPROCESSING_OBSERVABILITY_AUDIT" in s
    assert "for ARM in ('R0','C0','F0','W0')" in s
    assert "for FOLD in range(1,6)" in s
    assert "run_preprocessing_arm" in s
    assert "build_primary_confirmation" in s
    assert "AUTHORIZE_OOF_TEST_EVALUATION" in s
    assert "run_oof" in s
    assert "authorize_test=True" in s
    assert "run_preprocessing_bootstrap" in s
    assert "run_preprocessing_analysis" in s
    assert "run_preprocessing_efficiency" in s
    assert "run_preprocessing_final_report" in s
    assert "google.colab" not in s
    assert "drive.mount" not in s
    assert "HF_TOKEN" not in s
    assert "HfApi" not in s
    assert "snapshot_download" not in s
