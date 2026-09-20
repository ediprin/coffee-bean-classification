import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARMS=("R0","C0","F0","W0")
def source(name):
    path=ROOT/"notebooks"/name; payload=json.loads(path.read_text(encoding="utf-8")); chunks=[]
    for cell in payload["cells"]:
        if cell.get("cell_type")=="code":
            s="".join(cell.get("source",[])); compile(s,str(path),"exec"); chunks.append(s)
    return "\n".join(chunks)
def test_arm_notebooks_follow_detection_pattern_and_keep_test_locked():
    for arm in ARMS:
        s=source(f"Coffee17_{arm}_Seed42_Colab.ipynb")
        assert f"ARM='{arm}'" in s
        assert "BRANCH='codex/preprocessing-study-v1'" in s
        assert "'git','clone','--depth','1','--branch',BRANCH" in s
        assert "resolve_drive_project_root()" in s
        assert "common_setup.lock" in s
        assert "exclusive_training_lock" in s
        assert "audit_coffee17_provenance" in s
        assert "prepare_preprocessing_folds" in s
        assert "materialize_preprocessing_development" in s
        assert "for FOLD in range(1,6)" in s
        assert "run_preprocessing_arm" in s
        assert "--authorize-training" in s
        assert "materialize_preprocessing_test" not in s
        assert "authorize_test" not in s.lower()
def test_nonraw_arms_verify_detection_reference():
    for arm in ("C0","F0","W0"):
        s=source(f"Coffee17_{arm}_Seed42_Colab.ipynb")
        assert "coffee-bean-detection.git" in s and "verify_reference_equivalence" in s
def test_decision_is_training_and_test_free():
    s=source("Coffee17_Preprocessing_Validation_Decision_Colab.ipynb")
    assert "build_primary_confirmation" in s and "--authorize-training" not in s and "run_oof" not in s
def test_oof_is_separate_inference_only():
    s=source("Coffee17_Preprocessing_OOF_Colab.ipynb")
    assert "run_oof" in s and "authorize_test=True" in s and "--authorize-training" not in s
def test_analysis_is_training_free():
    s=source("Coffee17_Preprocessing_Analysis_Colab.ipynb")
    assert "run_preprocessing_bootstrap" in s and "run_preprocessing_final_report" in s and "--authorize-training" not in s
