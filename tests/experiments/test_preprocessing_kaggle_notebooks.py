import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("R0", "C0", "F0", "W0")
CODE_COMMIT = "7de2abb46efd5e71dbab508e2ae4b4e61102a7aa"


def source(name):
    path = ROOT / "notebooks" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            text = "".join(cell.get("source", []))
            compile(text, str(path), "exec")
            chunks.append(text)
    return "\n".join(chunks)


def test_kaggle_arm_notebooks_are_kaggle_only_and_test_locked():
    for arm in ARMS:
        s = source(f"Coffee17_{arm}_Seed42_Kaggle.ipynb")
        assert f"ARM='{arm}'" in s
        assert f"CODE_COMMIT='{CODE_COMMIT}'" in s
        assert "Path('/kaggle/input')" in s
        assert "Path('/kaggle/working')" in s
        assert "discover_directory_samples(INPUT)" in s
        assert "audit_coffee17_provenance" in s
        assert "prepare_preprocessing_folds" in s
        assert "run_observability_audit" in s
        assert "materialize_preprocessing_development" in s
        assert "run_preprocessing_arm" in s
        assert "--authorize-training" in s
        assert "google.colab" not in s
        assert "drive.mount" not in s
        assert "HF_TOKEN" not in s
        assert "huggingface" not in s.lower()
        assert "materialize_preprocessing_test" not in s
        assert "authorize_test" not in s.lower()


def test_kaggle_f0_only_uses_detection_reference():
    f0 = source("Coffee17_F0_Seed42_Kaggle.ipynb")
    assert "coffee-bean-detection.git" in f0
    assert "6ef389c23932e44fe4135c32d471b3008b1cbf39" in f0
    assert "verify_reference_equivalence" in f0
    for arm in ("R0", "C0", "W0"):
        assert "EQUIVALENCE=None" in source(f"Coffee17_{arm}_Seed42_Kaggle.ipynb")


def test_kaggle_decision_is_training_and_test_free():
    s = source("Coffee17_Preprocessing_Validation_Decision_Kaggle.ipynb")
    assert "build_primary_confirmation" in s
    assert "--authorize-training" not in s
    assert "run_oof" not in s
    assert "google.colab" not in s
    assert "huggingface" not in s.lower()


def test_kaggle_oof_is_separate_inference_only():
    s = source("Coffee17_Preprocessing_OOF_Kaggle.ipynb")
    assert "run_oof" in s
    assert "authorize_test=True" in s
    assert "--authorize-training" not in s


def test_kaggle_analysis_is_training_free():
    s = source("Coffee17_Preprocessing_Analysis_Kaggle.ipynb")
    assert "run_preprocessing_bootstrap" in s
    assert "run_preprocessing_final_report" in s
    assert "--authorize-training" not in s
