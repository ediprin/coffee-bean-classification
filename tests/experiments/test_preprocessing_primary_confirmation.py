import json
from pathlib import Path

from bilinear_lmmd.core.reproducibility import sha256_file
from bilinear_lmmd.experiments.run_preprocessing_primary_confirmation import (
    build_primary_confirmation,
)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_confirmation_requires_and_accepts_exact_20_primary_runs(tmp_path):
    experiments = tmp_path / "experiments"
    clean = tmp_path / "clean.json"
    folds = tmp_path / "folds.json"
    clean_sha = "clean-population"
    _write_json(
        clean,
        {
            "clean_content_sha256": clean_sha,
            "clean_count": 10,
            "images": [],
        },
    )
    _write_json(
        folds,
        {
            "decision": "PASS",
            "clean_content_sha256": clean_sha,
            "assignments": {f"fold_{i}": {} for i in range(1, 6)},
        },
    )
    fold_file_sha = sha256_file(folds)
    invariants = {
        "git_commit": "abc",
        "clean_content_sha256": clean_sha,
        "fold_manifest_sha256": fold_file_sha,
        "static_preflight_sha256": "static",
        "environment_reference_sha256": "env",
        "software_sha256": "soft",
        "common_config_sha256": "common",
        "common_initialized_model_state_sha256": "init",
    }

    for arm in ("R0", "C0", "F0", "W0"):
        for fold in range(1, 6):
            run = experiments / "primary" / arm / f"fold_{fold}" / "seed42"
            run.mkdir(parents=True, exist_ok=True)
            checkpoint = run / "best.pt"
            checkpoint.write_bytes(f"{arm}-{fold}".encode())
            contract = {
                "protocol": "coffee17-preprocessing-primary-v1",
                "scientific_evidence": True,
                "arm": arm,
                "fold": fold,
                "seed": 42,
                "epochs": 50,
                "evaluation_split_during_training": "val",
                "test_images_accessed": False,
                **invariants,
            }
            _write_json(run / "run_contract.json", contract)
            _write_json(
                run / "result.json",
                {
                    "protocol": "coffee17-preprocessing-primary-v1",
                    "scientific_evidence": True,
                    "arm": arm,
                    "fold": fold,
                    "seed": 42,
                    "evaluation_split": "val",
                    "test_images_accessed": False,
                    "best_checkpoint_sha256": sha256_file(checkpoint),
                    "metrics": {"macro_f1": 0.5},
                    "run_contract": contract,
                },
            )

    output = tmp_path / "authority.json"
    result = build_primary_confirmation(experiments, clean, folds, output)
    assert result["decision"] == "AUTHORIZE_OOF_TEST_EVALUATION"
    assert result["completed_runs"] == 20
    assert result["test_images_accessed"] is False
    assert result["further_primary_tuning_authorized"] is False
