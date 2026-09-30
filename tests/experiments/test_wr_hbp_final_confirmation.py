from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import yaml

from bilinear_lmmd.core.reproducibility import sha256_file
from bilinear_lmmd.experiments.build_wr_hbp_final_test_authority import (
    build_authority,
)
from bilinear_lmmd.experiments.run_wr_hbp_final_outer_fold import validate_config
from bilinear_lmmd.experiments.run_wr_hbp_final_outer_summary import summarize


CONFIG = Path(
    "configs/wr_hbp_final_confirmation/WR_HBP_FINAL_CONFIRMATION_V1.yaml"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_registered_final_confirmation_config_validates() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(cfg)
    assert cfg["uncertainty"]["paired_stratified_bootstrap_replicates"] == 10000
    assert cfg["evaluation"]["hard_groups"] == {
        "sour_black": ["Partial Black", "Partial Sour", "Full Sour"],
        "shape_withered": ["Withered", "Immature", "Cut"],
        "insect_damage": ["Slight Insect Damage", "Severe Insect Damage"],
    }


def test_authority_requires_exact_completed_development_checkpoints(tmp_path) -> None:
    clean = tmp_path / "clean.json"
    folds = tmp_path / "folds.json"
    _write_json(clean, {"clean_content_sha256": "clean", "clean_count": 10})
    _write_json(
        folds,
        {"decision": "PASS", "clean_content_sha256": "clean"},
    )
    dev = tmp_path / "dev"
    for fold in range(1, 6):
        pair = dev / f"fold_{fold}" / "seed42"
        hashes = {}
        for arm in ("R0_HBP", "WR_HBP"):
            arm_root = pair / arm
            best = arm_root / "best.pt"
            last = arm_root / "last.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            classes = [f"class_{i}" for i in range(17)]
            torch.save(
                {"fold": fold, "arm": arm, "classes": classes, "config": {}},
                best,
            )
            torch.save(
                {"fold": fold, "arm": arm, "classes": classes, "epoch": 50},
                last,
            )
            _write_json(
                arm_root / "run_contract.json",
                {
                    "protocol": "coffee17-wavelet-residual-hbp-v1",
                    "arm": arm,
                    "seed": 42,
                    "git_commit": "01c9212965bc9040ef151204b9404d564f523a0f",
                    "outer_test_accessed": False,
                    "training": {"epochs": 50},
                },
            )
            hashes[arm] = sha256_file(best)
        _write_json(
            pair / "pair_result.json",
            {
                "protocol": "coffee17-wavelet-residual-hbp-v1",
                "seed": 42,
                "matched_core_initialization": True,
                "matched_validation_rows": True,
                "r0_control_retrained": True,
                "wr_candidate_trained": True,
                "outer_test_accessed": False,
                "best_checkpoint_sha256": hashes,
                "DELTA_WR_MINUS_R0": {
                    "macro_f1": 0.01,
                    "hard_class_f1": 0.0,
                    "worst_class_f1": 0.0,
                },
            },
        )
    out = tmp_path / "authority.json"
    result = build_authority(
        development_root=dev,
        clean_manifest_path=clean,
        fold_manifest_path=folds,
        output=out,
    )
    assert result["decision"] == "AUTHORIZE_OOF_TEST_EVALUATION"
    assert result["further_primary_tuning_authorized"] is False
    assert result["test_images_accessed"] is False


def _write_predictions(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["identity", "actual", "predicted", "correct"],
        )
        writer.writeheader()
        for identity, actual, predicted in rows:
            writer.writerow({
                "identity": identity,
                "actual": actual,
                "predicted": predicted,
                "correct": "1" if actual == predicted else "0",
            })


def test_summary_is_one_shot_pooled_oof_and_gate_is_frozen(tmp_path) -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    class_names = [
        "Broken", "Cut", "Fade", "Full Black", "Full Sour", "Fungus Damage",
        "Hull Husk", "Immature", "Normal", "Parchment", "Partial Black",
        "Partial Sour", "Severe Insect Damage", "Shell",
        "Slight Insect Damage", "Withered", "Other",
    ]
    # Replace only the evaluation names that matter with a valid 17-class list.
    class_names[-1] = "Dried Cherry"
    clean = tmp_path / "clean.json"
    clean_rows = []
    for fold in range(1, 6):
        for idx, name in enumerate(class_names):
            clean_rows.append({"identity": f"{name}/f{fold}_{idx}_0.jpg"})
    _write_json(
        clean,
        {
            "clean_content_sha256": "clean-sha",
            "clean_count": 85,
            "images": clean_rows,
        },
    )
    authority = tmp_path / "authority.json"
    _write_json(
        authority,
        {
            "decision": "AUTHORIZE_OOF_TEST_EVALUATION",
            "scope": "wr-hbp-final-confirmation-v1",
            "clean_content_sha256": "clean-sha",
            "clean_count": 85,
        },
    )
    authority_sha = sha256_file(authority)
    root = tmp_path / "outer"
    for fold in range(1, 6):
        fold_dir = root / f"fold_{fold}"
        rows_r0 = []
        rows_wr = []
        for idx, name in enumerate(class_names):
            for rep in range(1):
                identity = f"{name}/f{fold}_{idx}_{rep}.jpg"
                # Candidate rescues one deterministic sample per fold.
                r0_pred = class_names[(idx + 1) % len(class_names)] if idx == 0 else name
                wr_pred = name
                rows_r0.append((identity, name, r0_pred))
                rows_wr.append((identity, name, wr_pred))
        _write_predictions(fold_dir / "R0_HBP/predictions.csv", rows_r0)
        _write_predictions(fold_dir / "WR_HBP/predictions.csv", rows_wr)
        _write_json(
            fold_dir / "fold_result.json",
            {
                "protocol": "wr-hbp-final-confirmation-v1",
                "training_executed": False,
                "outer_test_accessed": True,
                "classes": class_names,
                "authority_sha256": authority_sha,
                "DELTA_WR_MINUS_R0": {"macro_f1": 0.01},
            },
        )
    output = tmp_path / "summary.json"
    result = summarize(
        output_root=root,
        authority_path=authority,
        config_path=CONFIG,
        clean_manifest_path=clean,
        output=output,
        bootstrap_replicates_override=20,
    )
    assert result["oof_identity_count"] == 85
    assert result["oof_identity_unique"] is True
    assert result["confirmation_gate"]["decision"] == "CONFIRMED_WR_HBP"
    assert result["further_tuning_authorized"] is False


def test_final_config_rejects_changed_bootstrap_contract() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["uncertainty"]["paired_stratified_bootstrap_replicates"] = 9999
    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "bootstrap" in str(exc).lower()
    else:
        raise AssertionError("Changed bootstrap contract harus ditolak")
