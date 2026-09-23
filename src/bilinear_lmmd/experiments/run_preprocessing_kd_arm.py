from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    current_git_commit,
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.engine.preprocessing_kd import (
    PROTOCOL,
    TEACHER_MODES,
    train_preprocessing_kd,
    validate_kd_config,
)


def _json(path: Path, label: str) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_kd_arm(
    *,
    config_path: Path,
    fold: int,
    data_root: Path,
    authority_path: Path,
    experiments_root: Path,
    output_root: Path,
    required_commit: str,
    authorize_training: bool = False,
    resume: bool = False,
    device: str = "auto",
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training KD memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")
    if not required_commit.strip():
        raise RuntimeError("required_commit wajib diisi.")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari frozen KD commit: {actual_commit} != {required_commit}"
        )

    cfg = copy.deepcopy(load_config(config_path))
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    validate_kd_config(cfg)

    authority_path = Path(authority_path).expanduser().resolve()
    authority = _json(authority_path, "Primary confirmation")
    if authority.get("decision") != "AUTHORIZE_OOF_TEST_EVALUATION":
        raise RuntimeError("Primary authority tidak valid.")
    if authority.get("further_primary_tuning_authorized") is not False:
        raise RuntimeError("Primary closure tidak valid.")

    teacher_mode = str(cfg["distillation"]["teacher_mode"]).upper()
    if teacher_mode not in TEACHER_MODES:
        raise RuntimeError("teacher_mode tidak valid.")

    run_dir = (
        Path(output_root).expanduser().resolve()
        / teacher_mode
        / f"fold_{fold}"
        / "seed42"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    run_contract = {
        "format": "bilinear_lmmd.preprocessing.kd_run_contract.v1",
        "protocol": PROTOCOL,
        "scientific_scope": "post-primary exploratory",
        "teacher_mode": teacher_mode,
        "fold": int(fold),
        "seed": 42,
        "git_commit": actual_commit,
        "authority_sha256": sha256_file(authority_path),
        "primary_protocol_closed": True,
        "outer_oof_already_opened_before_kd_design": True,
        "student_inference_view": "R0",
        "teacher_views_training_only": (
            ["R0"] if teacher_mode == "R0" else ["R0", "C0", "F0", "W0"]
        ),
        "temperature": float(cfg["distillation"]["temperature"]),
        "hard_weight": float(cfg["distillation"]["hard_weight"]),
        "resolved_config_sha256": canonical_json_sha256(cfg),
        "test_images_accessed": False,
    }
    run_contract_sha = canonical_json_sha256(run_contract)
    run_contract["run_contract_sha256"] = run_contract_sha

    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        old = _json(contract_path, "Existing KD run contract")
        if old != run_contract:
            raise RuntimeError("Existing KD run directory memiliki kontrak berbeda.")
    else:
        contract_path.write_text(
            json.dumps(run_contract, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "resolved_config.json").write_text(
            json.dumps(cfg, indent=2) + "\n",
            encoding="utf-8",
        )

    result_path = run_dir / "result.json"
    if result_path.is_file():
        return _json(result_path, "Existing KD result")

    lock_name = f"preprocessing_kd_{teacher_mode}_fold{fold}_seed42.training.lock"
    with exclusive_training_lock(output_root, lock_name=lock_name):
        result = train_preprocessing_kd(
            cfg,
            fold=fold,
            data_root=data_root,
            authority_path=authority_path,
            experiments_root=experiments_root,
            run_dir=run_dir,
            run_contract_sha256=run_contract_sha,
            resume=resume,
        )

    payload = {
        "format": "bilinear_lmmd.preprocessing.kd_result.v1",
        "protocol": PROTOCOL,
        "scientific_scope": "post-primary exploratory",
        "teacher_mode": teacher_mode,
        "fold": int(fold),
        "seed": 42,
        "validation_macro_f1": result["best_validation_macro_f1"],
        "best_checkpoint": result["best_checkpoint"],
        "initial_model_state_sha256": result["initial_model_state_sha256"],
        "teacher_metadata": result["teacher_metadata"],
        "run_contract": run_contract,
        "training_executed": True,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_kd_arm(
        config_path=args.config,
        fold=args.fold,
        data_root=args.data_root,
        authority_path=args.authority,
        experiments_root=args.experiments_root,
        output_root=args.output_root,
        required_commit=args.required_commit,
        authorize_training=args.authorize_training,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
