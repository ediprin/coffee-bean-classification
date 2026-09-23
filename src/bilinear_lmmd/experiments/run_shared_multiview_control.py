from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    current_git_commit,
    model_state_fingerprint,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.engine.preprocessing_study import train_preprocessing_study
from bilinear_lmmd.engine.shared_multiview import (
    validation_identity_label_sha256,
)
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-shared-multiview-r0-control-v1"


def _json(path: Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return False
    import torch
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def run_control(
    *,
    config_path: Path,
    authority_path: Path,
    fold: int,
    data_root: Path,
    output_root: Path,
    required_commit: str,
    device: str = "auto",
    resume: bool = False,
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari frozen commit: {actual_commit} != {required_commit}"
        )

    cfg = copy.deepcopy(load_config(config_path))
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())

    if cfg["preprocessing"]["code"] != "R0" or cfg["preprocessing"]["method"] != "raw":
        raise RuntimeError("R0 control harus raw.")
    if cfg["model"]["backbone"] != "mobilenetv3_large_100" or cfg["model"]["head"] != "gap":
        raise RuntimeError("R0 control model contract berubah.")
    if int(cfg["training"]["epochs"]) != 50:
        raise RuntimeError("R0 control harus 50 epoch.")

    authority = _json(authority_path, "Validation authority")
    fold_key = f"fold_{fold}"
    expected = authority["folds"][fold_key]
    count, val_sha = validation_identity_label_sha256(data_root)
    if count != int(expected["count"]) or val_sha != expected["identity_label_sha256"]:
        raise RuntimeError("Validation split tidak cocok dengan authority lama.")

    seed_everything(42)
    probe = build_model(copy.deepcopy(cfg["model"]))
    current_initial = model_state_fingerprint(probe)
    del probe
    historical_initial = authority["expected_initial_model_state_sha256"]

    run_dir = (
        Path(output_root).expanduser().resolve()
        / "R0_CONTROL"
        / f"fold_{fold}"
        / "seed42"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["output_dir"] = str(run_dir)

    contract = {
        "format": "bilinear_lmmd.shared_multiview.r0_control_contract.v1",
        "protocol": PROTOCOL,
        "scientific_scope": "matched same-environment control",
        "fold": int(fold),
        "seed": 42,
        "git_commit": actual_commit,
        "validation_authority_sha256": sha256_file(authority_path),
        "validation_identity_label_sha256": val_sha,
        "validation_count": count,
        "current_initial_model_state_sha256": current_initial,
        "historical_initial_model_state_sha256": historical_initial,
        "historical_initial_match": current_initial == historical_initial,
        "training_view": "R0",
        "deployment_view": "R0",
        "epochs": int(cfg["training"]["epochs"]),
        "evaluation_split_during_training": "val",
        "test_images_accessed": False,
        "resolved_config_sha256": canonical_json_sha256(cfg),
    }
    contract_sha = canonical_json_sha256(contract)
    contract["run_contract_sha256"] = contract_sha

    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        if _json(contract_path, "Existing control contract") != contract:
            raise RuntimeError("Existing R0 control contract berbeda.")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        (run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    result_path = run_dir / "result.json"
    if result_path.is_file():
        old = _json(result_path, "Existing R0 control result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing R0 control result berasal dari kontrak berbeda.")
        return old

    lock_name = f"R0_CONTROL_fold{fold}_seed42.training.lock"
    with exclusive_training_lock(output_root, lock_name=lock_name, stale_seconds=900):
        complete = _run_complete(run_dir, int(cfg["training"]["epochs"]))
        training_executed = False
        if not complete:
            train_preprocessing_study(
                cfg,
                run_dir=run_dir,
                run_contract_sha256=contract_sha,
                expected_initial_model_sha256=current_initial,
                resume=resume or (run_dir / "last.pt").is_file(),
            )
            training_executed = True

    if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
        raise RuntimeError("R0 control belum selesai.")

    metrics = _json(run_dir / "validation" / "metrics.json", "R0 control metrics")
    result = {
        "format": "bilinear_lmmd.shared_multiview.r0_control_result.v1",
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": 42,
        "metrics": {
            key: metrics[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "worst_class_f1",
                "hard_class_f1",
            )
        },
        "historical_r0_baseline": expected["r0_baseline"],
        "best_checkpoint": str(run_dir / "best.pt"),
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_control(
        config_path=args.config,
        authority_path=args.authority,
        fold=args.fold,
        data_root=args.data_root,
        output_root=args.output_root,
        required_commit=args.required_commit,
        device=args.device,
        resume=args.resume,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
