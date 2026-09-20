from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    current_git_commit,
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.engine.preprocessing_study import train_preprocessing_study
from bilinear_lmmd.experiments.preprocessing_contract import (
    CONFIGS,
    validate_development,
    validate_environment,
    validate_observability,
    validate_primary_configs,
    validate_static_and_equivalence,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return False
    import torch

    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def run_arm(
    *,
    arm: str,
    fold: int,
    seed: int,
    data_root: Path,
    development_contract: Path,
    static_preflight: Path,
    observability_audit: Path,
    environment: Path,
    output_root: Path,
    required_commit: str,
    equivalence: Path | None = None,
    smoke_epochs: int | None = None,
    authorize_training: bool = False,
    device: str = "auto",
) -> dict:
    arm = arm.upper()
    if arm not in CONFIGS:
        raise ValueError(f"Arm tidak dikenal: {arm}")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training.")
    if seed != 42:
        raise ValueError(
            "Primary/smoke preprocessing contract awal dikunci seed 42."
        )
    if not required_commit.strip():
        raise RuntimeError(
            "required_commit wajib diisi dengan commit branch eksperimen."
        )

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari frozen commit: "
            f"{actual_commit} != {required_commit}"
        )

    config_gate = validate_primary_configs()
    development = validate_development(data_root, development_contract)
    if development["fold"] != fold:
        raise RuntimeError(
            f"Development fold {development['fold']} "
            f"tidak cocok dengan --fold {fold}."
        )
    observability = validate_observability(observability_audit)
    static = validate_static_and_equivalence(
        static_preflight, arm, equivalence
    )
    environment_gate = validate_environment(environment)

    cfg = copy.deepcopy(config_gate["configs"][arm])
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).resolve())
    mode = "smoke" if smoke_epochs is not None else "primary"
    if smoke_epochs is not None:
        if smoke_epochs <= 0 or smoke_epochs > 3:
            raise ValueError("Smoke epochs harus 1..3.")
        cfg["training"]["epochs"] = int(smoke_epochs)

    run_dir = (
        Path(output_root).expanduser().resolve()
        / mode
        / arm
        / f"fold_{fold}"
        / f"seed{seed}"
    )
    cfg["training"]["output_dir"] = str(run_dir)

    frozen_config_path = run_dir / "run_config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_config_text = yaml.safe_dump(
        cfg, sort_keys=False, allow_unicode=True
    )
    frozen_config_sha = canonical_json_sha256(cfg)

    contract = {
        "format": "bilinear_lmmd.preprocessing.arm_contract.v2",
        "protocol": (
            "coffee17-preprocessing-smoke-v1"
            if smoke_epochs is not None
            else "coffee17-preprocessing-primary-v1"
        ),
        "scientific_evidence": smoke_epochs is None,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "git_commit": actual_commit,
        "clean_content_sha256": development["clean_content_sha256"],
        "fold_manifest_sha256": development["fold_manifest_sha256"],
        "development_contract_sha256": development[
            "development_contract_sha256"
        ],
        "observability_audit_sha256": observability[
            "observability_audit_sha256"
        ],
        "static_preflight_sha256": static["static_preflight_sha256"],
        "reference_equivalence_sha256": static[
            "reference_equivalence_sha256"
        ],
        "environment_reference_sha256": environment_gate[
            "environment_reference_sha256"
        ],
        "software_sha256": environment_gate["software_sha256"],
        "common_config_sha256": config_gate["common_config_sha256"],
        "arm_config_sha256": config_gate["arm_config_sha256"][arm],
        "resolved_run_config_sha256": frozen_config_sha,
        "common_initialized_model_state_sha256": static[
            "expected_initial_model_sha256"
        ],
        "epochs": int(cfg["training"]["epochs"]),
        "evaluation_split_during_training": "val",
        "test_images_accessed": False,
    }
    contract_sha = canonical_json_sha256(contract)
    contract["run_contract_sha256"] = contract_sha

    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        old = _json(contract_path)
        if old != contract:
            raise RuntimeError(
                "Existing run directory memiliki run-contract berbeda."
            )
    else:
        frozen_config_path.write_text(
            frozen_config_text, encoding="utf-8"
        )
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )

    result_path = run_dir / "result.json"
    if result_path.is_file():
        old = _json(result_path)
        if old.get("run_contract") != contract:
            raise RuntimeError(
                "Existing result berasal dari kontrak berbeda."
            )
        return old

    lock_name = f"{arm}_fold{fold}_seed{seed}.{mode}.training.lock"
    with exclusive_training_lock(output_root, lock_name=lock_name):
        complete = _run_complete(
            run_dir, int(cfg["training"]["epochs"])
        )
        training_executed = False
        if not complete:
            train_preprocessing_study(
                cfg,
                run_dir=run_dir,
                run_contract_sha256=contract_sha,
                expected_initial_model_sha256=static[
                    "expected_initial_model_sha256"
                ],
                resume=(run_dir / "last.pt").is_file(),
            )
            training_executed = True

    if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
        raise RuntimeError("Run belum selesai secara valid.")
    metrics = _json(run_dir / "validation/metrics.json")
    result = {
        "format": "bilinear_lmmd.preprocessing.arm_result.v2",
        "protocol": contract["protocol"],
        "scientific_evidence": contract["scientific_evidence"],
        "arm": arm,
        "fold": fold,
        "seed": seed,
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
        "best_checkpoint": str(run_dir / "best.pt"),
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--development-contract", required=True, type=Path
    )
    parser.add_argument("--static-preflight", required=True, type=Path)
    parser.add_argument(
        "--observability-audit", required=True, type=Path
    )
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--equivalence", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_arm(
        arm=args.arm,
        fold=args.fold,
        seed=args.seed,
        data_root=args.data_root,
        development_contract=args.development_contract,
        static_preflight=args.static_preflight,
        observability_audit=args.observability_audit,
        environment=args.environment,
        equivalence=args.equivalence,
        output_root=args.output_root,
        required_commit=args.required_commit,
        smoke_epochs=args.smoke_epochs,
        authorize_training=args.authorize_training,
        device=args.device,
    )


if __name__ == "__main__":
    main()
