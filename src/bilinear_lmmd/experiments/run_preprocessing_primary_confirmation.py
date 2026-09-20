from __future__ import annotations

import argparse
import json
from pathlib import Path

from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    sha256_file,
)
from bilinear_lmmd.experiments.preprocessing_contract import ARMS


FOLDS = (1, 2, 3, 4, 5)
SEED = 42


def _json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_run_dir(
    experiments_root: Path, arm: str, fold: int
) -> Path:
    return (
        experiments_root
        / "primary"
        / arm
        / f"fold_{fold}"
        / f"seed{SEED}"
    )


def build_primary_confirmation(
    experiments_root: Path,
    clean_manifest_path: Path,
    fold_manifest_path: Path,
    output: Path,
) -> dict:
    experiments_root = Path(experiments_root).expanduser().resolve()
    clean = _json(
        Path(clean_manifest_path).resolve(), "Clean manifest"
    )
    folds = _json(
        Path(fold_manifest_path).resolve(), "Fold manifest"
    )

    if folds.get("decision") != "PASS":
        raise RuntimeError("Fold manifest belum PASS.")
    if (
        clean.get("clean_content_sha256")
        != folds.get("clean_content_sha256")
    ):
        raise RuntimeError(
            "Clean/fold content fingerprint berbeda."
        )

    records = []
    invariant_keys = (
        "git_commit",
        "clean_content_sha256",
        "fold_manifest_sha256",
        "observability_audit_sha256",
        "static_preflight_sha256",
        "environment_reference_sha256",
        "software_sha256",
        "common_config_sha256",
        "common_initialized_model_state_sha256",
    )
    invariant_values: dict[str, set[str]] = {
        key: set() for key in invariant_keys
    }

    for arm in ARMS:
        for fold in FOLDS:
            run_dir = _expected_run_dir(
                experiments_root, arm, fold
            )
            result_path = run_dir / "result.json"
            contract_path = run_dir / "run_contract.json"
            checkpoint_path = run_dir / "best.pt"

            result = _json(
                result_path, f"{arm} fold {fold} result"
            )
            contract = _json(
                contract_path, f"{arm} fold {fold} contract"
            )

            gates = {
                "primary_protocol": result.get("protocol")
                == "coffee17-preprocessing-primary-v1",
                "scientific_evidence": result.get(
                    "scientific_evidence"
                )
                is True,
                "arm_exact": result.get("arm") == arm,
                "fold_exact": int(result.get("fold", -1))
                == fold,
                "seed_exact": int(result.get("seed", -1))
                == SEED,
                "validation_only": result.get(
                    "evaluation_split"
                )
                == "val",
                "test_not_accessed": result.get(
                    "test_images_accessed"
                )
                is False,
                "contract_exact": result.get(
                    "run_contract"
                )
                == contract,
                "contract_primary": contract.get("protocol")
                == "coffee17-preprocessing-primary-v1",
                "contract_test_false": contract.get(
                    "test_images_accessed"
                )
                is False,
                "contract_scientific": contract.get(
                    "scientific_evidence"
                )
                is True,
                "epochs_exact": int(
                    contract.get("epochs", -1)
                )
                == 50,
                "checkpoint_exists": checkpoint_path.is_file(),
            }
            if gates["checkpoint_exists"]:
                checkpoint_sha = sha256_file(checkpoint_path)
                gates["checkpoint_hash_exact"] = (
                    checkpoint_sha
                    == result.get("best_checkpoint_sha256")
                )
            else:
                checkpoint_sha = None
                gates["checkpoint_hash_exact"] = False

            if not all(gates.values()):
                raise RuntimeError(
                    f"Primary run invalid {arm}/fold{fold}: "
                    f"{[key for key, value in gates.items() if not value]}"
                )

            for key in invariant_keys:
                value = contract.get(key)
                if value is None:
                    raise RuntimeError(
                        f"{arm}/fold{fold} kehilangan invariant {key}"
                    )
                invariant_values[key].add(str(value))

            records.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "seed": SEED,
                    "result_sha256": sha256_file(result_path),
                    "run_contract_sha256": sha256_file(
                        contract_path
                    ),
                    "checkpoint_relative_path": (
                        Path("primary")
                        / arm
                        / f"fold_{fold}"
                        / f"seed{SEED}"
                        / "best.pt"
                    ).as_posix(),
                    "checkpoint_sha256": checkpoint_sha,
                    "validation_macro_f1": float(
                        result["metrics"]["macro_f1"]
                    ),
                }
            )

    inconsistent = {
        key: sorted(values)
        for key, values in invariant_values.items()
        if len(values) != 1
    }
    if inconsistent:
        raise RuntimeError(
            f"Primary invariants berbeda antar-run: {inconsistent}"
        )

    clean_sha = clean["clean_content_sha256"]
    if (
        next(iter(invariant_values["clean_content_sha256"]))
        != clean_sha
    ):
        raise RuntimeError(
            "Primary runs tidak berasal dari clean population authority."
        )
    if (
        next(iter(invariant_values["fold_manifest_sha256"]))
        != sha256_file(Path(fold_manifest_path).resolve())
    ):
        raise RuntimeError(
            "Primary runs tidak berasal dari fold-manifest file ini."
        )

    payload = {
        "format": "bilinear_lmmd.preprocessing.primary_confirmation.v2",
        "protocol": "coffee17-preprocessing-primary-v1",
        "decision": "AUTHORIZE_OOF_TEST_EVALUATION",
        "required_arms": list(ARMS),
        "required_folds": list(FOLDS),
        "seed": SEED,
        "completed_runs": len(records),
        "clean_content_sha256": clean_sha,
        "clean_manifest_sha256": sha256_file(
            Path(clean_manifest_path).resolve()
        ),
        "fold_manifest_sha256": sha256_file(
            Path(fold_manifest_path).resolve()
        ),
        "invariants": {
            key: next(iter(values))
            for key, values in invariant_values.items()
        },
        "primary_runs": records,
        "all_validation_only": True,
        "all_run_contracts_valid": True,
        "training_executed_by_confirmation": False,
        "inference_executed_by_confirmation": False,
        "test_images_accessed": False,
        "further_primary_tuning_authorized": False,
    }
    payload["confirmation_content_sha256"] = (
        canonical_json_sha256(payload)
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-root", required=True, type=Path
    )
    parser.add_argument(
        "--clean-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--fold-manifest", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_primary_confirmation(
        args.experiments_root,
        args.clean_manifest,
        args.fold_manifest,
        args.output,
    )


if __name__ == "__main__":
    main()
