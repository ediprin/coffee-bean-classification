from __future__ import annotations

import argparse
import json
import statistics

import torch
from pathlib import Path

from bilinear_lmmd.core.reproducibility import sha256_file

FOLDS = (1, 2, 3, 4, 5)
DEVELOPMENT_SCIENTIFIC_COMMIT = "01c9212965bc9040ef151204b9404d564f523a0f"


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_authority(
    *,
    development_root: Path,
    clean_manifest_path: Path,
    fold_manifest_path: Path,
    output: Path,
    recovery_development_commit: str | None = None,
) -> dict:
    development_root = Path(development_root).resolve()
    clean = _read(Path(clean_manifest_path).resolve())
    folds = _read(Path(fold_manifest_path).resolve())

    if folds.get("decision") != "PASS":
        raise RuntimeError("Fold manifest belum PASS.")
    if folds.get("clean_content_sha256") != clean.get("clean_content_sha256"):
        raise RuntimeError("Clean/fold fingerprint mismatch.")

    records = {}
    deltas = {"macro_f1": [], "hard_class_f1": [], "worst_class_f1": []}
    checkpoint_sha = {}
    observed_development_commits: set[str] = set()
    recovery_determinism_records: list[dict] = []

    for fold in FOLDS:
        pair = development_root / f"fold_{fold}" / "seed42"
        result = _read(pair / "pair_result.json")
        if result.get("protocol") != "coffee17-wavelet-residual-hbp-v1":
            raise RuntimeError(f"Fold {fold}: protocol development salah.")
        required_true = (
            "matched_core_initialization",
            "matched_validation_rows",
            "r0_control_retrained",
            "wr_candidate_trained",
        )
        for key in required_true:
            if result.get(key) is not True:
                raise RuntimeError(f"Fold {fold}: {key} bukan true.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test development sudah tersentuh.")
        if int(result.get("seed", -1)) != 42:
            raise RuntimeError(f"Fold {fold}: seed development bukan 42.")

        actual = {}
        contracts = {}
        for arm in ("R0_HBP", "WR_HBP"):
            arm_root = pair / arm
            best = arm_root / "best.pt"
            last = arm_root / "last.pt"
            contract_path = arm_root / "run_contract.json"
            digest = sha256_file(best)
            expected = result.get("best_checkpoint_sha256", {}).get(arm)
            if digest != expected:
                raise RuntimeError(
                    f"Fold {fold} {arm}: SHA checkpoint tidak cocok."
                )

            contract = _read(contract_path)
            if contract.get("protocol") != "coffee17-wavelet-residual-hbp-v1":
                raise RuntimeError(f"Fold {fold} {arm}: run contract protocol salah.")
            if contract.get("arm") != arm or int(contract.get("seed", -1)) != 42:
                raise RuntimeError(f"Fold {fold} {arm}: run contract arm/seed salah.")
            development_commit = str(contract.get("git_commit", ""))
            allowed_commits = {DEVELOPMENT_SCIENTIFIC_COMMIT}
            if recovery_development_commit:
                allowed_commits.add(str(recovery_development_commit))
            if development_commit not in allowed_commits:
                raise RuntimeError(
                    f"Fold {fold} {arm}: scientific commit development tidak diizinkan: "
                    f"{development_commit}"
                )
            observed_development_commits.add(development_commit)

            if development_commit != DEVELOPMENT_SCIENTIFIC_COMMIT:
                det = contract.get("strict_determinism")
                if not isinstance(det, dict):
                    raise RuntimeError(
                        f"Fold {fold} {arm}: recovery wajib memiliki strict_determinism."
                    )
                required_det = {
                    "cublas_workspace_config": ":4096:8",
                    "deterministic_algorithms": True,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                }
                for key, expected_value in required_det.items():
                    if det.get(key) != expected_value:
                        raise RuntimeError(
                            f"Fold {fold} {arm}: strict determinism {key} salah."
                        )
                recovery_determinism_records.append(det)
            if contract.get("outer_test_accessed") is not False:
                raise RuntimeError(f"Fold {fold} {arm}: contract menyatakan test tersentuh.")
            if int(contract.get("training", {}).get("epochs", -1)) != 50:
                raise RuntimeError(f"Fold {fold} {arm}: epochs contract bukan 50.")

            last_state = torch.load(last, map_location="cpu", weights_only=False)
            if int(last_state.get("epoch", 0)) < 50:
                raise RuntimeError(f"Fold {fold} {arm}: training belum 50 epoch.")
            if last_state.get("classes") is None:
                raise RuntimeError(f"Fold {fold} {arm}: last checkpoint tanpa classes.")

            best_state = torch.load(best, map_location="cpu", weights_only=False)
            if best_state.get("arm") != arm:
                raise RuntimeError(f"Fold {fold} {arm}: best checkpoint arm salah.")
            if best_state.get("classes") != last_state.get("classes"):
                raise RuntimeError(f"Fold {fold} {arm}: class order best/last berbeda.")

            actual[arm] = digest
            contracts[arm] = {
                "run_contract_sha256": sha256_file(contract_path),
                "last_checkpoint_sha256": sha256_file(last),
                "completed_epoch": int(last_state["epoch"]),
                "development_git_commit": contract["git_commit"],
            }

        for metric in deltas:
            deltas[metric].append(float(result["DELTA_WR_MINUS_R0"][metric]))

        checkpoint_sha[str(fold)] = actual
        records[str(fold)] = {
            "pair_result_sha256": sha256_file(pair / "pair_result.json"),
            "best_checkpoint_sha256": actual,
            "arm_contracts": contracts,
        }

    if len(observed_development_commits) != 1:
        raise RuntimeError(
            "Development folds mencampur scientific commit yang berbeda."
        )
    observed_development_commit = next(iter(observed_development_commits))
    development_mode = (
        "exact_original_checkpoints"
        if observed_development_commit == DEVELOPMENT_SCIENTIFIC_COMMIT
        else "checkpoint_loss_recovery_v1"
    )
    if development_mode == "checkpoint_loss_recovery_v1":
        if not recovery_development_commit:
            raise RuntimeError("Recovery commit tidak diregistrasikan.")
        if observed_development_commit != str(recovery_development_commit):
            raise RuntimeError("Recovery development commit tidak cocok.")
        if len(recovery_determinism_records) != 10:
            raise RuntimeError("Strict determinism recovery tidak lengkap 5 fold x 2 arm.")

    gate = {
        "macro_mean_positive": statistics.mean(deltas["macro_f1"]) > 0.0,
        "macro_positive_folds_at_least_3": sum(v > 0 for v in deltas["macro_f1"]) >= 3,
        "hard_mean_nonnegative": statistics.mean(deltas["hard_class_f1"]) >= 0.0,
        "worst_mean_nonnegative": statistics.mean(deltas["worst_class_f1"]) >= 0.0,
    }
    if not all(gate.values()):
        raise RuntimeError("Development WR-HBP tidak lagi memenuhi frozen gate.")

    payload = {
        "format": "bilinear_lmmd.wr_hbp_final_test_authority.v1",
        "decision": "AUTHORIZE_OOF_TEST_EVALUATION",
        "scope": "wr-hbp-final-confirmation-v1",
        "seed": 42,
        "clean_content_sha256": clean["clean_content_sha256"],
        "clean_count": int(clean["clean_count"]),
        "fold_manifest_file_sha256": sha256_file(Path(fold_manifest_path)),
        "development_gate": gate,
        "development_scientific_commit": observed_development_commit,
        "development_mode": development_mode,
        "recovery_development_commit": (
            str(recovery_development_commit)
            if development_mode == "checkpoint_loss_recovery_v1"
            else None
        ),
        "development_checkpoint_sha256": checkpoint_sha,
        "development_records": records,
        "test_images_accessed": False,
        "further_primary_tuning_authorized": False,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", required=True, type=Path)
    parser.add_argument("--clean-manifest", required=True, type=Path)
    parser.add_argument("--fold-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recovery-development-commit")
    args = parser.parse_args()
    build_authority(
        development_root=args.development_root,
        clean_manifest_path=args.clean_manifest,
        fold_manifest_path=args.fold_manifest,
        output=args.output,
        recovery_development_commit=args.recovery_development_commit,
    )


if __name__ == "__main__":
    main()
