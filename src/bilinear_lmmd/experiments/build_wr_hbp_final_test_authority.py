from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from bilinear_lmmd.core.reproducibility import sha256_file

FOLDS = (1, 2, 3, 4, 5)


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
        for arm in ("R0_HBP", "WR_HBP"):
            best = pair / arm / "best.pt"
            digest = sha256_file(best)
            expected = result.get("best_checkpoint_sha256", {}).get(arm)
            if digest != expected:
                raise RuntimeError(
                    f"Fold {fold} {arm}: SHA checkpoint tidak cocok."
                )
            actual[arm] = digest

        for metric in deltas:
            deltas[metric].append(float(result["DELTA_WR_MINUS_R0"][metric]))

        checkpoint_sha[str(fold)] = actual
        records[str(fold)] = {
            "pair_result_sha256": sha256_file(pair / "pair_result.json"),
            "best_checkpoint_sha256": actual,
        }

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
    args = parser.parse_args()
    build_authority(
        development_root=args.development_root,
        clean_manifest_path=args.clean_manifest,
        fold_manifest_path=args.fold_manifest,
        output=args.output,
    )


if __name__ == "__main__":
    main()
