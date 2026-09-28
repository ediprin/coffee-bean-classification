from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    sha256_file,
)
from bilinear_lmmd.data.preparation.materialize_preprocessing_test import (
    materialize_preprocessing_test,
)
from bilinear_lmmd.engine.preprocessing_study import (
    evaluate_preprocessing_checkpoint,
)
from bilinear_lmmd.experiments.preprocessing_contract import ARMS
from bilinear_lmmd.reporting.preprocessing_oof import merge_primary_oof


FOLDS = (1, 2, 3, 4, 5)
SEED = 42


def _json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _authority_run_map(authority: dict) -> dict[tuple[str, int], dict]:
    result = {}
    for row in authority.get("primary_runs", []):
        key = (row["arm"], int(row["fold"]))
        if key in result:
            raise RuntimeError(f"Authority duplicate run: {key}")
        result[key] = row
    return result


def run_oof(
    *,
    canonical_root: Path,
    clean_manifest: Path,
    fold_manifest: Path,
    authority_path: Path,
    experiments_root: Path,
    output_root: Path,
    authorize_test: bool = False,
) -> dict:
    if not authorize_test:
        raise RuntimeError("OOF outer-test memerlukan --authorize-test.")

    authority_path = Path(authority_path).expanduser().resolve()
    authority = _json(authority_path, "Primary confirmation")
    if (
        authority.get("decision") != "AUTHORIZE_OOF_TEST_EVALUATION"
        or authority.get("completed_runs") != 20
        or authority.get("test_images_accessed") is not False
        or authority.get("further_primary_tuning_authorized") is not False
        or authority.get("training_executed_by_confirmation") is not False
        or authority.get("inference_executed_by_confirmation") is not False
    ):
        raise RuntimeError("Primary confirmation tidak mengotorisasi OOF test.")

    authority_sha = sha256_file(authority_path)
    run_map = _authority_run_map(authority)
    expected_keys = {(arm, fold) for arm in ARMS for fold in FOLDS}
    if set(run_map) != expected_keys:
        raise RuntimeError("Authority tidak memuat tepat 20 primary runs.")

    clean = _json(Path(clean_manifest).resolve(), "Clean manifest")
    folds = _json(Path(fold_manifest).resolve(), "Fold manifest")
    if authority.get("clean_content_sha256") != clean.get("clean_content_sha256"):
        raise RuntimeError("Authority clean population mismatch.")
    if authority.get("fold_manifest_sha256") != sha256_file(Path(fold_manifest).resolve()):
        raise RuntimeError("Authority fold-manifest mismatch.")
    if folds.get("clean_content_sha256") != clean.get("clean_content_sha256"):
        raise RuntimeError("Clean/fold manifest mismatch.")

    experiments_root = Path(experiments_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    raw_reports = output_root / "raw_reports"
    raw_reports.mkdir(parents=True, exist_ok=True)

    for fold in FOLDS:
        with tempfile.TemporaryDirectory(prefix=f"coffee17_oof_fold{fold}_") as temp:
            test_root = Path(temp)
            test_contract = materialize_preprocessing_test(
                Path(canonical_root),
                Path(clean_manifest),
                Path(fold_manifest),
                authority_path,
                test_root,
                fold=fold,
                authorize_test=True,
            )
            test_contract_path = test_root / "test_contract.json"
            test_contract_sha = sha256_file(test_contract_path)
            fold_test_ids = folds["assignments"][f"fold_{fold}"]["test"]
            fold_test_content_sha = canonical_json_sha256(sorted(fold_test_ids))

            for arm in ARMS:
                authority_run = run_map[(arm, fold)]
                checkpoint = (
                    experiments_root / authority_run["checkpoint_relative_path"]
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Checkpoint OOF tidak ditemukan: {checkpoint}")
                checkpoint_sha = sha256_file(checkpoint)
                if checkpoint_sha != authority_run["checkpoint_sha256"]:
                    raise RuntimeError(
                        f"Checkpoint hash berubah setelah confirmation: {arm}/fold{fold}"
                    )

                report_dir = raw_reports / f"{arm}_fold{fold}_seed{SEED}"
                report_meta_path = report_dir / "test_report.json"
                expected_meta = {
                    "format": "bilinear_lmmd.preprocessing.test_report.v1",
                    "protocol": "coffee17-preprocessing-primary-v1",
                    "arm": arm,
                    "fold": fold,
                    "seed": SEED,
                    "checkpoint_sha256": checkpoint_sha,
                    "authority_sha256": authority_sha,
                    "test_contract_sha256": test_contract_sha,
                    "fold_test_content_sha256": fold_test_content_sha,
                    "test_count": int(test_contract["test_count"]),
                    "training_executed": False,
                    "test_images_accessed": True,
                }

                if report_meta_path.is_file():
                    cached = _json(report_meta_path, "Cached OOF report")
                    if cached != expected_meta:
                        raise RuntimeError(
                            f"Cached OOF report berbeda kontrak: {arm}/fold{fold}"
                        )
                    for required in (
                        report_dir / "metrics.json",
                        report_dir / "predictions.csv",
                        report_dir / "confusion_matrix.csv",
                    ):
                        if not required.is_file():
                            raise RuntimeError(f"Cached OOF report parsial: {required}")
                    print(f"REUSE OOF: {arm}/fold{fold}", flush=True)
                    continue

                evaluate_preprocessing_checkpoint(
                    checkpoint,
                    data_root=test_root,
                    split="test",
                    output_dir=report_dir,
                )
                report_meta_path.write_text(
                    json.dumps(expected_meta, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"OOF COMPLETE: {arm}/fold{fold}", flush=True)

            if not (test_root / "source/test").exists():
                raise RuntimeError("Test runtime unexpectedly disappeared during inference.")

    summary = merge_primary_oof(
        raw_reports,
        Path(clean_manifest),
        Path(fold_manifest),
        output_root / "merged",
        authority_sha256=authority_sha,
    )
    final = {
        "format": "bilinear_lmmd.preprocessing.oof_execution.v1",
        "protocol": "coffee17-preprocessing-primary-v1",
        "authority_sha256": authority_sha,
        "reports": 20,
        "merged_summary": str(output_root / "merged/primary_oof_summary.json"),
        "sample_count": summary["sample_count"],
        "training_executed": False,
        "test_images_accessed": True,
        "primary_protocol_closed": True,
        "further_primary_tuning_authorized": False,
    }
    (output_root / "oof_execution_summary.json").write_text(
        json.dumps(final, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--clean-manifest", required=True, type=Path)
    parser.add_argument("--fold-manifest", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authorize-test", action="store_true")
    args = parser.parse_args()
    run_oof(
        canonical_root=args.canonical_root,
        clean_manifest=args.clean_manifest,
        fold_manifest=args.fold_manifest,
        authority_path=args.authority,
        experiments_root=args.experiments_root,
        output_root=args.output_root,
        authorize_test=args.authorize_test,
    )


if __name__ == "__main__":
    main()
