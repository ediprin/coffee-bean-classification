from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "MSF0": Path(
        "configs/finegrained/MSF0_efficientnetv2_fixed_multistage.yaml"
    ),
    "MSF1": Path(
        "configs/finegrained/MSF1_efficientnetv2_adaptive_multistage.yaml"
    ),
}
BASELINES = ("BE2G", "BE2H")
SCREENING_SEEDS = (42,)
REQUIRED_COMPARISONS = ("MSF0_vs_MSF1", "BE2G_vs_MSF1")


def _paired_summary(baseline: list[float], candidate: list[float]) -> dict:
    deltas = [new - old for old, new in zip(baseline, candidate)]
    return {
        "baseline_mean": statistics.mean(baseline),
        "candidate_mean": statistics.mean(candidate),
        "delta_mean": statistics.mean(deltas),
        "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "improved_seeds": sum(delta > 0 for delta in deltas),
        "total_seeds": len(deltas),
        "deltas": deltas,
    }


def _bottom_three_f1(report: dict) -> float:
    scores = sorted(
        float(row["f1"]) for row in report["per_class"].values()
    )
    if len(scores) < 3:
        raise ValueError("Bottom-three F1 membutuhkan sedikitnya tiga kelas.")
    return statistics.mean(scores[:3])


def screening_decision(summary: dict) -> dict:
    criteria = {
        "macro_f1_improved": float(summary["macro_f1"]["delta_mean"]) > 0.0,
        "hard_f1_improved": float(
            summary["hard_class_f1"]["delta_mean"]
        ) > 0.0,
        "bottom3_f1_preserved": float(
            summary["bottom3_class_f1"]["delta_mean"]
        ) >= -0.01,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return len(history) >= epochs


def _metrics_path(root: Path, code: str, seed: int) -> Path:
    return root / "val_reports" / f"{code}_seed{seed}" / "metrics.json"


def _evaluate(
    checkpoint: Path,
    output_dir: Path,
    data_root: Path,
) -> None:
    if output_dir.joinpath("metrics.json").is_file():
        print(f"SKIP evaluasi lengkap: {output_dir.name}", flush=True)
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint belum tersedia: {checkpoint}")
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "bilinear_lmmd.engine.evaluate_checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--domain",
            "source",
            "--split",
            "val",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
    )


def _compare(
    baseline_root: Path,
    candidate_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
) -> dict:
    baseline_paths = [
        _metrics_path(baseline_root, baseline, seed) for seed in seeds
    ]
    candidate_paths = [
        _metrics_path(candidate_root, candidate, seed) for seed in seeds
    ]
    result = aggregate(baseline_paths, candidate_paths)
    baseline_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in baseline_paths
    ]
    candidate_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in candidate_paths
    ]
    result["summary"]["bottom3_class_f1"] = _paired_summary(
        [_bottom_three_f1(report) for report in baseline_reports],
        [_bottom_three_f1(report) for report in candidate_reports],
    )
    destination = (
        candidate_root
        / "val_reports"
        / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== {baseline} vs {candidate} ===")
    for key, label in (
        ("macro_f1", "Macro-F1 "),
        ("hard_class_f1", "Hard-F1  "),
        ("bottom3_class_f1", "Bottom-3 F1"),
        ("worst_class_f1", "Worst-F1 "),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    return result


def _audit_models() -> dict:
    rows = {}
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        rows[code] = {
            "head": cfg["model"]["head"],
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "fusion_dim": int(cfg["model"]["multistage_fusion_dim"]),
            "out_indices": list(cfg["model"]["out_indices"]),
            "adaptive_gate": getattr(model, "gate", None) is not None,
        }
    rows["MSF1"]["parameter_delta_vs_MSF0"] = (
        rows["MSF1"]["parameters"] - rows["MSF0"]["parameters"]
    )
    return rows


def _restore_run_and_report(
    hf_repo: str,
    remote_run_path: str,
    remote_report_path: str,
    run_dir: Path,
    report_dir: Path,
) -> int:
    restored = restore_artifacts(
        hf_repo,
        remote_run_path,
        run_dir,
        overwrite=False,
    )
    restored += restore_artifacts(
        hf_repo,
        remote_report_path,
        report_dir,
        filenames=(
            "metrics.json",
            "confusion_matrix.csv",
            "predictions.csv",
        ),
        overwrite=False,
    )
    return len(restored)


def run_multistage_recalibration_screening(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
    *,
    hf_repo: str | None = None,
    hf_namespace: str = "coffee17-multistage-recalibration-v1",
    hf_sync_every: int = 1,
) -> dict:
    if tuple(seeds) != SCREENING_SEEDS:
        raise ValueError(
            "Screening v1 dikunci pada seed 42. Seed 123/2026 hanya boleh "
            "dibuka setelah MSF1 lolos validation."
        )
    if hf_sync_every <= 0:
        raise ValueError("hf_sync_every harus lebih besar dari nol.")
    if hf_repo:
        try:
            ensure_artifact_repo(hf_repo, private=True)
        except Exception as exc:
            raise RuntimeError(
                "Repo checkpoint Hugging Face tidak dapat ditulis. "
                "Training dibatalkan sebelum dimulai."
            ) from exc

    audits = _audit_models()
    print("=== COFFEE17 MULTISTAGE RECALIBRATION FAIL-FAST ===", flush=True)
    print("Split: validation | Seed: 42 | Test locked", flush=True)
    print(
        f"Checkpoint: {hf_repo + '/' + hf_namespace if hf_repo else output_root}",
        flush=True,
    )
    for code, row in audits.items():
        print(
            f"{code}: head={row['head']} adaptive={row['adaptive_gate']} "
            f"params={row['parameters']:,}",
            flush=True,
        )

    for baseline in BASELINES:
        for seed in seeds:
            run_dir = baseline_root / "outputs" / f"{baseline}_seed{seed}"
            report_dir = (
                baseline_root / "val_reports" / f"{baseline}_seed{seed}"
            )
            if hf_repo:
                restored = _restore_run_and_report(
                    hf_repo,
                    f"outputs/{baseline}_seed{seed}",
                    f"val_reports/{baseline}_seed{seed}",
                    run_dir,
                    report_dir,
                )
                print(
                    f"HF RESTORE {baseline} seed {seed}: {restored} file",
                    flush=True,
                )
            _evaluate(run_dir / "best.pt", report_dir, data_root)

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = output_root / "val_reports" / f"{code}_seed{seed}"
            artifact_run_path = (
                f"{hf_namespace}/outputs/{code}_seed{seed}"
            )
            artifact_report_path = (
                f"{hf_namespace}/val_reports/{code}_seed{seed}"
            )
            if hf_repo:
                restored = _restore_run_and_report(
                    hf_repo,
                    artifact_run_path,
                    artifact_report_path,
                    run_dir,
                    report_dir,
                )
                print(
                    f"HF RESTORE {code} seed {seed}: {restored} file",
                    flush=True,
                )
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                print(f"START / RESUME: {code} seed {seed}", flush=True)
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.engine.train",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(run_dir),
                        "--resume",
                        *(
                            [
                                "--artifact-repo",
                                hf_repo,
                                "--artifact-path",
                                artifact_run_path,
                                "--artifact-sync-every",
                                str(hf_sync_every),
                                "--artifact-required",
                            ]
                            if hf_repo
                            else []
                        ),
                    ]
                )
            _evaluate(run_dir / "best.pt", report_dir, data_root)
            if hf_repo:
                report_files = [
                    path.name for path in report_dir.iterdir() if path.is_file()
                ]
                sync_artifacts(
                    hf_repo,
                    artifact_report_path,
                    report_dir,
                    filenames=report_files,
                    commit_message=f"Evaluate {code} seed {seed} on Coffee17 val",
                )

    comparisons = {}
    for candidate in MODEL_CONFIGS:
        for baseline in BASELINES:
            key = f"{baseline}_vs_{candidate}"
            comparisons[key] = _compare(
                baseline_root,
                output_root,
                baseline,
                candidate,
                seeds,
            )["summary"]
    comparisons["MSF0_vs_MSF1"] = _compare(
        output_root,
        output_root,
        "MSF0",
        "MSF1",
        seeds,
    )["summary"]
    decisions = {
        key: screening_decision(summary)
        for key, summary in comparisons.items()
    }
    final_decision = (
        "PASS"
        if all(
            decisions[key]["decision"] == "PASS"
            for key in REQUIRED_COMPARISONS
        )
        else "FAIL"
    )
    report = {
        "method": "class-supervised multistage recalibration",
        "dataset": "Coffee17 clean grouped fold 1",
        "seeds": seeds,
        "selection_split": "val",
        "test_opened": False,
        "audits": audits,
        "comparisons": comparisons,
        "decisions": decisions,
        "required_comparisons": list(REQUIRED_COMPARISONS),
        "final_decision": final_decision,
    }
    destination = (
        output_root
        / "val_reports"
        / "multistage_recalibration_screening.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if hf_repo:
        sync_artifacts(
            hf_repo,
            f"{hf_namespace}/val_reports",
            destination.parent,
            filenames=(destination.name,),
            commit_message="Coffee17 multistage recalibration screening decision",
        )

    print("\n=== PUTUSAN MULTISTAGE RECALIBRATION ===")
    for key, row in decisions.items():
        print(f"{key:16s}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final_decision)
    print("Test dibuka: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coffee17 validation-only multistage recalibration screening"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    parser.add_argument(
        "--hf-repo",
        help="Repo model Hugging Face private untuk checkpoint lintas akun.",
    )
    parser.add_argument(
        "--hf-namespace",
        default="coffee17-multistage-recalibration-v1",
    )
    parser.add_argument("--hf-sync-every", type=int, default=1)
    args = parser.parse_args()
    run_multistage_recalibration_screening(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
        hf_repo=args.hf_repo,
        hf_namespace=args.hf_namespace,
        hf_sync_every=args.hf_sync_every,
    )


if __name__ == "__main__":
    main()
