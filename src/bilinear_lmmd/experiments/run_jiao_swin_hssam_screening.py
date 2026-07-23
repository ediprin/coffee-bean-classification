from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.sni_ontology import validate_sni_classes
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "SJ0": Path("configs/sni/jiao/SJ0_swin_tiny_gap_ce.yaml"),
    "SJH": Path("configs/sni/jiao/SJH_swin_tiny_hsfpn_ce.yaml"),
    "SJS": Path("configs/sni/jiao/SJS_swin_tiny_sam_ce.yaml"),
    "SJL": Path("configs/sni/jiao/SJL_swin_tiny_gap_fusion_loss.yaml"),
    "SJHL": Path("configs/sni/jiao/SJHL_swin_tiny_hsfpn_fusion_loss.yaml"),
    "SJSL": Path("configs/sni/jiao/SJSL_swin_tiny_sam_fusion_loss.yaml"),
    "SJHS": Path("configs/sni/jiao/SJHS_swin_tiny_hssam_ce.yaml"),
    "SJFULL": Path("configs/sni/jiao/SJFULL_swin_hssam_fusion_loss.yaml"),
}

# Fail-fast first: only baseline and full proposal. The remaining six models
# reproduce Table 7's factorial only after the complete method earns it.
STAGE_MODELS = {
    "screen": ("SJ0", "SJFULL"),
    "ablation": ("SJH", "SJS", "SJL", "SJHL", "SJSL", "SJHS"),
    "all": tuple(MODEL_CONFIGS),
}
FACTORIAL_COMPARISONS = (
    ("SJ0", "SJH"),
    ("SJ0", "SJS"),
    ("SJ0", "SJL"),
    ("SJ0", "SJHL"),
    ("SJ0", "SJSL"),
    ("SJ0", "SJHS"),
    ("SJ0", "SJFULL"),
)


def screening_decision(summary: dict) -> dict:
    criteria = {
        "macro_f1_improved": float(summary["macro_f1"]["delta_mean"]) > 0.0,
        "hard_f1_improved": float(summary["hard_class_f1"]["delta_mean"]) > 0.0,
        "worst_f1_preserved": float(summary["worst_class_f1"]["delta_mean"])
        >= -0.01,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    checkpoint = run_dir / "best.pt"
    if not history.is_file() or not checkpoint.is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def _validate_dataset(data_root: Path) -> None:
    for split in ("train", "val"):
        split_root = data_root / "source" / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split SNI belum tersedia: {split_root}")
        classes = sorted(
            path.name for path in split_root.iterdir() if path.is_dir()
        )
        validate_sni_classes(classes)


def _audit_models() -> dict:
    audits = {}
    for code, path in MODEL_CONFIGS.items():
        cfg = load_config(path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        audits[code] = {
            "head": cfg["model"]["head"],
            "loss": cfg["training"]["classification_loss"],
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "hsfpn": cfg["model"]["head"] in {"swin_hsfpn", "swin_hssam"},
            "sam": cfg["model"]["head"] in {"swin_sam", "swin_hssam"},
            "fusion_loss": cfg["training"]["classification_loss"]
            == "fusion_ce_focal",
        }
    return audits


def _metrics_path(output_root: Path, code: str, seed: int) -> Path:
    return output_root / "val_reports" / f"{code}_seed{seed}" / "metrics.json"


def _evaluate(
    checkpoint: Path, output_dir: Path, data_root: Path
) -> None:
    if output_dir.joinpath("metrics.json").is_file():
        print(f"SKIP evaluasi lengkap: {output_dir.name}", flush=True)
        return
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
    output_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
) -> dict:
    result = aggregate(
        [_metrics_path(output_root, baseline, seed) for seed in seeds],
        [_metrics_path(output_root, candidate, seed) for seed in seeds],
    )
    destination = (
        output_root
        / "val_reports"
        / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate} ===")
    for metric, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][metric]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    return result


def run_jiao_swin_hssam_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    stage: str = "screen",
    hf_repo: str | None = None,
    hf_namespace: str = "sni-jiao-hssam-v1",
    hf_sync_every: int = 1,
) -> dict:
    if stage not in STAGE_MODELS:
        raise ValueError(f"Stage harus salah satu dari {sorted(STAGE_MODELS)}.")
    if hf_sync_every <= 0:
        raise ValueError("hf_sync_every harus lebih besar dari nol.")
    _validate_dataset(data_root)
    if hf_repo:
        try:
            ensure_artifact_repo(hf_repo, private=True)
        except Exception as exc:
            raise RuntimeError(
                "Repo checkpoint Hugging Face tidak dapat ditulis. "
                "Training dibatalkan sebelum dimulai."
            ) from exc
    audits = _audit_models()
    print("=== JIAO SWIN-HSSAM CONTROLLED REPRODUCTION ===", flush=True)
    print("Dataset: SNI 21-class instance crops", flush=True)
    print("Selection: validation only | test locked", flush=True)
    print(
        "Adaptation note: paper architecture/loss, established SNI training "
        "protocol; not a numerical reproduction of the proprietary dataset.",
        flush=True,
    )
    print(
        f"Checkpoint: {hf_repo + '/' + hf_namespace if hf_repo else output_root} "
        f"| setiap {hf_sync_every} epoch",
        flush=True,
    )
    for code, row in audits.items():
        print(
            f"{code}: HS-FPN={row['hsfpn']} SAM={row['sam']} "
            f"FusionLoss={row['fusion_loss']} params={row['parameters']:,}",
            flush=True,
        )

    for code in STAGE_MODELS[stage]:
        config_path = MODEL_CONFIGS[code]
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            artifact_run_path = (
                f"{hf_namespace}/outputs/{code}_seed{seed}"
            )
            report_dir = output_root / "val_reports" / f"{code}_seed{seed}"
            artifact_report_path = (
                f"{hf_namespace}/val_reports/{code}_seed{seed}"
            )
            if hf_repo:
                restored = restore_artifacts(
                    hf_repo,
                    artifact_run_path,
                    run_dir,
                    overwrite=False,
                )
                restored += restore_artifacts(
                    hf_repo,
                    artifact_report_path,
                    report_dir,
                    filenames=(
                        "metrics.json",
                        "confusion_matrix.csv",
                        "predictions.csv",
                    ),
                    overwrite=False,
                )
                print(
                    f"HF RESTORE {code} seed {seed}: {len(restored)} file",
                    flush=True,
                )
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                print(f"START training: {code} seed {seed}", flush=True)
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
            _evaluate(
                run_dir / "best.pt",
                report_dir,
                data_root,
            )
            if hf_repo:
                report_files = [
                    path.name for path in report_dir.iterdir() if path.is_file()
                ]
                sync_artifacts(
                    hf_repo,
                    artifact_report_path,
                    report_dir,
                    filenames=report_files,
                    commit_message=(
                        f"Evaluation {code} seed {seed} on validation"
                    ),
                )

    comparisons = (
        (("SJ0", "SJFULL"),)
        if stage == "screen"
        else FACTORIAL_COMPARISONS
    )
    summaries = {}
    decisions = {}
    for baseline, candidate in comparisons:
        missing = [
            path
            for code in (baseline, candidate)
            for seed in seeds
            if not (path := _metrics_path(output_root, code, seed)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Ablation membutuhkan hasil screen pada seed yang sama: "
                f"{missing}"
            )
        key = f"{baseline}_vs_{candidate}"
        summary = _compare(output_root, baseline, candidate, seeds)["summary"]
        summaries[key] = summary
        decisions[key] = screening_decision(summary)

    final_key = "SJ0_vs_SJFULL"
    final_decision = decisions[final_key]["decision"]
    report = {
        "method": "Jiao et al. Swin-HSSAM controlled classification reproduction",
        "stage": stage,
        "seeds": seeds,
        "selection_split": "val",
        "test_opened": False,
        "paper_components": ["Swin-T", "HS-FPN", "SAM", "Fusion Loss"],
        "official_source_corrections": [
            "all three selected Swin stages remain in the top-down HS-FPN path",
            "SAM performs the element-wise multiplication shown in Fig. 8",
            "multiclass focal loss is computed without redundant one-hot broadcast",
        ],
        "audits": audits,
        "comparisons": summaries,
        "decisions": decisions,
        "final_decision": final_decision,
    }
    destination = (
        output_root / "val_reports" / f"jiao_swin_hssam_{stage}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if hf_repo:
        sync_artifacts(
            hf_repo,
            f"{hf_namespace}/val_reports",
            destination.parent,
            filenames=(destination.name,),
            commit_message=f"Jiao Swin-HSSAM {stage} decision",
        )
    print("\n=== PUTUSAN JIAO SWIN-HSSAM ===")
    for name, row in decisions.items():
        print(f"{name}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final_decision)
    print("TEST TETAP TERKUNCI: True")
    if stage == "screen" and final_decision == "FAIL":
        print("STOP: jangan jalankan factorial ablation atau seed tambahan.")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only Jiao Swin-HSSAM controlled reproduction"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_MODELS),
        default="screen",
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    parser.add_argument(
        "--hf-repo",
        help="Repo model Hugging Face private untuk checkpoint lintas akun.",
    )
    parser.add_argument(
        "--hf-namespace",
        default="sni-jiao-hssam-v1",
        help="Folder eksperimen di dalam repo Hugging Face.",
    )
    parser.add_argument(
        "--hf-sync-every",
        type=int,
        default=1,
        help="Sinkronkan checkpoint setiap N epoch (default: 1).",
    )
    args = parser.parse_args()
    run_jiao_swin_hssam_screening(
        args.data_root,
        args.output_root,
        args.seeds,
        stage=args.stage,
        hf_repo=args.hf_repo,
        hf_namespace=args.hf_namespace,
        hf_sync_every=args.hf_sync_every,
    )


if __name__ == "__main__":
    main()
