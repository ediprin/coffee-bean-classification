from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.dcl_finegrained import DCLFineGrainedModel
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "DCL0": Path("configs/finegrained/DCL0_efficientnetv2_dcl.yaml"),
    "DCL1": Path("configs/finegrained/DCL1_efficientnetv2_dcl_supcon.yaml"),
    "DCL2": Path("configs/finegrained/DCL2_efficientnetv2_dcl_confusion.yaml"),
}
BASELINES = ("BE2G", "BE2H")
STAGE_MODELS = {
    "dcl": ("DCL0",),
    "contrastive": ("DCL1", "DCL2"),
    "supcon_confirmation": ("DCL1",),
}


def screening_decision(summary: dict) -> dict:
    criteria = {
        "macro_f1_improved": float(summary["macro_f1"]["delta_mean"]) > 0.0,
        "hard_f1_improved": float(summary["hard_class_f1"]["delta_mean"]) > 0.0,
        "worst_f1_preserved": float(
            summary["worst_class_f1"]["delta_mean"]
        )
        >= -0.01,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _report_root(root: Path, split: str) -> Path:
    return root / ("reports" if split == "test" else f"{split}_reports")


def _metrics_path(root: Path, code: str, seed: int, split: str) -> Path:
    return _report_root(root, split) / f"{code}_seed{seed}" / "metrics.json"


def _complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    if not history.is_file() or not (run_dir / "best.pt").is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def _evaluate(
    checkpoint: Path,
    destination: Path,
    data_root: Path,
    split: str,
) -> None:
    if destination.joinpath("metrics.json").is_file() and destination.joinpath(
        "predictions.csv"
    ).is_file():
        print(f"SKIP evaluasi lengkap: {destination.name}", flush=True)
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
            split,
            "--data-root",
            str(data_root),
            "--output-dir",
            str(destination),
        ]
    )


def _compare(
    baseline_root: Path,
    candidate_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
    split: str,
) -> dict:
    result = aggregate(
        [
            _metrics_path(baseline_root, baseline, seed, split)
            for seed in seeds
        ],
        [
            _metrics_path(candidate_root, candidate, seed, split)
            for seed in seeds
        ],
    )
    destination = (
        _report_root(candidate_root, split)
        / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate} ===")
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} "
            f"Delta={row['delta_mean']:+.2%} ± {row['delta_std']:.2%} "
            f"naik={row['improved_seeds']}/{len(seeds)}"
        )
    return result


def _require_dcl_gate(
    output_root: Path,
    split: str,
    required_seeds: list[int] | None = None,
) -> None:
    path = _report_root(output_root, split) / "dcl_stage_decision.json"
    if not path.is_file():
        raise RuntimeError(
            "Stage contrastive dikunci. Jalankan stage dcl dan nilai "
            "DCL0_vs_BE2G terlebih dahulu."
        )
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("DCL0_final", {}).get("decision") != "PASS":
        raise RuntimeError(
            "DCL0 tidak lolos; contrastive tidak boleh dijalankan."
        )
    if required_seeds is not None and sorted(decision.get("seeds", [])) != sorted(
        required_seeds
    ):
        raise RuntimeError(
            "Seed pada keputusan DCL0 belum sesuai tahap lanjutan: "
            f"{decision.get('seeds', [])} != {required_seeds}."
        )


def run_coffee17_dcl_screening(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
    *,
    stage: str = "dcl",
    evaluation_split: str = "val",
    artifact_repo: str | None = None,
    artifact_namespace: str = "coffee17-dcl-v1",
    artifact_sync_every: int = 1,
    artifact_required: bool = False,
) -> dict:
    if evaluation_split != "val":
        raise ValueError("DCL fail-fast dikunci ke validation.")
    if stage not in STAGE_MODELS:
        raise ValueError(f"Stage tidak dikenal: {stage}")
    if stage in {"contrastive", "supcon_confirmation"}:
        _require_dcl_gate(
            output_root,
            evaluation_split,
            required_seeds=seeds,
        )

    selected = STAGE_MODELS[stage]
    print(
        f"=== COFFEE17 DCL {stage.upper()} SCREENING ===",
        flush=True,
    )
    for code in selected:
        cfg = load_config(MODEL_CONFIGS[code])
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        if not isinstance(model, DCLFineGrainedModel):
            raise TypeError(f"{code} bukan DCLFineGrainedModel.")
        print(
            f"{code}: mode={cfg['training']['dcl_contrastive_mode']} "
            f"inference_params={model.inference_parameter_count():,} "
            f"aux_params={model.auxiliary_parameter_count():,}",
            flush=True,
        )

    for baseline in BASELINES:
        for seed in seeds:
            checkpoint = (
                baseline_root / "outputs" / f"{baseline}_seed{seed}" / "best.pt"
            )
            destination = (
                _report_root(baseline_root, evaluation_split)
                / f"{baseline}_seed{seed}"
            )
            _evaluate(checkpoint, destination, data_root, evaluation_split)

    for code in selected:
        config_path = MODEL_CONFIGS[code]
        cfg = load_config(config_path)
        epochs = int(cfg["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            if _complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "bilinear_lmmd.engine.train_dcl_finegrained",
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(run_dir),
                    "--resume",
                    "--artifact-sync-every",
                    str(artifact_sync_every),
                ]
                if artifact_repo:
                    command.extend(
                        [
                            "--artifact-repo",
                            artifact_repo,
                            "--artifact-path",
                            (
                                f"{artifact_namespace}/outputs/"
                                f"{code}_seed{seed}"
                            ),
                        ]
                    )
                if artifact_required:
                    command.append("--artifact-required")
                _run(command)
            _evaluate(
                run_dir / "best.pt",
                (
                    _report_root(output_root, evaluation_split)
                    / f"{code}_seed{seed}"
                ),
                data_root,
                evaluation_split,
            )

    comparisons: dict[str, dict] = {}
    for code in selected:
        for baseline in BASELINES:
            comparisons[f"{baseline}_vs_{code}"] = _compare(
                baseline_root,
                output_root,
                baseline,
                code,
                seeds,
                evaluation_split,
            )
    if stage in {"contrastive", "supcon_confirmation"}:
        for candidate in selected:
            comparisons[f"DCL0_vs_{candidate}"] = _compare(
                output_root,
                output_root,
                "DCL0",
                candidate,
                seeds,
                evaluation_split,
            )
        if stage == "contrastive":
            comparisons["DCL1_vs_DCL2"] = _compare(
                output_root,
                output_root,
                "DCL1",
                "DCL2",
                seeds,
                evaluation_split,
            )

    decision: dict = {
        "stage": stage,
        "seeds": seeds,
        "evaluation_split": evaluation_split,
        "test_opened": False,
    }
    for key, result in comparisons.items():
        decision[key] = screening_decision(result["summary"])

    if stage == "dcl":
        required = ("BE2G_vs_DCL0",)
        decision["DCL0_final"] = {
            "decision": (
                "PASS"
                if all(decision[key]["decision"] == "PASS" for key in required)
                else "FAIL"
            ),
            "requires": list(required),
        }
        filename = "dcl_stage_decision.json"
    elif stage == "contrastive":
        required = (
            "BE2H_vs_DCL2",
            "DCL0_vs_DCL2",
            "DCL1_vs_DCL2",
        )
        decision["DCL2_final"] = {
            "decision": (
                "PASS"
                if all(decision[key]["decision"] == "PASS" for key in required)
                else "FAIL"
            ),
            "requires": list(required),
        }
        filename = "contrastive_stage_decision.json"
    else:
        required = (
            "BE2H_vs_DCL1",
            "DCL0_vs_DCL1",
        )
        decision["DCL1_final"] = {
            "decision": (
                "PASS"
                if all(decision[key]["decision"] == "PASS" for key in required)
                else "FAIL"
            ),
            "requires": list(required),
        }
        filename = "supcon_confirmation_decision.json"

    destination = _report_root(output_root, evaluation_split) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN ===")
    for key, row in decision.items():
        if isinstance(row, dict) and "decision" in row:
            print(f"{key:18s}: {row['decision']}")
    print("Test dibuka: False")
    print("SAVED:", destination)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only Coffee17 DCL fail-fast runner."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_MODELS),
        default="dcl",
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    parser.add_argument("--artifact-repo")
    parser.add_argument(
        "--artifact-namespace",
        default="coffee17-dcl-v1",
    )
    parser.add_argument("--artifact-sync-every", type=int, default=1)
    parser.add_argument("--artifact-required", action="store_true")
    args = parser.parse_args()
    run_coffee17_dcl_screening(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
        stage=args.stage,
        evaluation_split=args.evaluation_split,
        artifact_repo=args.artifact_repo,
        artifact_namespace=args.artifact_namespace,
        artifact_sync_every=args.artifact_sync_every,
        artifact_required=args.artifact_required,
    )


if __name__ == "__main__":
    main()
