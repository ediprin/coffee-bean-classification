from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.engine.train_pairwise_contrastive import (
    ContrastiveProjectionHead,
)
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "CP1": Path("configs/finegrained/CP1_efficientnetv2_gap_supcon.yaml"),
    "CP2": Path("configs/finegrained/CP2_efficientnetv2_gap_confusion_pairwise.yaml"),
}
BASELINE_CODES = {"GAP": "BE2G", "HBP": "BE2H"}


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


def _report_root(root: Path, split: str) -> Path:
    return root / ("reports" if split == "test" else f"{split}_reports")


def _complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    if not history.is_file() or not (run_dir / "best.pt").is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def _metrics_path(root: Path, code: str, seed: int, split: str) -> Path:
    return _report_root(root, split) / f"{code}_seed{seed}" / "metrics.json"


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
        [_metrics_path(baseline_root, baseline, seed, split) for seed in seeds],
        [_metrics_path(candidate_root, candidate, seed, split) for seed in seeds],
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
    print("SAVED:", destination)
    return result


def run_pairwise_screening(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
    models: list[str],
    evaluation_split: str = "val",
) -> dict:
    if evaluation_split != "val":
        raise ValueError(
            "Runner fail-fast dikunci ke validation. Test baru boleh dibuka "
            "setelah protokol konfirmasi dibekukan."
        )
    unknown = sorted(set(models).difference(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Model pairwise tidak dikenal: {unknown}")
    selected = {code: MODEL_CONFIGS[code] for code in MODEL_CONFIGS if code in models}
    if not selected:
        raise ValueError("Minimal satu model CP1/CP2 harus dipilih.")

    print("=== CONFUSION-AWARE PAIRWISE SCREENING ===", flush=True)
    for code, config_path in selected.items():
        cfg = load_config(config_path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        projection = ContrastiveProjectionHead(
            int(model.pool.output_dim),
            int(cfg["training"]["contrastive_hidden_dim"]),
            int(cfg["training"]["contrastive_projection_dim"]),
        )
        train_parameters = sum(parameter.numel() for parameter in model.parameters())
        train_parameters += sum(parameter.numel() for parameter in projection.parameters())
        inference_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{code}: mode={cfg['training']['pairwise_mode']} "
            f"train_params={train_parameters:,} inference_params={inference_parameters:,}",
            flush=True,
        )

    for baseline in BASELINE_CODES.values():
        for seed in seeds:
            checkpoint = baseline_root / "outputs" / f"{baseline}_seed{seed}" / "best.pt"
            destination = _report_root(baseline_root, evaluation_split) / f"{baseline}_seed{seed}"
            _evaluate(checkpoint, destination, data_root, evaluation_split)

    for code, config_path in selected.items():
        cfg = load_config(config_path)
        epochs = int(cfg["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            if _complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.engine.train_pairwise_contrastive",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(run_dir),
                        "--resume",
                    ]
                )
            _evaluate(
                run_dir / "best.pt",
                _report_root(output_root, evaluation_split) / f"{code}_seed{seed}",
                data_root,
                evaluation_split,
            )

    comparisons = {}
    for code in selected:
        for baseline in BASELINE_CODES.values():
            key = f"{baseline}_vs_{code}"
            comparisons[key] = _compare(
                baseline_root,
                output_root,
                baseline,
                code,
                seeds,
                evaluation_split,
            )
    if {"CP1", "CP2"}.issubset(selected):
        comparisons["CP1_vs_CP2"] = _compare(
            output_root,
            output_root,
            "CP1",
            "CP2",
            seeds,
            evaluation_split,
        )

    decision: dict = {
        "seeds": seeds,
        "evaluation_split": evaluation_split,
        "models": list(selected),
    }
    for key, result in comparisons.items():
        decision[key] = screening_decision(result["summary"])
    if "CP2" in selected:
        required = [decision.get("BE2H_vs_CP2")]
        if "CP1" in selected:
            required.append(decision.get("CP1_vs_CP2"))
        decision["CP2_final"] = {
            "decision": (
                "PASS"
                if required and all(row and row["decision"] == "PASS" for row in required)
                else "FAIL"
            ),
            "requires": [
                "BE2H_vs_CP2",
                *(["CP1_vs_CP2"] if "CP1" in selected else []),
            ],
        }
    destination = _report_root(output_root, evaluation_split) / "pairwise_decision.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN PAIRWISE ===")
    for key, row in decision.items():
        if isinstance(row, dict) and "decision" in row:
            print(f"{key:14s}: {row['decision']}")
    print("SAVED:", destination)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast CP1 SupCon dan CP2 confusion-aware pairwise"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIGS), default=list(MODEL_CONFIGS)
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_pairwise_screening(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
        args.models,
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
