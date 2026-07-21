from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "E2": Path(
        "configs/finegrained/E2_efficientnetv2_progressive_multigranularity.yaml"
    ),
    "E3": Path(
        "configs/finegrained/E3_efficientnetv2_progressive_consistency.yaml"
    ),
}
BASELINE_CODES = {"E0": "BE2G", "E1": "BE2H"}


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
    if destination.joinpath("metrics.json").is_file():
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
    output_root: Path,
    baseline_code: str,
    candidate_code: str,
    seeds: list[int],
    split: str,
) -> dict:
    result = aggregate(
        [
            _metrics_path(baseline_root, baseline_code, seed, split)
            for seed in seeds
        ],
        [
            _metrics_path(output_root, candidate_code, seed, split)
            for seed in seeds
        ],
    )
    destination = (
        _report_root(output_root, split)
        / f"{baseline_code}_vs_{candidate_code}_aggregate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline_code} vs {candidate_code} ===")
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    print("SAVED:", destination)
    return result


def run_progressive_screening(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
    models: list[str],
    evaluation_split: str = "val",
) -> dict:
    print("=== EFFICIENTNET PROGRESSIVE MULTI-GRANULARITY ===", flush=True)
    unknown = sorted(set(models).difference(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Model progressive tidak dikenal: {unknown}")
    if not models:
        raise ValueError("Minimal satu model progressive harus dipilih.")
    selected_configs = {
        code: MODEL_CONFIGS[code] for code in MODEL_CONFIGS if code in models
    }
    for code, config_path in selected_configs.items():
        cfg = load_config(config_path)
        cfg["model"]["pretrained"] = False
        parameters = sum(
            parameter.numel() for parameter in build_model(cfg["model"]).parameters()
        )
        print(
            f"{code}: head={cfg['model']['head']} params={parameters:,}", flush=True
        )

    for alias, baseline_code in BASELINE_CODES.items():
        for seed in seeds:
            destination = _report_root(baseline_root, evaluation_split) / (
                f"{baseline_code}_seed{seed}"
            )
            checkpoint = baseline_root / "outputs" / f"{baseline_code}_seed{seed}" / "best.pt"
            print(f"BASELINE {alias} menggunakan {baseline_code} seed {seed}", flush=True)
            _evaluate(checkpoint, destination, data_root, evaluation_split)

    for code, config_path in selected_configs.items():
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
                        "bilinear_lmmd.engine.train_progressive",
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
            destination = _report_root(output_root, evaluation_split) / f"{code}_seed{seed}"
            _evaluate(run_dir / "best.pt", destination, data_root, evaluation_split)

    comparisons = {}
    for baseline in ("BE2G", "BE2H"):
        for candidate in selected_configs:
            key = f"{baseline}_vs_{candidate}"
            comparisons[key] = _compare(
                baseline_root,
                output_root,
                baseline,
                candidate,
                seeds,
                evaluation_split,
            )
    decision = {
        "seeds": seeds,
        "models": list(selected_configs),
        "evaluation_split": evaluation_split,
    }
    for candidate in selected_configs:
        decision[f"{candidate}_vs_GAP"] = screening_decision(
            comparisons[f"BE2G_vs_{candidate}"]["summary"]
        )
        decision[f"{candidate}_vs_HBP"] = screening_decision(
            comparisons[f"BE2H_vs_{candidate}"]["summary"]
        )
    if {"E2", "E3"}.issubset(selected_configs):
        comparisons["E2_vs_E3"] = _compare(
            output_root,
            output_root,
            "E2",
            "E3",
            seeds,
            evaluation_split,
        )
        decision["E3_vs_E2"] = screening_decision(
            comparisons["E2_vs_E3"]["summary"]
        )
    destination = _report_root(output_root, evaluation_split) / "progressive_decision.json"
    destination.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN ===")
    for comparison, row in decision.items():
        if isinstance(row, dict) and "decision" in row:
            print(f"{comparison:12s}: {row['decision']}")
    print("SAVED:", destination)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EfficientNetV2 PMG/PMG-V2-inspired controlled screening"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
    )
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    run_progressive_screening(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
        args.models,
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
