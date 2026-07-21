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
    "COV0": Path("configs/covariance/COV0_efficientnetv2_gap_source.yaml"),
    "COV1": Path("configs/covariance/COV1_efficientnetv2_hbp_source.yaml"),
    "COV2": Path("configs/covariance/COV2_efficientnetv2_mpncov_source.yaml"),
}


def screening_decision(summary: dict) -> dict:
    """Frozen fail-fast gate for COV2 relative to COV0 on validation."""
    macro_delta = float(summary["macro_f1"]["delta_mean"])
    hard_delta = float(summary["hard_class_f1"]["delta_mean"])
    worst_delta = float(summary["worst_class_f1"]["delta_mean"])
    criteria = {
        "macro_f1_improved": macro_delta > 0.0,
        "hard_f1_improved": hard_delta > 0.0,
        "worst_f1_preserved": worst_delta >= -0.01,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "deltas": {
            "macro_f1": macro_delta,
            "hard_class_f1": hard_delta,
            "worst_class_f1": worst_delta,
        },
    }


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    if not history_path.is_file() or not (run_dir / "best.pt").is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _parameter_count(config_path: Path) -> tuple[int, int]:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = int(model.pool.output_dim)
    return total, embedding


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _report_paths(
    output_root: Path, code: str, seeds: list[int], split: str
) -> list[Path]:
    return [
        _report_root(output_root, split) / f"{code}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]


def _compare(
    output_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
    split: str,
) -> dict:
    result = aggregate(
        _report_paths(output_root, baseline, seeds, split),
        _report_paths(output_root, candidate, seeds, split),
    )
    destination = (
        _report_root(output_root, split)
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
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    print("SAVED:", destination)
    return result


def run_covariance_pooling_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    evaluation_split: str = "val",
) -> dict:
    print("=== COMPACT MPN-COV FAIL-FAST ===")
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        parameters, embedding = _parameter_count(config_path)
        print(
            f"{code}: backbone={cfg['model']['backbone']} "
            f"head={cfg['model']['head']} embedding={embedding:,} "
            f"params={parameters:,}",
            flush=True,
        )

    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        epochs = int(cfg["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = _report_root(output_root, evaluation_split) / f"{code}_seed{seed}"
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
                    ]
                )
            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {code} seed {seed}", flush=True)
            else:
                print(f"EVALUATE: {code} seed {seed}", flush=True)
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.engine.evaluate_checkpoint",
                        "--checkpoint",
                        str(run_dir / "best.pt"),
                        "--domain",
                        "source",
                        "--split",
                        evaluation_split,
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    _compare(output_root, "COV0", "COV1", seeds, evaluation_split)
    cov_result = _compare(output_root, "COV0", "COV2", seeds, evaluation_split)
    decision = screening_decision(cov_result["summary"])
    decision.update(
        {
            "seeds": seeds,
            "evaluation_split": evaluation_split,
            "baseline": "COV0",
            "candidate": "COV2",
        }
    )
    decision_path = (
        _report_root(output_root, evaluation_split)
        / f"covariance_failfast_{evaluation_split}.json"
    )
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN PRE-REGISTERED COV2 ===")
    for name, passed in decision["criteria"].items():
        print(f"{name:22s}: {'PASS' if passed else 'FAIL'}")
    print("KEPUTUSAN:", decision["decision"])
    if decision["decision"] == "FAIL":
        print("STOP: hentikan pencarian refinement arsitektur ini.")
    else:
        print("NEXT: konfirmasi validation seed 123 dan 2026 sebelum membuka test.")
    print("SAVED:", decision_path)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast EfficientNetV2 GAP/HBP/compact iSQRT-COV"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"), default="val"
    )
    args = parser.parse_args()
    run_covariance_pooling_screening(
        args.data_root,
        args.output_root,
        args.seeds,
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
