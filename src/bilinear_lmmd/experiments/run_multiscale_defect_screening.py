from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import (
    CapacityResidualGAP,
    MultiScaleDefectGAP,
    build_model,
)
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "MDE0": Path(
        "configs/finegrained/MDE0_efficientnetv2_capacity_control_gap.yaml"
    ),
    "MDE1": Path(
        "configs/finegrained/MDE1_efficientnetv2_multiscale_defect_gap.yaml"
    ),
}
BASELINES = ("BE2G", "BE2H")
SCREENING_SEEDS = (42,)
REQUIRED_COMPARISONS = ("BE2G_vs_MDE1", "MDE0_vs_MDE1")


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


def _metrics_path(root: Path, code: str, seed: int) -> Path:
    return root / "val_reports" / f"{code}_seed{seed}" / "metrics.json"


def _evaluate(checkpoint: Path, output_dir: Path, data_root: Path) -> None:
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
    result = aggregate(
        [_metrics_path(baseline_root, baseline, seed) for seed in seeds],
        [_metrics_path(candidate_root, candidate, seed) for seed in seeds],
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
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    return result


def _audit(code: str, config_path: Path) -> dict:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    expected_type = CapacityResidualGAP if code == "MDE0" else MultiScaleDefectGAP
    if not isinstance(model.pool, expected_type):
        raise TypeError(f"{code} membangun head yang salah: {type(model.pool).__name__}")
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "refinement_parameters": sum(
            parameter.numel() for parameter in model.pool.refinement.parameters()
        ),
        "head": cfg["model"]["head"],
    }


def run_multiscale_defect_screening(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
) -> dict:
    if tuple(seeds) != SCREENING_SEEDS:
        raise ValueError(
            "Screening Chang-Liu v1 dikunci pada seed 42. Tiga seed hanya boleh "
            "dibuka setelah MDE1 mengalahkan GAP dan capacity control pada validation."
        )

    audits = {code: _audit(code, path) for code, path in MODEL_CONFIGS.items()}
    parameter_gap = abs(
        audits["MDE0"]["parameters"] - audits["MDE1"]["parameters"]
    ) / audits["MDE1"]["parameters"]
    if parameter_gap >= 0.001:
        raise AssertionError(
            f"MDE0 dan MDE1 tidak capacity-matched: gap={parameter_gap:.4%}."
        )

    print("=== CHANG-LIU MULTISCALE DEFECT FAIL-FAST ===", flush=True)
    for code, row in audits.items():
        print(
            f"{code}: head={row['head']} params={row['parameters']:,} "
            f"refinement={row['refinement_parameters']:,}",
            flush=True,
        )
    print(f"Capacity gap total: {parameter_gap:.4%}", flush=True)

    for baseline in BASELINES:
        for seed in seeds:
            _evaluate(
                baseline_root / "outputs" / f"{baseline}_seed{seed}" / "best.pt",
                baseline_root / "val_reports" / f"{baseline}_seed{seed}",
                data_root,
            )

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
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
            _evaluate(
                run_dir / "best.pt",
                output_root / "val_reports" / f"{code}_seed{seed}",
                data_root,
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
    comparisons["MDE0_vs_MDE1"] = _compare(
        output_root,
        output_root,
        "MDE0",
        "MDE1",
        seeds,
    )["summary"]

    decisions = {
        key: screening_decision(summary)
        for key, summary in comparisons.items()
    }
    final = (
        "PASS"
        if all(decisions[key]["decision"] == "PASS" for key in REQUIRED_COMPARISONS)
        else "FAIL"
    )
    report = {
        "paper": (
            "Chang and Liu, Multiscale Defect Extraction Neural Network for "
            "Green Coffee Bean Defects Detection, IEEE Access 2024"
        ),
        "adaptation": (
            "parallel standard 3x3/5x5 residual refinement on EfficientNetV2-B0 "
            "deep features before GAP"
        ),
        "seeds": seeds,
        "split": "val",
        "audits": audits,
        "capacity_gap": parameter_gap,
        "comparisons": comparisons,
        "decisions": decisions,
        "required_for_confirmation": list(REQUIRED_COMPARISONS),
        "final_decision": final,
        "test_opened": False,
    }
    destination = output_root / "val_reports" / "mde_screening_decision.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== PUTUSAN CHANG-LIU MDE SCREENING ===")
    for key, row in decisions.items():
        print(f"{key:18s}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final)
    print("Test dibuka: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only Chang-Liu 3x3/5x5 multiscale screening"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_multiscale_defect_screening(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
    )


if __name__ == "__main__":
    main()
