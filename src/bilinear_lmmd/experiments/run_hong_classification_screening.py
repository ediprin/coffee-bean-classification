from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.dsconv import DistributionShiftConv2d
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "HCD1": Path("configs/finegrained/HCD1_efficientnetv2_dsconv_gap.yaml"),
    "HCS1": Path(
        "configs/finegrained/HCS1_efficientnetv2_sppf_attention_gap.yaml"
    ),
    "HCDS1": Path(
        "configs/finegrained/HCDS1_efficientnetv2_dsconv_sppf_attention_gap.yaml"
    ),
}
BASELINES = ("BE2G", "BE2H")


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


def _report_root(root: Path) -> Path:
    return root / "val_reports"


def _metrics_path(root: Path, code: str, seed: int) -> Path:
    return _report_root(root) / f"{code}_seed{seed}" / "metrics.json"


def _training_complete(run_dir: Path, epochs: int) -> bool:
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
            "val",
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
) -> dict:
    result = aggregate(
        [_metrics_path(baseline_root, baseline, seed) for seed in seeds],
        [_metrics_path(candidate_root, candidate, seed) for seed in seeds],
    )
    destination = (
        _report_root(candidate_root) / f"{baseline}_vs_{candidate}_aggregate.json"
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


def _model_audit(config_path: Path) -> dict:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    dsconv_modules = [
        module
        for module in model.modules()
        if isinstance(module, DistributionShiftConv2d)
    ]
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding": int(model.pool.output_dim),
        "dsconv_layers": list(getattr(model, "dsconv_replaced_layers", [])),
        "dsconv_theoretical_mb": sum(
            module.theoretical_kernel_bits() for module in dsconv_modules
        )
        / 8.0
        / 1024.0**2,
        "head": str(cfg["model"]["head"]),
    }


def run_hong_classification_screening(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
    models: list[str],
) -> dict:
    if not seeds:
        raise ValueError("Minimal satu seed harus diberikan.")
    unknown = sorted(set(models).difference(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Model Hong-classification tidak dikenal: {unknown}")
    if not models:
        raise ValueError("Minimal satu kandidat harus dipilih.")
    selected = {code: MODEL_CONFIGS[code] for code in models}

    print("=== HONG CLASSIFICATION 2x2 ABLATION ===", flush=True)
    for code, config_path in selected.items():
        audit = _model_audit(config_path)
        print(
            f"{code}: head={audit['head']} params={audit['parameters']:,} "
            f"embedding={audit['embedding']:,} "
            f"DSConv={len(audit['dsconv_layers'])} layer",
            flush=True,
        )
        if audit["dsconv_layers"]:
            print("  DSConv paths:", ", ".join(audit["dsconv_layers"]), flush=True)
            print(
                "  Theoretical DSConv kernel storage: "
                f"{audit['dsconv_theoretical_mb']:.3f} MB "
                "(bukan ukuran checkpoint/latency PyTorch)",
                flush=True,
            )

    for baseline in BASELINES:
        for seed in seeds:
            checkpoint = baseline_root / "outputs" / f"{baseline}_seed{seed}" / "best.pt"
            destination = _report_root(baseline_root) / f"{baseline}_seed{seed}"
            print(f"BASELINE {baseline} seed {seed}", flush=True)
            _evaluate(checkpoint, destination, data_root)

    for code, config_path in selected.items():
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
                _report_root(output_root) / f"{code}_seed{seed}",
                data_root,
            )

    comparisons: dict[str, dict] = {}
    for candidate in selected:
        for baseline in BASELINES:
            key = f"{baseline}_vs_{candidate}"
            comparisons[key] = _compare(
                baseline_root,
                output_root,
                baseline,
                candidate,
                seeds,
            )
    for baseline, candidate in (
        ("HCD1", "HCDS1"),
        ("HCS1", "HCDS1"),
    ):
        if baseline in selected and candidate in selected:
            key = f"{baseline}_vs_{candidate}"
            comparisons[key] = _compare(
                output_root,
                output_root,
                baseline,
                candidate,
                seeds,
            )

    decisions = {
        key: screening_decision(result["summary"])
        for key, result in comparisons.items()
    }
    required = (
        "BE2H_vs_HCDS1",
        "HCD1_vs_HCDS1",
        "HCS1_vs_HCDS1",
    )
    final_ready = all(key in decisions for key in required)
    decisions["HCDS1_final"] = {
        "decision": (
            "PASS"
            if final_ready
            and all(decisions[key]["decision"] == "PASS" for key in required)
            else "FAIL"
        ),
        "requires": list(required),
    }
    report = {
        "seeds": seeds,
        "split": "val",
        "models": list(selected),
        "decisions": decisions,
        "test_opened": False,
        "dsconv_runtime_claim": False,
    }
    destination = _report_root(output_root) / "hong_classification_decision.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN HONG CLASSIFICATION ===")
    for key, row in decisions.items():
        print(f"{key:20s}: {row['decision']}")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only DSConv x SPPF-Attention classification ablation"
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
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_hong_classification_screening(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
        args.models,
    )


if __name__ == "__main__":
    main()
