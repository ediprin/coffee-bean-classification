from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .aggregate_ablation import aggregate
from .config import load_config
from .models import build_model


MODEL_CONFIGS = {
    "M0a": Path("configs/M0a_mobilenetv3_gap_fixed_cbam_source.yaml"),
    "M0lgf": Path("configs/M0lgf_mobilenetv3_gap_lgf_cbam_source.yaml"),
}


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _parameter_count(config_path: Path) -> int:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    return sum(parameter.numel() for parameter in model.parameters())


def _report_paths(output_root: Path, model: str, seeds: list[int]) -> list[Path]:
    return [
        output_root / "reports" / f"{model}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]


def _print_comparison(
    baseline: str,
    candidate: str,
    output_root: Path,
    seeds: list[int],
) -> None:
    baseline_paths = _report_paths(output_root, baseline, seeds)
    candidate_paths = _report_paths(output_root, candidate, seeds)
    missing = [
        str(path)
        for path in baseline_paths + candidate_paths
        if not path.is_file()
    ]
    if missing:
        print(
            f"SKIP agregasi {baseline} vs {candidate}; report belum ada:\n- "
            + "\n- ".join(missing),
            flush=True,
        )
        return
    result = aggregate(baseline_paths, candidate_paths)
    destination = (
        output_root / "reports" / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate} ===")
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        item = result["summary"][key]
        print(
            f"{label}: {item['baseline_mean']:.2%} -> "
            f"{item['candidate_mean']:.2%} ({item['delta_mean']:+.2%})"
        )
    print(f"SAVED: {destination}")


def run_attention_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    models: list[str] | None = None,
) -> None:
    selected_models = models or ["M0a", "M0lgf"]
    counts = {
        code: _parameter_count(MODEL_CONFIGS[code]) for code in selected_models
    }
    print("=== PARAMETER COUNT ===")
    for code, count in counts.items():
        print(f"{code:5s}: {count:,}")
    if {"M0a", "M0lgf"}.issubset(counts):
        extra = counts["M0lgf"] - counts["M0a"]
        print(f"LGF gate tambahan: {extra:,} parameter")

    for model_code in selected_models:
        config_path = MODEL_CONFIGS[model_code]
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            label = f"{model_code} | seed {seed}"
            run_dir = output_root / "outputs" / f"{model_code}_seed{seed}"
            report_dir = output_root / "reports" / f"{model_code}_seed{seed}"
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {label}", flush=True)
            else:
                print(f"START training: {label}", flush=True)
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.train",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(run_dir),
                    ]
                )
            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {label}", flush=True)
            else:
                print(f"EVALUATE: {label}", flush=True)
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.evaluate_checkpoint",
                        "--checkpoint",
                        str(run_dir / "best.pt"),
                        "--domain",
                        "source",
                        "--split",
                        "test",
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    if {"M0a", "M0lgf"}.issubset(selected_models):
        _print_comparison("M0a", "M0lgf", output_root, seeds)
    for candidate in selected_models:
        _print_comparison("M0", candidate, output_root, seeds)
    print(
        f"\nPASS: screening attention {', '.join(selected_models)} selesai.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screening MobileNetV3 GAP dengan fixed/LGF-CBAM tanpa HBP"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/coffee"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=["M0a", "M0lgf"],
    )
    args = parser.parse_args()
    run_attention_screening(
        args.data_root,
        args.output_root,
        args.seeds,
        models=args.models,
    )


if __name__ == "__main__":
    main()
