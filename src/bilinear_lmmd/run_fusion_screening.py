from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .aggregate_ablation import aggregate
from .config import load_config


MODEL_CONFIGS = {
    "M1c": Path("configs/M1c_mobilenetv3_hbp_mlp_source.yaml"),
    "M1f": Path("configs/M1f_mobilenetv3_gap_hbp_fusion_source.yaml"),
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


def _report_paths(output_root: Path, model: str, seeds: list[int]) -> list[Path]:
    return [
        output_root / "reports" / f"{model}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]


def run_fusion_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
) -> None:
    for model_code, config_path in MODEL_CONFIGS.items():
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

    comparisons = (("M1", "M1c"), ("M1c", "M1f"), ("M1", "M1f"))
    for baseline, candidate in comparisons:
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
            continue
        result = aggregate(baseline_paths, candidate_paths)
        destination = (
            output_root
            / "reports"
            / f"{baseline}_vs_{candidate}_aggregate.json"
        )
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        macro = result["summary"]["macro_f1"]
        hard = result["summary"]["hard_class_f1"]
        worst = result["summary"]["worst_class_f1"]
        print(f"\n=== {baseline} vs {candidate} ===")
        print(
            f"Macro-F1: {macro['baseline_mean']:.2%} -> "
            f"{macro['candidate_mean']:.2%} ({macro['delta_mean']:+.2%})"
        )
        print(
            f"Hard-F1 : {hard['baseline_mean']:.2%} -> "
            f"{hard['candidate_mean']:.2%} ({hard['delta_mean']:+.2%})"
        )
        print(
            f"Worst-F1: {worst['baseline_mean']:.2%} -> "
            f"{worst['candidate_mean']:.2%} ({worst['delta_mean']:+.2%})"
        )
        print(f"SAVED: {destination}")

    print("\nPASS: screening M1c dan M1f selesai.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screening 3-seed untuk kontrol HBP-MLP dan GAP-HBP fusion"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/coffee"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    args = parser.parse_args()
    run_fusion_screening(args.data_root, args.output_root, args.seeds)


if __name__ == "__main__":
    main()
