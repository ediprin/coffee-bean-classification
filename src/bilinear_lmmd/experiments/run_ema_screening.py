from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.reporting.aggregate_ablation import aggregate
from bilinear_lmmd.core.config import load_config


CONFIG = Path("configs/coffee17/M1e_mobilenetv3_hbp_ema_source.yaml")


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    if not all((run_dir / name).is_file() for name in ("best.pt", "best_raw.pt")):
        return False
    if not history.is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def run_ema_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    evaluation_split: str = "val",
) -> dict:
    epochs = int(load_config(CONFIG)["training"]["epochs"])
    report_root = output_root / f"{evaluation_split}_reports"
    raw_paths = []
    ema_paths = []

    for seed in seeds:
        run_dir = output_root / "outputs" / f"M1e_seed{seed}"
        if _complete(run_dir, epochs):
            print(f"SKIP training lengkap: shared M1/M1e | seed {seed}")
        else:
            print(f"START shared trajectory: M1/M1e | seed {seed}")
            _run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "bilinear_lmmd.engine.train",
                    "--config",
                    str(CONFIG),
                    "--seed",
                    str(seed),
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(run_dir),
                    "--resume",
                ]
            )

        checkpoints = {
            "M1": run_dir / "best_raw.pt",
            "M1e": run_dir / "best.pt",
        }
        for model, checkpoint in checkpoints.items():
            report_dir = report_root / f"{model}_seed{seed}"
            metrics = report_dir / "metrics.json"
            if metrics.is_file():
                print(f"SKIP evaluasi lengkap: {model} | seed {seed}")
            else:
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
                        evaluation_split,
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )
        raw_paths.append(report_root / f"M1_seed{seed}" / "metrics.json")
        ema_paths.append(report_root / f"M1e_seed{seed}" / "metrics.json")

    result = aggregate(raw_paths, ema_paths)
    destination = report_root / "M1_vs_M1e_aggregate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")

    raw_reports = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
    ema_reports = [json.loads(path.read_text(encoding="utf-8")) for path in ema_paths]
    print("\n=== M1 vs M1e: efek EMA pada trajectory HBP yang sama ===")
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        item = result["summary"][key]
        raw_values = [float(report[key]) for report in raw_reports]
        ema_values = [float(report[key]) for report in ema_reports]
        raw_std = statistics.stdev(raw_values) if len(raw_values) > 1 else 0.0
        ema_std = statistics.stdev(ema_values) if len(ema_values) > 1 else 0.0
        print(
            f"{label}: {item['baseline_mean']:.2%}±{raw_std:.2%} -> "
            f"{item['candidate_mean']:.2%}±{ema_std:.2%} "
            f"({item['delta_mean']:+.2%}±{item['delta_std']:.2%}; "
            f"naik {item['improved_seeds']}/{item['total_seeds']} seed)"
        )
    print(f"SAVED: {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screening raw vs EMA dari satu trajectory MobileNetV3-HBP."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--evaluation-split",
        choices=("val",),
        default="val",
        help="Dikunci ke validation; test fold_1 tidak boleh dipakai untuk tuning EMA.",
    )
    args = parser.parse_args()
    run_ema_screening(
        args.data_root,
        args.output_root,
        list(dict.fromkeys(args.seeds)),
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
