from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.reporting.aggregate_ablation import aggregate
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.data.preparation.prepare_coarse_coffee17 import prepare_coarse_coffee17


MODEL_CONFIGS = {
    "GF0": Path("configs/coffee17/M0_mobilenetv3_gap_source.yaml"),
    "GF0b": Path("configs/coffee17/M0b_mobilenetv3_bilinear_source.yaml"),
    "GF1": Path("configs/coffee17/M1_mobilenetv3_hbp_source.yaml"),
    "GC0": Path("configs/granularity/GC0_mobilenetv3_gap_coarse9_source.yaml"),
    "GC0b": Path("configs/granularity/GC0b_mobilenetv3_bilinear_coarse9_source.yaml"),
    "GC1": Path("configs/granularity/GC1_mobilenetv3_hbp_coarse9_source.yaml"),
}

TASK_ROOT = {
    "GF0": "fine",
    "GF0b": "fine",
    "GF1": "fine",
    "GC0": "coarse",
    "GC0b": "coarse",
    "GC1": "coarse",
}


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    best = run_dir / "best.pt"
    if not history.is_file() or not best.is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _paths(report_root: Path, model: str, seeds: list[int]) -> list[Path]:
    return [report_root / f"{model}_seed{seed}" / "metrics.json" for seed in seeds]


def _comparison(
    report_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
) -> dict | None:
    baseline_paths = _paths(report_root, baseline, seeds)
    candidate_paths = _paths(report_root, candidate, seeds)
    if any(not path.is_file() for path in baseline_paths + candidate_paths):
        return None
    result = aggregate(baseline_paths, candidate_paths)
    destination = report_root / f"{baseline}_vs_{candidate}_aggregate.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate} ===")
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("balanced_accuracy", "Balanced"),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    return result


def _paired_granularity_effect(
    report_root: Path, seeds: list[int]
) -> dict | None:
    required = {
        code: _paths(report_root, code, seeds)
        for code in ("GF0", "GF0b", "GF1", "GC0", "GC0b", "GC1")
    }
    if any(not path.is_file() for paths in required.values() for path in paths):
        return None
    reports = {
        code: [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        for code, paths in required.items()
    }
    result = {"seeds": seeds, "metrics": {}}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "worst_class_f1"):
        rows = []
        for index, seed in enumerate(seeds):
            fine_bilinear_gain = (
                reports["GF0b"][index][metric] - reports["GF0"][index][metric]
            )
            coarse_bilinear_gain = (
                reports["GC0b"][index][metric] - reports["GC0"][index][metric]
            )
            fine_hbp_gain = reports["GF1"][index][metric] - reports["GF0"][index][metric]
            coarse_hbp_gain = reports["GC1"][index][metric] - reports["GC0"][index][metric]
            rows.append(
                {
                    "seed": seed,
                    "fine_bilinear_gain": fine_bilinear_gain,
                    "coarse_bilinear_gain": coarse_bilinear_gain,
                    "bilinear_granularity_effect": fine_bilinear_gain
                    - coarse_bilinear_gain,
                    "fine_hbp_gain": fine_hbp_gain,
                    "coarse_hbp_gain": coarse_hbp_gain,
                    "hbp_granularity_effect": fine_hbp_gain - coarse_hbp_gain,
                    "fine_hierarchical_extra": reports["GF1"][index][metric]
                    - reports["GF0b"][index][metric],
                    "coarse_hierarchical_extra": reports["GC1"][index][metric]
                    - reports["GC0b"][index][metric],
                }
            )
        summary = {}
        for key in rows[0]:
            if key == "seed":
                continue
            values = [row[key] for row in rows]
            summary[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "positive_seeds": sum(value > 0 for value in values),
                "total_seeds": len(values),
                "values": values,
            }
        result["metrics"][metric] = {"per_seed": rows, "summary": summary}
    destination = report_root / "granularity_difference_in_differences.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    row = result["metrics"]["macro_f1"]["summary"]
    print("\n=== UJI HIPOTESIS GRANULARITAS: MACRO-F1 ===")
    print(
        "Gain HBP fine   : "
        f"{row['fine_hbp_gain']['mean']:+.2%} ± {row['fine_hbp_gain']['std']:.2%}"
    )
    print(
        "Gain HBP coarse : "
        f"{row['coarse_hbp_gain']['mean']:+.2%} ± {row['coarse_hbp_gain']['std']:.2%}"
    )
    print(
        "Fine-minus-coarse: "
        f"{row['hbp_granularity_effect']['mean']:+.2%} ± "
        f"{row['hbp_granularity_effect']['std']:.2%} | "
        f"positif={row['hbp_granularity_effect']['positive_seeds']}/"
        f"{row['hbp_granularity_effect']['total_seeds']} seed"
    )
    print(f"SAVED: {destination}")
    return result


def run_granularity_experiment(
    fine_root: Path,
    coarse_root: Path,
    output_root: Path,
    seeds: list[int],
    split: str,
    models: list[str] | None = None,
) -> None:
    prepare_coarse_coffee17(fine_root, coarse_root)
    selected = models or list(MODEL_CONFIGS)
    unknown = sorted(set(selected).difference(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Model granularitas tidak dikenal: {unknown}")
    roots = {"fine": fine_root, "coarse": coarse_root}
    report_root = _report_root(output_root, split)
    report_root.mkdir(parents=True, exist_ok=True)

    print("\n=== RANCANGAN FINE-vs-COARSE ===")
    for code in selected:
        cfg = load_config(MODEL_CONFIGS[code])
        cfg["model"]["pretrained"] = False
        parameters = sum(
            parameter.numel() for parameter in build_model(cfg["model"]).parameters()
        )
        print(
            f"{code}: task={TASK_ROOT[code]} head={cfg['model']['head']} "
            f"classes={cfg['model']['num_classes']} params={parameters:,}"
        )

    for code in selected:
        config_path = MODEL_CONFIGS[code]
        epochs = int(load_config(config_path)["training"]["epochs"])
        data_root = roots[TASK_ROOT[code]]
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = report_root / f"{code}_seed{seed}"
            if _complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
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
                print(f"SKIP report: {code} seed {seed}", flush=True)
            else:
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
                        split,
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    for baseline, candidate in (
        ("GF0", "GF0b"),
        ("GF0", "GF1"),
        ("GF0b", "GF1"),
        ("GC0", "GC0b"),
        ("GC0", "GC1"),
        ("GC0b", "GC1"),
    ):
        _comparison(report_root, baseline, candidate, seeds)
    _paired_granularity_effect(report_root, seeds)
    print("\nSELESAI: hasil fine dan coarse tidak dibandingkan secara absolut; gunakan gain terhadap GAP.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Coffee-17 fine-vs-coarse")
    parser.add_argument("--fine-root", required=True, type=Path)
    parser.add_argument("--coarse-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_CONFIGS))
    args = parser.parse_args()
    run_granularity_experiment(
        args.fine_root,
        args.coarse_root,
        args.output_root,
        args.seeds,
        args.evaluation_split,
        args.models,
    )


if __name__ == "__main__":
    main()
