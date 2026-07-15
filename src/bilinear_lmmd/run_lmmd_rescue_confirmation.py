from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from .config import load_config


MODEL_CONFIGS = {
    "M1": Path("configs/M1_mobilenetv3_hbp_source.yaml"),
    "M5w01": Path("configs/M5w01_mobilenetv3_hbp_lmmd_w01.yaml"),
}
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _train(model: str, seed: int, data_root: Path, output_root: Path) -> Path:
    config_path = MODEL_CONFIGS[model]
    epochs = int(load_config(config_path)["training"]["epochs"])
    run_dir = output_root / "outputs" / f"{model}_seed{seed}"
    if _training_complete(run_dir, epochs):
        print(f"SKIP training lengkap: {model} seed {seed}", flush=True)
    else:
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
                "--resume",
            ]
        )
    return run_dir / "best.pt"


def _evaluate(
    model: str,
    seed: int,
    checkpoint: Path,
    domain: str,
    data_root: Path,
    output_root: Path,
) -> Path:
    report_dir = output_root / "reports" / f"{model}_seed{seed}" / domain
    metrics_path = report_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"SKIP report lengkap: {model} seed {seed} | {domain}", flush=True)
    else:
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.evaluate_checkpoint",
                "--checkpoint",
                str(checkpoint),
                "--domain",
                domain,
                "--split",
                "test",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(report_dir),
            ]
        )
    return metrics_path


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def summarize_confirmation(output_root: Path, seeds: list[int]) -> dict:
    reports: dict[int, dict[str, dict[str, dict]]] = {}
    for seed in seeds:
        reports[seed] = {}
        for model in MODEL_CONFIGS:
            reports[seed][model] = {}
            for domain in ("source", "target"):
                path = (
                    output_root
                    / "reports"
                    / f"{model}_seed{seed}"
                    / domain
                    / "metrics.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Report konfirmasi belum lengkap: {path}")
                reports[seed][model][domain] = json.loads(
                    path.read_text(encoding="utf-8")
                )

    per_seed: dict[str, dict] = {}
    for seed in seeds:
        baseline = reports[seed]["M1"]
        rescue = reports[seed]["M5w01"]
        source_delta = (
            rescue["source"]["macro_f1"] - baseline["source"]["macro_f1"]
        )
        target_delta = {
            metric: rescue["target"][metric] - baseline["target"][metric]
            for metric in METRICS
        }
        criteria = {
            "source_retention": source_delta >= -0.05,
            "target_macro_improved": target_delta["macro_f1"] > 0,
            "worst_above_zero": rescue["target"]["worst_class_f1"] > 0,
        }
        per_seed[str(seed)] = {
            "M1": {
                "source": {metric: baseline["source"][metric] for metric in METRICS},
                "target": {metric: baseline["target"][metric] for metric in METRICS},
            },
            "M5w01": {
                "source": {metric: rescue["source"][metric] for metric in METRICS},
                "target": {metric: rescue["target"][metric] for metric in METRICS},
            },
            "source_macro_delta": source_delta,
            "target_delta": target_delta,
            "criteria": criteria,
            "pass": all(criteria.values()),
        }

    aggregate = {
        model: {
            domain: {
                metric: _mean_std(
                    [reports[seed][model][domain][metric] for seed in seeds]
                )
                for metric in METRICS
            }
            for domain in ("source", "target")
        }
        for model in MODEL_CONFIGS
    }
    aggregate["delta_M5w01_vs_M1"] = {
        "source_macro_f1": _mean_std(
            [per_seed[str(seed)]["source_macro_delta"] for seed in seeds]
        ),
        "target": {
            metric: _mean_std(
                [per_seed[str(seed)]["target_delta"][metric] for seed in seeds]
            )
            for metric in METRICS
        },
    }
    result = {
        "protocol": "held-out-seed confirmation after seed-123 screening",
        "fixed_config": str(MODEL_CONFIGS["M5w01"]),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "decision": {
            "passed_seeds": sum(per_seed[str(seed)]["pass"] for seed in seeds),
            "total_seeds": len(seeds),
            "pass": all(per_seed[str(seed)]["pass"] for seed in seeds),
        },
    }
    destination = output_root / "reports" / "confirmation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _print_summary(result: dict) -> None:
    print("\n=== KONFIRMASI M1 vs M5w01 ===")
    for seed in result["seeds"]:
        row = result["per_seed"][str(seed)]
        print(f"\nSEED {seed} | {'PASS' if row['pass'] else 'FAIL'}")
        for model in ("M1", "M5w01"):
            source = row[model]["source"]
            target = row[model]["target"]
            print(
                f"{model:<6} TargetMacro={target['macro_f1']:.2%} "
                f"Hard={target['hard_class_f1']:.2%} "
                f"Worst={target['worst_class_f1']:.2%} "
                f"SourceMacro={source['macro_f1']:.2%}"
            )
        delta = row["target_delta"]
        print(
            f"Delta  Macro={delta['macro_f1']:+.2%} "
            f"Hard={delta['hard_class_f1']:+.2%} "
            f"Worst={delta['worst_class_f1']:+.2%} "
            f"Source={row['source_macro_delta']:+.2%}"
        )
        print(
            "Kriteria: "
            + ", ".join(
                f"{name}={'PASS' if passed else 'FAIL'}"
                for name, passed in row["criteria"].items()
            )
        )

    delta = result["aggregate"]["delta_M5w01_vs_M1"]
    print("\n=== AGREGAT SEED KONFIRMASI ===")
    print(
        f"Macro delta : {delta['target']['macro_f1']['mean']:+.2%} "
        f"± {delta['target']['macro_f1']['std']:.2%}"
    )
    print(
        f"Hard delta  : {delta['target']['hard_class_f1']['mean']:+.2%} "
        f"± {delta['target']['hard_class_f1']['std']:.2%}"
    )
    print(
        f"Worst delta : {delta['target']['worst_class_f1']['mean']:+.2%} "
        f"± {delta['target']['worst_class_f1']['std']:.2%}"
    )
    print(
        f"Source delta: {delta['source_macro_f1']['mean']:+.2%} "
        f"± {delta['source_macro_f1']['std']:.2%}"
    )
    decision = result["decision"]
    print(
        f"KEPUTUSAN   : {'PASS' if decision['pass'] else 'FAIL'} "
        f"({decision['passed_seeds']}/{decision['total_seeds']} seed)"
    )


def run_confirmation(data_root: Path, output_root: Path, seeds: list[int]) -> dict:
    for seed in seeds:
        print(f"\n========== SEED {seed} ==========", flush=True)
        for model in MODEL_CONFIGS:
            checkpoint = _train(model, seed, data_root, output_root)
            for domain in ("source", "target"):
                _evaluate(
                    model,
                    seed,
                    checkpoint,
                    domain,
                    data_root,
                    output_root,
                )
    result = summarize_confirmation(output_root, seeds)
    _print_summary(result)
    print(f"\nSAVED: {output_root / 'reports' / 'confirmation.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konfirmasi held-out seeds M1 vs low-weight HBP-LMMD"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        parser.error("Minimal satu seed konfirmasi diperlukan.")
    run_confirmation(args.data_root, args.output_root, seeds)


if __name__ == "__main__":
    main()
