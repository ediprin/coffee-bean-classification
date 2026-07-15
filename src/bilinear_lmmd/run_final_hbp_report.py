from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .aggregate_ablation import SUMMARY_KEYS, aggregate
from .benchmark import benchmark


DEFAULT_CONFIGS = {
    "M0": Path("configs/M0_mobilenetv3_gap_source.yaml"),
    "M1": Path("configs/M1_mobilenetv3_hbp_source.yaml"),
}


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _load_reports(report_root: Path, model: str, seeds: list[int]) -> list[dict]:
    paths = [report_root / f"{model}_seed{seed}" / "metrics.json" for seed in seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Report test belum lengkap:\n- " + "\n- ".join(missing))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_final_report(
    report_root: Path,
    output_dir: Path,
    seeds: list[int],
    baseline: str = "M0",
    candidate: str = "M1",
    baseline_config: Path | None = None,
    candidate_config: Path | None = None,
    warmup: int = 20,
    iterations: int = 100,
    include_benchmark: bool = True,
) -> dict:
    baseline_reports = _load_reports(report_root, baseline, seeds)
    candidate_reports = _load_reports(report_root, candidate, seeds)
    classes = baseline_reports[0]["classes"]
    if any(report["classes"] != classes for report in baseline_reports + candidate_reports):
        raise ValueError("Urutan kelas report test tidak sama.")

    paired = aggregate(
        [report_root / f"{baseline}_seed{seed}" / "metrics.json" for seed in seeds],
        [report_root / f"{candidate}_seed{seed}" / "metrics.json" for seed in seeds],
    )
    model_reports = {baseline: baseline_reports, candidate: candidate_reports}
    model_summary = {
        model: {
            key: _mean_std([float(report[key]) for report in reports])
            for key in SUMMARY_KEYS
        }
        for model, reports in model_reports.items()
    }

    per_seed_rows: list[dict] = []
    for seed, old, new in zip(seeds, baseline_reports, candidate_reports):
        row: dict[str, int | float] = {"seed": seed}
        for key in SUMMARY_KEYS:
            row[f"{baseline}_{key}"] = old[key]
            row[f"{candidate}_{key}"] = new[key]
            row[f"delta_{key}"] = new[key] - old[key]
        per_seed_rows.append(row)

    per_class_rows = []
    for name in classes:
        item = paired["per_class"][name]
        per_class_rows.append(
            {
                "class": name,
                f"{baseline}_f1_mean": item["baseline_mean"],
                f"{candidate}_f1_mean": item["candidate_mean"],
                "delta_f1_mean": item["delta_mean"],
                "delta_f1_std": item["delta_std"],
                "improved_seeds": item["improved_seeds"],
                "total_seeds": item["total_seeds"],
            }
        )
    per_class_rows.sort(key=lambda row: row["delta_f1_mean"])

    efficiency = {}
    if include_benchmark:
        configs = {
            baseline: baseline_config or DEFAULT_CONFIGS[baseline],
            candidate: candidate_config or DEFAULT_CONFIGS[candidate],
        }
        efficiency = {
            model: benchmark(str(config), warmup, iterations)
            for model, config in configs.items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "split": "test",
            "seeds": seeds,
            "baseline": baseline,
            "candidate": candidate,
        },
        "models": model_summary,
        "paired_comparison": paired,
        "efficiency": efficiency,
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_csv(
        output_dir / "per_seed.csv",
        list(per_seed_rows[0]),
        per_seed_rows,
    )
    _write_csv(
        output_dir / "per_class.csv",
        list(per_class_rows[0]),
        per_class_rows,
    )

    lines = [
        "# Final HBP Test Report",
        "",
        f"Split: test | Seeds: {', '.join(map(str, seeds))}",
        "",
        "| Metric | " + baseline + " mean±SD | " + candidate + " mean±SD | Paired delta mean±SD |",
        "|---|---:|---:|---:|",
    ]
    for key in SUMMARY_KEYS:
        old = model_summary[baseline][key]
        new = model_summary[candidate][key]
        delta = paired["summary"][key]
        lines.append(
            f"| {key} | {old['mean']:.2%} ± {old['std']:.2%} | "
            f"{new['mean']:.2%} ± {new['std']:.2%} | "
            f"{delta['delta_mean']:+.2%} ± {delta['delta_std']:.2%} |"
        )
    lines.extend(["", "## Per-class delta (worst to best)", ""])
    lines.extend(
        f"- {row['class']}: {row['delta_f1_mean']:+.2%} "
        f"({row['improved_seeds']}/{row['total_seeds']} seeds improved)"
        for row in per_class_rows
    )
    if efficiency:
        lines.extend(["", "## Efficiency (batch 1)", ""])
        for model in (baseline, candidate):
            item = efficiency[model]
            lines.append(
                f"- {model}: {item['parameters']:,} parameters, "
                f"{item['model_size_fp32_mb']:.2f} MB FP32, "
                f"{item['latency_ms_batch1']:.3f} ms on {item['device']}"
            )
    (output_dir / "FINAL_HBP_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("=== FINAL M0 vs M1 TEST REPORT ===")
    for key in SUMMARY_KEYS:
        old = model_summary[baseline][key]
        new = model_summary[candidate][key]
        delta = paired["summary"][key]
        print(
            f"{key:22s} {baseline}={old['mean']:.2%}±{old['std']:.2%} "
            f"{candidate}={new['mean']:.2%}±{new['std']:.2%} "
            f"Delta={delta['delta_mean']:+.2%}±{delta['delta_std']:.2%}"
        )
    print(f"SAVED: {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat paket laporan final M0 vs M1 dari report test yang terkunci."
    )
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--baseline", default="M0")
    parser.add_argument("--candidate", default="M1")
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    generate_final_report(
        report_root=args.report_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
        baseline=args.baseline,
        candidate=args.candidate,
        baseline_config=args.baseline_config,
        candidate_config=args.candidate_config,
        warmup=args.warmup,
        iterations=args.iterations,
        include_benchmark=not args.skip_benchmark,
    )


if __name__ == "__main__":
    main()
