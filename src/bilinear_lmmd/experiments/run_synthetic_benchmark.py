from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.preparation.prepare_synthetic_domains import DOMAIN_NAMES, prepare_synthetic_domains


MODEL_CONFIGS = {
    "M0": Path("configs/coffee17/M0_mobilenetv3_gap_source.yaml"),
    "M1": Path("configs/coffee17/M1_mobilenetv3_hbp_source.yaml"),
    "M2": Path("configs/coffee17/M2_mobilenetv3_gap_mmd.yaml"),
    "M3": Path("configs/coffee17/M3_mobilenetv3_gap_lmmd.yaml"),
    "M4": Path("configs/coffee17/M4_mobilenetv3_hbp_dann.yaml"),
    "M5": Path("configs/coffee17/M5_mobilenetv3_hbp_lmmd.yaml"),
}
BASELINE_FOR = {"M2": "M0", "M3": "M0", "M4": "M1", "M5": "M1"}
PAIRWISE_COMPARISONS = (
    ("M0", "M1"),
    ("M0", "M2"),
    ("M0", "M3"),
    ("M2", "M3"),
    ("M1", "M5"),
)
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


def _parse_source_checkpoint_specs(
    specifications: list[str],
) -> dict[tuple[str, int], Path]:
    parsed: dict[tuple[str, int], Path] = {}
    for specification in specifications:
        try:
            model_seed, path_text = specification.split("=", 1)
            model, seed_text = model_seed.split(":", 1)
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError(
                "Format --source-checkpoints harus MODEL:SEED=PATH, "
                f"diterima: {specification!r}"
            ) from error
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Kode model checkpoint tidak dikenal: {model}")
        method = load_config(MODEL_CONFIGS[model])["adaptation"]["method"]
        if method != "source_only":
            raise ValueError(
                f"Checkpoint reuse hanya untuk source-only; {model} memakai {method}."
            )
        key = (model, seed)
        if key in parsed:
            raise ValueError(f"Checkpoint diberikan dua kali: {model} seed {seed}")
        parsed[key] = Path(path_text)
    return parsed


def _validate_source_checkpoint(model: str, seed: int, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint reuse tidak ditemukan: {checkpoint_path}")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(f"Checkpoint tidak memiliki resolved config: {checkpoint_path}")
    observed_seed = int(checkpoint_cfg.get("seed", -1))
    if observed_seed != seed:
        raise ValueError(
            f"Seed checkpoint {checkpoint_path} adalah {observed_seed}, diminta {seed}."
        )
    expected_cfg = load_config(MODEL_CONFIGS[model])
    expected_epochs = int(expected_cfg["training"]["epochs"])
    history_path = checkpoint_path.parent / "history.json"
    if not history_path.is_file():
        raise ValueError(
            f"Checkpoint reuse tidak memiliki history.json: {checkpoint_path.parent}"
        )
    try:
        completed_epochs = len(json.loads(history_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"History checkpoint tidak valid: {history_path}") from error
    if completed_epochs < expected_epochs:
        raise ValueError(
            f"Checkpoint {model} seed {seed} belum lengkap: "
            f"{completed_epochs}/{expected_epochs} epoch. Gunakan --resume."
        )
    if checkpoint_cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError(f"Checkpoint {checkpoint_path} bukan source-only.")
    for key in ("backbone", "head", "classifier", "out_indices"):
        expected = expected_cfg["model"].get(key)
        observed = checkpoint_cfg.get("model", {}).get(key)
        if observed != expected:
            raise ValueError(
                f"Checkpoint {model} tidak cocok pada model.{key}: "
                f"ditemukan {observed!r}, diharapkan {expected!r}."
            )
    expected_size = expected_cfg["data"]["image_size"]
    observed_size = checkpoint_cfg.get("data", {}).get("image_size")
    if observed_size != expected_size:
        raise ValueError(
            f"Checkpoint {model} memakai image_size={observed_size}, "
            f"diharapkan {expected_size}."
        )


def _train(
    model: str,
    seed: int,
    data_root: Path,
    run_dir: Path,
) -> None:
    config_path = MODEL_CONFIGS[model]
    epochs = int(load_config(config_path)["training"]["epochs"])
    if _training_complete(run_dir, epochs):
        print(f"SKIP training lengkap: {model} seed {seed} | {data_root.name}")
        return
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


def _evaluate(
    checkpoint: Path,
    domain: str,
    data_root: Path,
    report_dir: Path,
) -> None:
    if (report_dir / "metrics.json").is_file():
        print(f"SKIP report lengkap: {report_dir}")
        return
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "bilinear_lmmd.engine.evaluate_checkpoint",
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


def _collect_summary(
    output_root: Path,
    domains: list[str],
    models: list[str],
    seeds: list[int],
) -> dict:
    rows: list[dict] = []
    for domain in domains:
        for model in models:
            for seed in seeds:
                report_root = output_root / "reports" / domain / f"{model}_seed{seed}"
                source_path = report_root / "source" / "metrics.json"
                target_path = report_root / "target" / "metrics.json"
                if not source_path.is_file() or not target_path.is_file():
                    continue
                source = json.loads(source_path.read_text(encoding="utf-8"))
                target = json.loads(target_path.read_text(encoding="utf-8"))
                row = {"domain": domain, "model": model, "seed": seed}
                for metric in METRICS:
                    row[f"source_{metric}"] = source[metric]
                    row[f"target_{metric}"] = target[metric]
                rows.append(row)

    aggregates: dict[str, dict] = {}
    for domain in domains:
        aggregates[domain] = {}
        for model in models:
            selected = [
                row for row in rows if row["domain"] == domain and row["model"] == model
            ]
            if not selected:
                continue
            model_summary = {}
            for metric in METRICS:
                values = [row[f"target_{metric}"] for row in selected]
                model_summary[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                }
            model_summary["source"] = {}
            for metric in METRICS:
                values = [row[f"source_{metric}"] for row in selected]
                model_summary["source"][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                }
            baseline_name = BASELINE_FOR.get(model)
            if baseline_name in models:
                baseline_by_seed = {
                    row["seed"]: row
                    for row in rows
                    if row["domain"] == domain and row["model"] == baseline_name
                }
                paired = [
                    (baseline_by_seed[row["seed"]], row)
                    for row in selected
                    if row["seed"] in baseline_by_seed
                ]
                model_summary["versus_baseline"] = {
                    "baseline": baseline_name,
                    "delta": {
                        metric: {
                            "mean": statistics.mean(
                                candidate[f"target_{metric}"]
                                - baseline[f"target_{metric}"]
                                for baseline, candidate in paired
                            ),
                            "values": [
                                candidate[f"target_{metric}"]
                                - baseline[f"target_{metric}"]
                                for baseline, candidate in paired
                            ],
                        }
                        for metric in METRICS
                    },
                }
            aggregates[domain][model] = model_summary

    pairwise: dict[str, dict] = {}
    for domain in domains:
        pairwise[domain] = {}
        for baseline_name, candidate_name in PAIRWISE_COMPARISONS:
            if baseline_name not in models or candidate_name not in models:
                continue
            baseline_by_seed = {
                row["seed"]: row
                for row in rows
                if row["domain"] == domain and row["model"] == baseline_name
            }
            candidate_by_seed = {
                row["seed"]: row
                for row in rows
                if row["domain"] == domain and row["model"] == candidate_name
            }
            common_seeds = sorted(set(baseline_by_seed).intersection(candidate_by_seed))
            if not common_seeds:
                continue
            pairwise[domain][f"{baseline_name}_vs_{candidate_name}"] = {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "seeds": common_seeds,
                "delta": {
                    metric: {
                        "mean": statistics.mean(
                            candidate_by_seed[seed][f"target_{metric}"]
                            - baseline_by_seed[seed][f"target_{metric}"]
                            for seed in common_seeds
                        ),
                        "values": [
                            candidate_by_seed[seed][f"target_{metric}"]
                            - baseline_by_seed[seed][f"target_{metric}"]
                            for seed in common_seeds
                        ],
                    }
                    for metric in METRICS
                },
            }

    summary = {
        "protocol": "controlled_synthetic_domain_shift",
        "claim_scope": "synthetic robustness and UDA sanity-check; not real-world validation",
        "domains": domains,
        "models": models,
        "seeds": seeds,
        "rows": rows,
        "aggregates": aggregates,
        "pairwise": pairwise,
    }
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if rows:
        with (report_root / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _print_summary(summary: dict) -> None:
    print("\n=== HASIL CONTROLLED SYNTHETIC DOMAIN SHIFT ===")
    print("Klaim: robustness sintetis/UDA sanity-check, bukan validasi dunia nyata")
    for domain, models in summary["aggregates"].items():
        print(f"\nDOMAIN: {domain}")
        for model, result in models.items():
            print(
                f"{model:<3} Macro={result['macro_f1']['mean']:.2%} "
                f"Hard={result['hard_class_f1']['mean']:.2%} "
                f"Worst={result['worst_class_f1']['mean']:.2%}"
            )
            comparison = result.get("versus_baseline")
            if comparison:
                delta = comparison["delta"]
                print(
                    f"    vs {comparison['baseline']}: "
                    f"Macro={delta['macro_f1']['mean']:+.2%} "
                    f"Hard={delta['hard_class_f1']['mean']:+.2%} "
                    f"Worst={delta['worst_class_f1']['mean']:+.2%}"
                )
        if summary["pairwise"].get(domain):
            print("  PAIRWISE TARGET DELTA")
            for result in summary["pairwise"][domain].values():
                delta = result["delta"]
                print(
                    f"    {result['candidate']} vs {result['baseline']}: "
                    f"Macro={delta['macro_f1']['mean']:+.2%} "
                    f"Hard={delta['hard_class_f1']['mean']:+.2%} "
                    f"Worst={delta['worst_class_f1']['mean']:+.2%}"
                )


def run_synthetic_benchmark(
    source_root: Path,
    data_root: Path,
    output_root: Path,
    domains: list[str],
    models: list[str],
    seeds: list[int],
    generation_seed: int,
    source_checkpoints: dict[tuple[str, int], Path] | None = None,
) -> dict:
    source_checkpoints = source_checkpoints or {}
    prepare_synthetic_domains(source_root, data_root, domains, generation_seed)
    source_only = {
        model
        for model in models
        if load_config(MODEL_CONFIGS[model])["adaptation"]["method"] == "source_only"
    }
    first_domain_root = data_root / domains[0]
    for model in models:
        for seed in seeds:
            if model in source_only:
                run_dir = output_root / "outputs" / "shared_source" / f"{model}_seed{seed}"
                checkpoint_path = source_checkpoints.get((model, seed))
                if checkpoint_path is not None:
                    _validate_source_checkpoint(model, seed, checkpoint_path)
                    print(
                        f"REUSE checkpoint: {model} seed {seed} | {checkpoint_path}",
                        flush=True,
                    )
                else:
                    _train(model, seed, first_domain_root, run_dir)
                    checkpoint_path = run_dir / "best.pt"
                for domain in domains:
                    domain_root = data_root / domain
                    report_root = output_root / "reports" / domain / f"{model}_seed{seed}"
                    _evaluate(checkpoint_path, "source", domain_root, report_root / "source")
                    _evaluate(checkpoint_path, "target", domain_root, report_root / "target")
            else:
                for domain in domains:
                    domain_root = data_root / domain
                    run_dir = output_root / "outputs" / domain / f"{model}_seed{seed}"
                    report_root = output_root / "reports" / domain / f"{model}_seed{seed}"
                    _train(model, seed, domain_root, run_dir)
                    _evaluate(run_dir / "best.pt", "source", domain_root, report_root / "source")
                    _evaluate(run_dir / "best.pt", "target", domain_root, report_root / "target")
    summary = _collect_summary(output_root, domains, models, seeds)
    _print_summary(summary)
    print(f"\nSAVED: {output_root / 'reports' / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Siapkan dan jalankan benchmark M0-M5 pada domain sintetis terkontrol"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAIN_NAMES, default=["combined"]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=["M0", "M1", "M2", "M3", "M5"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument(
        "--source-checkpoints",
        nargs="*",
        default=[],
        metavar="MODEL:SEED=PATH",
        help=(
            "Pakai checkpoint source-only lama tanpa training ulang, misalnya "
            "M0:123=/path/M0_seed123/best.pt M1:123=/path/M1_seed123/best.pt."
        ),
    )
    args = parser.parse_args()
    run_synthetic_benchmark(
        source_root=args.source_root,
        data_root=args.data_root,
        output_root=args.output_root,
        domains=list(dict.fromkeys(args.domains)),
        models=list(dict.fromkeys(args.models)),
        seeds=list(dict.fromkeys(args.seeds)),
        generation_seed=args.generation_seed,
        source_checkpoints=_parse_source_checkpoint_specs(args.source_checkpoints),
    )


if __name__ == "__main__":
    main()
