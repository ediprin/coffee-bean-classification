from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_synthetic_benchmark import _validate_source_checkpoint


M5W01_CONFIG = Path("configs/coffee17/M5w01_mobilenetv3_hbp_lmmd_w01.yaml")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def source_fingerprint(data_root: Path) -> dict:
    source_root = data_root / "source"
    digest = hashlib.sha256()
    count = 0
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split source tidak ditemukan: {split_root}")
        for path in sorted(split_root.glob("*/*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(source_root).as_posix()
            digest.update(relative.encode("utf-8"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            count += 1
    if count == 0:
        raise ValueError(f"Tidak ada gambar source di {source_root}")
    return {"count": count, "sha256": digest.hexdigest()}


def validate_baseline_for_domain(
    checkpoint_path: Path,
    seed: int,
    domain_root: Path,
    fingerprint_cache: dict[Path, dict],
) -> dict:
    _validate_source_checkpoint("M1", seed, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference_root = Path(checkpoint["config"]["data"]["root"]).resolve()
    candidate_root = domain_root.resolve()
    for root in (reference_root, candidate_root):
        if root not in fingerprint_cache:
            fingerprint_cache[root] = source_fingerprint(root)
    reference = fingerprint_cache[reference_root]
    candidate = fingerprint_cache[candidate_root]
    if candidate != reference:
        raise ValueError(
            "Source domain checkpoint dan cross-shift tidak identik: "
            f"checkpoint={reference_root} {reference}, "
            f"candidate={candidate_root} {candidate}."
        )
    return candidate


def _train_m5w01(
    seed: int,
    domain: str,
    data_root: Path,
    output_root: Path,
) -> Path:
    run_dir = output_root / "outputs" / domain / f"M5w01_seed{seed}"
    epochs = int(load_config(M5W01_CONFIG)["training"]["epochs"])
    if _training_complete(run_dir, epochs):
        print(f"SKIP training lengkap: {domain} | M5w01 seed {seed}", flush=True)
    else:
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.engine.train",
                "--config",
                str(M5W01_CONFIG),
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
    checkpoint: Path,
    model: str,
    seed: int,
    shift: str,
    evaluation_domain: str,
    data_root: Path,
    output_root: Path,
) -> Path:
    report_dir = (
        output_root
        / "reports"
        / shift
        / f"{model}_seed{seed}"
        / evaluation_domain
    )
    metrics_path = report_dir / "metrics.json"
    if metrics_path.is_file():
        print(
            f"SKIP report lengkap: {shift} | {model} seed {seed} | "
            f"{evaluation_domain}",
            flush=True,
        )
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
                evaluation_domain,
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


def summarize_cross_shift(
    output_root: Path,
    domains: list[str],
    seeds: list[int],
) -> dict:
    reports: dict[str, dict[int, dict[str, dict[str, dict]]]] = {}
    for shift in domains:
        reports[shift] = {}
        for seed in seeds:
            reports[shift][seed] = {}
            for model in ("M1", "M5w01"):
                reports[shift][seed][model] = {}
                for evaluation_domain in ("source", "target"):
                    path = (
                        output_root
                        / "reports"
                        / shift
                        / f"{model}_seed{seed}"
                        / evaluation_domain
                        / "metrics.json"
                    )
                    if not path.is_file():
                        raise FileNotFoundError(f"Report cross-shift belum lengkap: {path}")
                    reports[shift][seed][model][evaluation_domain] = json.loads(
                        path.read_text(encoding="utf-8")
                    )

    domain_results: dict[str, dict] = {}
    for shift in domains:
        per_seed: dict[str, dict] = {}
        for seed in seeds:
            baseline = reports[shift][seed]["M1"]
            rescue = reports[shift][seed]["M5w01"]
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
                    evaluation_domain: {
                        metric: baseline[evaluation_domain][metric] for metric in METRICS
                    }
                    for evaluation_domain in ("source", "target")
                },
                "M5w01": {
                    evaluation_domain: {
                        metric: rescue[evaluation_domain][metric] for metric in METRICS
                    }
                    for evaluation_domain in ("source", "target")
                },
                "source_macro_delta": source_delta,
                "target_delta": target_delta,
                "criteria": criteria,
                "pass": all(criteria.values()),
            }

        aggregate_delta = {
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
        domain_results[shift] = {
            "per_seed": per_seed,
            "aggregate_delta": aggregate_delta,
            "decision": {
                "passed_seeds": sum(per_seed[str(seed)]["pass"] for seed in seeds),
                "total_seeds": len(seeds),
                "pass": all(per_seed[str(seed)]["pass"] for seed in seeds),
            },
        }

    result = {
        "protocol": "cross-shift confirmation with fixed LMMD weight 0.1",
        "domains": domains,
        "seeds": seeds,
        "fixed_config": str(M5W01_CONFIG),
        "domain_results": domain_results,
        "decision": {
            "passed_domains": sum(
                domain_results[shift]["decision"]["pass"] for shift in domains
            ),
            "total_domains": len(domains),
            "pass": all(domain_results[shift]["decision"]["pass"] for shift in domains),
        },
    }
    destination = output_root / "reports" / "cross_shift_confirmation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _print_summary(result: dict) -> None:
    print("\n=== KONFIRMASI CROSS-SHIFT M1 vs M5w01 ===")
    for shift in result["domains"]:
        domain_result = result["domain_results"][shift]
        print(f"\nDOMAIN: {shift}")
        for seed in result["seeds"]:
            row = domain_result["per_seed"][str(seed)]
            print(f"SEED {seed} | {'PASS' if row['pass'] else 'FAIL'}")
            for model in ("M1", "M5w01"):
                source = row[model]["source"]
                target = row[model]["target"]
                print(
                    f"  {model:<6} TargetMacro={target['macro_f1']:.2%} "
                    f"Hard={target['hard_class_f1']:.2%} "
                    f"Worst={target['worst_class_f1']:.2%} "
                    f"SourceMacro={source['macro_f1']:.2%}"
                )
            delta = row["target_delta"]
            print(
                f"  Delta  Macro={delta['macro_f1']:+.2%} "
                f"Hard={delta['hard_class_f1']:+.2%} "
                f"Worst={delta['worst_class_f1']:+.2%} "
                f"Source={row['source_macro_delta']:+.2%}"
            )
        aggregate = domain_result["aggregate_delta"]
        print(
            f"AGGREGATE Macro={aggregate['target']['macro_f1']['mean']:+.2%} "
            f"Hard={aggregate['target']['hard_class_f1']['mean']:+.2%} "
            f"Worst={aggregate['target']['worst_class_f1']['mean']:+.2%} "
            f"Source={aggregate['source_macro_f1']['mean']:+.2%}"
        )
        decision = domain_result["decision"]
        print(
            f"DOMAIN DECISION: {'PASS' if decision['pass'] else 'FAIL'} "
            f"({decision['passed_seeds']}/{decision['total_seeds']} seed)"
        )
    decision = result["decision"]
    print(
        f"\nCROSS-SHIFT DECISION: {'PASS' if decision['pass'] else 'FAIL'} "
        f"({decision['passed_domains']}/{decision['total_domains']} domain)"
    )


def run_cross_shift_confirmation(
    data_root: Path,
    baseline_output_root: Path,
    output_root: Path,
    domains: list[str],
    seeds: list[int],
) -> dict:
    fingerprint_cache: dict[Path, dict] = {}
    fingerprints: dict[str, dict] = {}
    for shift in domains:
        domain_root = data_root / shift
        if not domain_root.is_dir():
            raise FileNotFoundError(f"Domain sintetis tidak ditemukan: {domain_root}")
        print(f"\n========== DOMAIN {shift} ==========", flush=True)
        for seed in seeds:
            print(f"\n----- SEED {seed} -----", flush=True)
            m1_checkpoint = (
                baseline_output_root / "outputs" / f"M1_seed{seed}" / "best.pt"
            )
            fingerprints[shift] = validate_baseline_for_domain(
                m1_checkpoint,
                seed,
                domain_root,
                fingerprint_cache,
            )
            print(
                f"REUSE M1 tervalidasi: seed {seed} | "
                f"source sha256={fingerprints[shift]['sha256'][:12]}...",
                flush=True,
            )
            for evaluation_domain in ("source", "target"):
                _evaluate(
                    m1_checkpoint,
                    "M1",
                    seed,
                    shift,
                    evaluation_domain,
                    domain_root,
                    output_root,
                )
            m5_checkpoint = _train_m5w01(
                seed,
                shift,
                domain_root,
                output_root,
            )
            for evaluation_domain in ("source", "target"):
                _evaluate(
                    m5_checkpoint,
                    "M5w01",
                    seed,
                    shift,
                    evaluation_domain,
                    domain_root,
                    output_root,
                )
    result = summarize_cross_shift(output_root, domains, seeds)
    result["source_fingerprints"] = fingerprints
    destination = output_root / "reports" / "cross_shift_confirmation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_summary(result)
    print(f"\nSAVED: {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konfirmasi fixed low-weight HBP-LMMD pada sensor/background"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=["sensor", "background"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    args = parser.parse_args()
    domains = list(dict.fromkeys(args.domains))
    seeds = list(dict.fromkeys(args.seeds))
    if not domains or not seeds:
        parser.error("Minimal satu domain dan satu seed diperlukan.")
    run_cross_shift_confirmation(
        data_root=args.data_root,
        baseline_output_root=args.baseline_output_root,
        output_root=args.output_root,
        domains=domains,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
