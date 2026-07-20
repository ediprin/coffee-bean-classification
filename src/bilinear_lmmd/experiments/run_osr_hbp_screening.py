from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from bilinear_lmmd.data.preparation.prepare_osr_splits import prepare_osr_splits
from bilinear_lmmd.experiments.run_osr_baselines import (
    _run,
    _write_split_config,
    evaluate_split,
)


TIERS = ("near", "medium")
PRIMARY_SCORE = "msp"
MIN_AUROC_GAIN = 0.02
MIN_OSCR_GAIN = 0.0
MAX_KNOWN_F1_DROP = 0.01


def compare_hbp_to_gap(baseline: dict, candidate: dict) -> dict:
    """Apply the frozen fail-fast gate to matching GAP and HBP reports."""

    comparisons = {}
    for tier in TIERS:
        baseline_metrics = baseline["splits"][tier]["primary_balanced"][
            PRIMARY_SCORE
        ]
        candidate_metrics = candidate["splits"][tier]["primary_balanced"][
            PRIMARY_SCORE
        ]
        deltas = {
            metric: float(candidate_metrics[metric] - baseline_metrics[metric])
            for metric in ("known_macro_f1", "oscr", "auroc", "fpr95")
        }
        criteria = {
            "auroc_gain_at_least_2pp": (
                deltas["auroc"] >= MIN_AUROC_GAIN - 1e-12
            ),
            "oscr_improved": deltas["oscr"] > MIN_OSCR_GAIN,
            "known_macro_f1_retained": (
                deltas["known_macro_f1"] >= -MAX_KNOWN_F1_DROP - 1e-12
            ),
        }
        comparisons[tier] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": deltas,
            "criteria": criteria,
            "decision": "PASS" if all(criteria.values()) else "FAIL",
        }
    passed_tiers = [
        tier for tier, row in comparisons.items() if row["decision"] == "PASS"
    ]
    return {
        "primary_score": PRIMARY_SCORE,
        "gate": {
            "minimum_auroc_gain": MIN_AUROC_GAIN,
            "minimum_oscr_gain_strict": MIN_OSCR_GAIN,
            "maximum_known_macro_f1_drop": MAX_KNOWN_F1_DROP,
        },
        "tiers": comparisons,
        "passed_tiers": passed_tiers,
        "decision": "PASS" if passed_tiers else "FAIL",
        "next_step": (
            "confirm passing tiers on seeds 42 and 2026"
            if passed_tiers
            else "stop EfficientNetV2-HBP for OSR"
        ),
    }


def run_osr_hbp_screening(
    data_root: Path,
    prepared_root: Path,
    output_root: Path,
    protocol_path: Path,
    config_path: Path,
    seed: int,
    resume: bool,
    artifact_repo: str | None = None,
    artifact_sync_every: int = 5,
) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    prepare_osr_splits(data_root, prepared_root, protocol_path)
    baseline_path = output_root / "reports" / f"osr_v1_seed{seed}_summary.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline GAP seed {seed} belum ada: {baseline_path}. "
            "Jalankan notebook baseline terlebih dahulu."
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    reports = {}
    for index, tier in enumerate(TIERS, start=1):
        print(
            f"\n=== HBP FAIL-FAST {tier.upper()} ({index}/{len(TIERS)}) "
            f"| SEED {seed} ===",
            flush=True,
        )
        tier_root = prepared_root / tier
        run_root = output_root / "outputs" / f"OSR1_{tier}_seed{seed}"
        checkpoint = run_root / "best.pt"
        resolved_config = (
            output_root / "resolved_configs" / f"OSR1_{tier}_seed{seed}.yaml"
        )
        _write_split_config(config_path, tier_root, run_root, resolved_config)
        if not checkpoint.is_file():
            command = [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.engine.train",
                "--config",
                str(resolved_config),
                "--seed",
                str(seed),
                "--output-dir",
                str(run_root),
            ]
            if resume:
                command.append("--resume")
            if artifact_repo:
                command.extend(
                    [
                        "--artifact-repo",
                        artifact_repo,
                        "--artifact-path",
                        f"osr-v1/OSR1_{tier}_seed{seed}",
                        "--artifact-sync-every",
                        str(artifact_sync_every),
                    ]
                )
            _run(command)
        else:
            print(f"SKIP training: checkpoint ditemukan: {checkpoint}", flush=True)
        reports[tier] = evaluate_split(
            checkpoint,
            tier_root,
            protocol,
            output_root / "reports" / f"OSR1_{tier}_seed{seed}",
        )

    candidate = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "model": "EfficientNetV2-B0 + HBP + CE",
        "splits": reports,
    }
    candidate_path = output_root / "reports" / f"osr1_seed{seed}_summary.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    result = compare_hbp_to_gap(baseline, candidate)
    result.update(
        {
            "seed": seed,
            "baseline": "EfficientNetV2-B0 + GAP + CE",
            "candidate": "EfficientNetV2-B0 + HBP + CE",
        }
    )
    result_path = output_root / "reports" / f"osr_hbp_failfast_seed{seed}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== KEPUTUSAN HBP OSR FAIL-FAST ===", flush=True)
    for tier, row in result["tiers"].items():
        delta = row["delta"]
        print(
            f"{tier.upper():6s} {row['decision']} | "
            f"AUROC={delta['auroc']:+.2%} OSCR={delta['oscr']:+.2%} "
            f"KnownMacro={delta['known_macro_f1']:+.2%}",
            flush=True,
        )
    print(f"FINAL: {result['decision']} — {result['next_step']}", flush=True)
    print(f"SAVED: {result_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast EfficientNetV2 GAP versus HBP on Coffee17 OSR"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/osr/coffee17_osr_v1.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/osr/OSR1_efficientnetv2_hbp_ce.yaml"),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-repo")
    parser.add_argument("--artifact-sync-every", type=int, default=5)
    args = parser.parse_args()
    run_osr_hbp_screening(
        args.data_root,
        args.prepared_root,
        args.output_root,
        args.protocol,
        args.config,
        args.seed,
        args.resume,
        args.artifact_repo,
        args.artifact_sync_every,
    )


if __name__ == "__main__":
    main()
