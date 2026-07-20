from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from bilinear_lmmd.data.preparation.prepare_osr_splits import prepare_osr_splits
from bilinear_lmmd.experiments.run_osr_baselines import evaluate_split


TIERS = ("near", "medium", "far")
PRIMARY_TIERS = ("near", "medium")
MIN_MEAN_AUROC_GAIN = 0.02
MIN_MEAN_OSCR_GAIN = 0.0
MIN_IMPROVED_SEEDS = 2


def aggregate_vim_comparison(rows: list[dict]) -> dict:
    """Aggregate a pre-registered ViM-versus-MSP post-hoc comparison."""

    if not rows:
        raise ValueError("Minimal satu hasil seed ViM diperlukan.")
    seeds = [int(row["seed"]) for row in rows]
    tiers = {}
    for tier in TIERS:
        deltas = {
            metric: np.asarray(
                [
                    row["splits"][tier]["vim"][metric]
                    - row["splits"][tier]["msp"][metric]
                    for row in rows
                ],
                dtype=np.float64,
            )
            for metric in ("known_macro_f1", "oscr", "auroc", "fpr95")
        }
        criteria = {
            "mean_auroc_gain_at_least_2pp": (
                float(deltas["auroc"].mean()) >= MIN_MEAN_AUROC_GAIN - 1e-12
            ),
            "mean_oscr_improved": float(deltas["oscr"].mean()) > MIN_MEAN_OSCR_GAIN,
            "auroc_improved_in_at_least_2_seeds": (
                int(np.sum(deltas["auroc"] > 0.0)) >= MIN_IMPROVED_SEEDS
            ),
        }
        tiers[tier] = {
            "delta": {
                metric: {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "values": {
                        str(seed): float(value)
                        for seed, value in zip(seeds, values, strict=True)
                    },
                }
                for metric, values in deltas.items()
            },
            "criteria": criteria,
            "decision": "PASS" if all(criteria.values()) else "FAIL",
        }
    passed_primary = [
        tier for tier in PRIMARY_TIERS if tiers[tier]["decision"] == "PASS"
    ]
    return {
        "method": "ViM post-hoc (official principal-dimension heuristic)",
        "baseline": "MSP on the same EfficientNetV2-B0 + GAP + CE checkpoint",
        "seeds": seeds,
        "fit_policy": "known_train_only; threshold from known_validation_only",
        "gate": {
            "primary_tiers": list(PRIMARY_TIERS),
            "minimum_mean_auroc_gain": MIN_MEAN_AUROC_GAIN,
            "minimum_mean_oscr_gain_strict": MIN_MEAN_OSCR_GAIN,
            "minimum_seeds_with_positive_auroc_gain": MIN_IMPROVED_SEEDS,
        },
        "tiers": tiers,
        "passed_primary_tiers": passed_primary,
        "decision": "PASS" if passed_primary else "FAIL",
        "next_step": (
            "retain ViM on passing semantic tiers"
            if passed_primary
            else "stop ViM on Coffee17 OSR; freeze GAP+CE+MSP baseline"
        ),
    }


def run_osr_vim_screening(
    data_root: Path,
    prepared_root: Path,
    output_root: Path,
    protocol_path: Path,
    seeds: list[int],
    principal_dimension: int | None = None,
) -> dict:
    """Evaluate ViM from existing GAP checkpoints without further training."""

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    prepare_osr_splits(data_root, prepared_root, protocol_path)
    rows = []
    for seed_index, seed in enumerate(seeds, start=1):
        print(
            f"\n{'#' * 72}\nViM SEED {seed} ({seed_index}/{len(seeds)})\n{'#' * 72}",
            flush=True,
        )
        reports = {}
        for tier_index, tier in enumerate(TIERS, start=1):
            print(
                f"\n=== ViM POST-HOC {tier.upper()} ({tier_index}/3) "
                f"| SEED {seed} ===",
                flush=True,
            )
            checkpoint = (
                output_root / "outputs" / f"OSR0_{tier}_seed{seed}" / "best.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Checkpoint GAP baseline belum ada: {checkpoint}. "
                    "ViM tidak melatih ulang model. Pulihkan artefak OSR0 dahulu."
                )
            report = evaluate_split(
                checkpoint,
                prepared_root / tier,
                protocol,
                output_root / "reports" / f"OSR3_VIM_{tier}_seed{seed}",
                include_vim=True,
                vim_principal_dimension=principal_dimension,
            )
            reports[tier] = report["primary_balanced"]
            msp = reports[tier]["msp"]
            vim = reports[tier]["vim"]
            print(
                f"MSP AUROC={msp['auroc']:.2%} OSCR={msp['oscr']:.2%} | "
                f"ViM AUROC={vim['auroc']:.2%} OSCR={vim['oscr']:.2%} | "
                f"Delta={vim['auroc'] - msp['auroc']:+.2%}/"
                f"{vim['oscr'] - msp['oscr']:+.2%}",
                flush=True,
            )
        row = {"seed": seed, "splits": reports}
        rows.append(row)
        seed_path = output_root / "reports" / f"osr_vim_seed{seed}_summary.json"
        seed_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    result = aggregate_vim_comparison(rows)
    result_path = output_root / "reports" / "osr_vim_aggregate.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== KEPUTUSAN ViM POST-HOC ===", flush=True)
    for tier, row in result["tiers"].items():
        delta = row["delta"]
        print(
            f"{tier.upper():6s} {row['decision']} | "
            f"AUROC={delta['auroc']['mean']:+.2%}±{delta['auroc']['std']:.2%} "
            f"OSCR={delta['oscr']['mean']:+.2%}±{delta['oscr']['std']:.2%} "
            f"naik={sum(value > 0 for value in delta['auroc']['values'].values())}/"
            f"{len(seeds)} seed",
            flush=True,
        )
    print(f"FINAL: {result['decision']} - {result['next_step']}", flush=True)
    print(f"SAVED: {result_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc ViM screening from existing Coffee17 GAP checkpoints"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/osr/coffee17_osr_v1.yaml"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument(
        "--principal-dimension",
        type=int,
        help=(
            "Override terpradaftar; default memakai heuristic resmi ViM "
            "(1000/512/F//2 berdasarkan dimensi feature)."
        ),
    )
    args = parser.parse_args()
    run_osr_vim_screening(
        args.data_root,
        args.prepared_root,
        args.output_root,
        args.protocol,
        args.seeds,
        args.principal_dimension,
    )


if __name__ == "__main__":
    main()
