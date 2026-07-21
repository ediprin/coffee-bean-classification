from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_hong_classification_screening import (
    BASELINES,
    MODEL_CONFIGS,
    _compare,
    _evaluate,
    _model_audit,
    _report_root,
    _run,
    _training_complete,
)


CANDIDATE = "HCD1"
CONFIRMATION_SEEDS = (42, 123, 2026)


def confirmation_decision(summary: dict) -> dict:
    macro = summary["macro_f1"]
    hard = summary["hard_class_f1"]
    worst = summary["worst_class_f1"]
    criteria = {
        "macro_mean_improved": float(macro["delta_mean"]) > 0.0,
        "macro_improved_at_least_2_of_3": int(macro["improved_seeds"]) >= 2,
        "hard_mean_improved": float(hard["delta_mean"]) > 0.0,
        "hard_improved_at_least_2_of_3": int(hard["improved_seeds"]) >= 2,
        "worst_mean_preserved": float(worst["delta_mean"]) >= -0.01,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def run_hong_dsconv_confirmation(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    seeds: list[int],
) -> dict:
    if tuple(seeds) != CONFIRMATION_SEEDS:
        raise ValueError(
            "Konfirmasi dikunci pada seed berurutan 42, 123, 2026; "
            f"diterima {seeds}."
        )

    config_path = MODEL_CONFIGS[CANDIDATE]
    audit = _model_audit(config_path)
    print("=== KONFIRMASI HONG DSCONV-ONLY ===", flush=True)
    print(
        f"{CANDIDATE}: params={audit['parameters']:,} "
        f"DSConv={len(audit['dsconv_layers'])} layer | seeds={seeds}",
        flush=True,
    )

    for baseline in BASELINES:
        for seed in seeds:
            checkpoint = (
                baseline_root / "outputs" / f"{baseline}_seed{seed}" / "best.pt"
            )
            destination = _report_root(baseline_root) / f"{baseline}_seed{seed}"
            _evaluate(checkpoint, destination, data_root)

    epochs = int(load_config(config_path)["training"]["epochs"])
    for seed in seeds:
        run_dir = output_root / "outputs" / f"{CANDIDATE}_seed{seed}"
        if _training_complete(run_dir, epochs):
            print(f"SKIP training lengkap: {CANDIDATE} seed {seed}", flush=True)
        else:
            print(f"START training: {CANDIDATE} seed {seed}", flush=True)
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
        _evaluate(
            run_dir / "best.pt",
            _report_root(output_root) / f"{CANDIDATE}_seed{seed}",
            data_root,
        )

    comparisons = {}
    decisions = {}
    for baseline in BASELINES:
        key = f"{baseline}_vs_{CANDIDATE}"
        result = _compare(
            baseline_root,
            output_root,
            baseline,
            CANDIDATE,
            seeds,
        )
        comparisons[key] = result["summary"]
        decisions[key] = confirmation_decision(result["summary"])

    final = "PASS" if all(
        row["decision"] == "PASS" for row in decisions.values()
    ) else "FAIL"
    report = {
        "candidate": CANDIDATE,
        "seeds": seeds,
        "split": "val",
        "comparisons": comparisons,
        "decisions": decisions,
        "final_decision": final,
        "test_opened": False,
        "dsconv_runtime_claim": False,
    }
    destination = _report_root(output_root) / "hong_dsconv_confirmation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== PUTUSAN KONFIRMASI HCD1 ===")
    for key, row in decisions.items():
        print(f"{key:16s}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final)
    print("Test dibuka: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konfirmasi tiga seed DSConv-only HCD1 pada validation"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(CONFIRMATION_SEEDS)
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_hong_dsconv_confirmation(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
    )


if __name__ == "__main__":
    main()
