from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_factorized_bilinear_conv_screening import (
    BASELINES,
    MODEL_CONFIGS,
    _audit,
    _compare,
    _evaluate,
    _run,
    _training_complete,
)


CONFIRMATION_SEEDS = (42, 123, 2026)
REQUIRED_COMPARISONS = ("BE2G_vs_FB1", "FB0_vs_FB1")


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


def _require_passed_screening(output_root: Path) -> dict:
    path = output_root / "val_reports" / "fbconv_screening_decision.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Report screening seed 42 belum ditemukan. Jalankan notebook "
            "screening terlebih dahulu."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("final_decision") != "PASS" or report.get("seeds") != [42]:
        raise RuntimeError(
            "Konfirmasi ditolak karena screening seed 42 belum berstatus PASS."
        )
    return report


def run_factorized_bilinear_conv_confirmation(
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
    screening = _require_passed_screening(output_root)
    audits = {code: _audit(code, path) for code, path in MODEL_CONFIGS.items()}
    if audits["FB0"]["parameters"] != audits["FB1"]["parameters"]:
        raise AssertionError("FB0 dan FB1 tidak capacity-matched.")

    print("=== KONFIRMASI FACTORIZED BILINEAR CONV ===", flush=True)
    print(
        "Screening seed 42: PASS | melanjutkan hanya seed 123 dan 2026 "
        "yang belum lengkap.",
        flush=True,
    )
    for code, row in audits.items():
        print(
            f"{code}: quadratic={row['quadratic']} rank={row['rank']} "
            f"keep={row['keep_prob']:.2f} params={row['parameters']:,}",
            flush=True,
        )

    for baseline in BASELINES:
        for seed in seeds:
            _evaluate(
                baseline_root / "outputs" / f"{baseline}_seed{seed}" / "best.pt",
                baseline_root / "val_reports" / f"{baseline}_seed{seed}",
                data_root,
            )

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                print(f"START training: {code} seed {seed}", flush=True)
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
                output_root / "val_reports" / f"{code}_seed{seed}",
                data_root,
            )

    comparisons = {}
    for candidate in MODEL_CONFIGS:
        for baseline in BASELINES:
            key = f"{baseline}_vs_{candidate}"
            comparisons[key] = _compare(
                baseline_root, output_root, baseline, candidate, seeds
            )["summary"]
    comparisons["FB0_vs_FB1"] = _compare(
        output_root, output_root, "FB0", "FB1", seeds
    )["summary"]
    decisions = {
        key: confirmation_decision(summary)
        for key, summary in comparisons.items()
    }
    final = (
        "PASS"
        if all(decisions[key]["decision"] == "PASS" for key in REQUIRED_COMPARISONS)
        else "FAIL"
    )
    report = {
        "paper": "Li et al., Factorized Bilinear Models, ICCV 2017",
        "seeds": seeds,
        "split": "val",
        "screening_decision": screening["final_decision"],
        "audits": audits,
        "comparisons": comparisons,
        "decisions": decisions,
        "required_for_final": list(REQUIRED_COMPARISONS),
        "final_decision": final,
        "test_opened": False,
    }
    destination = output_root / "val_reports" / "fbconv_confirmation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN KONFIRMASI FB CONV ===")
    for key, row in decisions.items():
        print(f"{key:16s}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final)
    print("Test dibuka: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-seed validation confirmation for Conv-FBN"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(CONFIRMATION_SEEDS)
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_factorized_bilinear_conv_confirmation(
        args.data_root,
        args.baseline_root,
        args.output_root,
        args.seeds,
    )


if __name__ == "__main__":
    main()
