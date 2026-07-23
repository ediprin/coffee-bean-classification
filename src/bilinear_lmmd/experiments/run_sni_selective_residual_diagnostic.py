from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.sni_ontology import validate_sni_classes
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "SNIDG": Path("configs/sni/SNIDG_efficientnetv2_selective_gap.yaml"),
    "SNIDH": Path("configs/sni/SNIDH_efficientnetv2_selective_hbp.yaml"),
}
BASELINE_CODE = "SNIB1"
SEED = 42


def screening_decision(summary: dict) -> dict:
    criteria = {
        "macro_f1_improved": float(summary["macro_f1"]["delta_mean"]) > 0.0,
        "hard_f1_improved": float(summary["hard_class_f1"]["delta_mean"]) > 0.0,
        "worst_f1_preserved": (
            float(summary["worst_class_f1"]["delta_mean"]) >= -0.01
        ),
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _validate_dataset(data_root: Path) -> None:
    for split in ("train", "val"):
        split_root = data_root / "source" / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split SNI belum tersedia: {split_root}")
        validate_sni_classes(
            sorted(path.name for path in split_root.iterdir() if path.is_dir())
        )


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    if not history_path.is_file() or not run_dir.joinpath("best.pt").is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def _evaluate(checkpoint: Path, destination: Path, data_root: Path) -> None:
    if destination.joinpath("metrics.json").is_file():
        print(f"SKIP evaluasi lengkap: {destination.name}", flush=True)
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
            "source",
            "--split",
            "val",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(destination),
        ]
    )


def _audit_capacity() -> dict:
    audits = {}
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        model = build_model(cfg["model"])
        audits[code] = {
            "head": cfg["model"]["head"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_after_freeze": sum(
                parameter.numel()
                for name in cfg["training"]["trainable_modules"]
                for parameter in model.get_submodule(name).parameters()
            ),
        }
    if audits["SNIDG"]["parameters"] != audits["SNIDH"]["parameters"]:
        raise AssertionError("Kontrol GAP dan HBP tidak capacity-matched.")
    if audits["SNIDG"]["trainable_after_freeze"] != audits["SNIDH"]["trainable_after_freeze"]:
        raise AssertionError("Parameter trainable kontrol GAP dan HBP berbeda.")
    return audits


def _metrics(root: Path, code: str) -> Path:
    return root / "val_reports" / f"{code}_seed{SEED}" / "metrics.json"


def _compare(
    baseline_path: Path,
    candidate_path: Path,
    destination: Path,
) -> dict:
    result = aggregate([baseline_path], [candidate_path])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result["summary"]


def run_diagnostic(
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
) -> dict:
    _validate_dataset(data_root)
    baseline_checkpoint = (
        baseline_root / "outputs" / f"{BASELINE_CODE}_seed{SEED}" / "best.pt"
    )
    baseline_metrics = _metrics(baseline_root, BASELINE_CODE)
    for required in (baseline_checkpoint, baseline_metrics):
        if not required.is_file():
            raise FileNotFoundError(f"Artefak SNIB1 seed 42 belum tersedia: {required}")

    audits = _audit_capacity()
    print("=== SELECTIVE RESIDUAL DIAGNOSTIC (VALIDATION ONLY) ===", flush=True)
    print(f"Warm start: {baseline_checkpoint}", flush=True)
    for code, row in audits.items():
        print(
            f"{code}: head={row['head']} params={row['parameters']:,} "
            f"trainable={row['trainable_after_freeze']:,}",
            flush=True,
        )
    print("Capacity match: PASS | epoch=10 | seed=42 | TEST LOCKED", flush=True)

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        run_dir = output_root / "outputs" / f"{code}_seed{SEED}"
        if _training_complete(run_dir, epochs):
            print(f"SKIP training lengkap: {code} seed {SEED}", flush=True)
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
                    str(SEED),
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(run_dir),
                    "--init-checkpoint",
                    str(baseline_checkpoint),
                    "--resume",
                ]
            )
        _evaluate(
            run_dir / "best.pt",
            output_root / "val_reports" / f"{code}_seed{SEED}",
            data_root,
        )

    comparisons = {
        "SNIB1_vs_SNIDG": _compare(
            baseline_metrics,
            _metrics(output_root, "SNIDG"),
            output_root / "val_reports" / "SNIB1_vs_SNIDG.json",
        ),
        "SNIB1_vs_SNIDH": _compare(
            baseline_metrics,
            _metrics(output_root, "SNIDH"),
            output_root / "val_reports" / "SNIB1_vs_SNIDH.json",
        ),
        "SNIDG_vs_SNIDH": _compare(
            _metrics(output_root, "SNIDG"),
            _metrics(output_root, "SNIDH"),
            output_root / "val_reports" / "SNIDG_vs_SNIDH.json",
        ),
    }
    decisive = ("SNIB1_vs_SNIDH", "SNIDG_vs_SNIDH")
    decisions = {name: screening_decision(comparisons[name]) for name in decisive}
    final_decision = (
        "GO_FULL_CONFIRMATION"
        if all(row["decision"] == "PASS" for row in decisions.values())
        else "STOP"
    )
    report = {
        "protocol": "selective residual HBP head-only diagnostic v1",
        "seed": SEED,
        "epochs": 10,
        "selection_split": "val",
        "test_opened": False,
        "warm_start": str(baseline_checkpoint),
        "audits": audits,
        "comparisons": comparisons,
        "decisions": decisions,
        "final_decision": final_decision,
    }
    destination = output_root / "val_reports" / "selective_residual_diagnostic.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== PUTUSAN DIAGNOSTIK ===")
    for name, row in decisions.items():
        print(f"{name}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final_decision)
    print("TEST DIBUKA: False")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed-42 validation-only selective GAP/HBP residual diagnostic"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_diagnostic(args.data_root, args.baseline_root, args.output_root)


if __name__ == "__main__":
    main()
