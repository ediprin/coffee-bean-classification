from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.sni_ontology import SNI_CLASSES, validate_sni_classes
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.reporting.aggregate_ablation import aggregate


MODEL_CONFIGS = {
    "SNIB0": Path("configs/sni/SNIB0_efficientnetv2_gap.yaml"),
    "SNIB1": Path("configs/sni/SNIB1_efficientnetv2_multiresolution_flat.yaml"),
    "SNIB2": Path("configs/sni/SNIB2_efficientnetv2_mre_gap.yaml"),
    "SNIB3": Path("configs/sni/SNIB3_efficientnetv2_mrenet.yaml"),
}
COMPARISONS = (
    ("SNIB0", "SNIB1"),
    ("SNIB1", "SNIB2"),
    ("SNIB2", "SNIB3"),
    ("SNIB0", "SNIB3"),
)
STAGE_MODELS = {
    "backbone": ("SNIB0", "SNIB1"),
    "ontology": ("SNIB2",),
    "bilinear": ("SNIB3",),
    "all": tuple(MODEL_CONFIGS),
}
STAGE_COMPARISONS = {
    "backbone": (("SNIB0", "SNIB1"),),
    "ontology": (("SNIB1", "SNIB2"),),
    "bilinear": (("SNIB2", "SNIB3"), ("SNIB0", "SNIB3")),
    "all": COMPARISONS,
}


def screening_decision(summary: dict) -> dict:
    criteria = {
        "macro_f1_improved": float(summary["macro_f1"]["delta_mean"]) > 0.0,
        "hard_f1_improved": float(summary["hard_class_f1"]["delta_mean"]) > 0.0,
        "worst_f1_preserved": float(summary["worst_class_f1"]["delta_mean"])
        >= -0.01,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
    }


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    checkpoint = run_dir / "best.pt"
    if not history.is_file() or not checkpoint.is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (OSError, json.JSONDecodeError):
        return False


def _validate_dataset(data_root: Path) -> None:
    for split in ("train", "val"):
        split_root = data_root / "source" / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split SNI belum tersedia: {split_root}")
        classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        validate_sni_classes(classes)


def _audit_models() -> dict:
    audits = {}
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"])
        audits[code] = {
            "head": cfg["model"]["head"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    if audits["SNIB2"]["parameters"] != audits["SNIB3"]["parameters"]:
        raise AssertionError(
            "SNIB2 dan SNIB3 tidak capacity-matched: "
            f"{audits['SNIB2']['parameters']:,} vs "
            f"{audits['SNIB3']['parameters']:,}."
        )
    return audits


def _evaluate(checkpoint: Path, output_dir: Path, data_root: Path) -> None:
    if output_dir.joinpath("metrics.json").is_file():
        print(f"SKIP evaluasi lengkap: {output_dir.name}", flush=True)
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
            str(output_dir),
        ]
    )


def _metrics_path(output_root: Path, code: str, seed: int) -> Path:
    return output_root / "val_reports" / f"{code}_seed{seed}" / "metrics.json"


def _compare(
    output_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
) -> dict:
    result = aggregate(
        [_metrics_path(output_root, baseline, seed) for seed in seeds],
        [_metrics_path(output_root, candidate, seed) for seed in seeds],
    )
    destination = (
        output_root
        / "val_reports"
        / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate} ===")
    for metric, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][metric]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    return result


def run_sni_mrenet_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    stage: str = "all",
) -> dict:
    if stage not in STAGE_MODELS:
        raise ValueError(f"Stage harus salah satu dari {sorted(STAGE_MODELS)}.")
    _validate_dataset(data_root)
    audits = _audit_models()
    print("=== SNI-MRENET VALIDATION-ONLY PROTOCOL ===", flush=True)
    print(f"Classes: {len(SNI_CLASSES)}", flush=True)
    for code, row in audits.items():
        print(
            f"{code}: head={row['head']} params={row['parameters']:,}",
            flush=True,
        )
    print("SNIB2/SNIB3 capacity match: PASS", flush=True)

    for code in STAGE_MODELS[stage]:
        config_path = MODEL_CONFIGS[code]
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
    decisions = {}
    for baseline, candidate in STAGE_COMPARISONS[stage]:
        key = f"{baseline}_vs_{candidate}"
        missing = [
            path
            for code in (baseline, candidate)
            for seed in seeds
            if not (path := _metrics_path(output_root, code, seed)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Stage {stage} membutuhkan hasil tahap sebelumnya: {missing}"
            )
        summary = _compare(output_root, baseline, candidate, seeds)["summary"]
        comparisons[key] = summary
        decisions[key] = screening_decision(summary)

    final_decision = (
        "PASS"
        if decisions and all(row["decision"] == "PASS" for row in decisions.values())
        else "FAIL"
    )

    report = {
        "method": "SNI Ontology-Guided Multi-Resolution Expert Network",
        "classes": list(SNI_CLASSES),
        "stage": stage,
        "seeds": seeds,
        "selection_split": "val",
        "test_opened": False,
        "audits": audits,
        "capacity_matched_comparison": ["SNIB2", "SNIB3"],
        "comparisons": comparisons,
        "decisions": decisions,
        "final_decision": final_decision,
    }
    destination = output_root / "val_reports" / f"sni_mrenet_{stage}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== PUTUSAN STAGE {stage.upper()} ===")
    for name, row in decisions.items():
        print(f"{name}: {row['decision']} | {row['criteria']}")
    print("FINAL:", final_decision)
    print("\nTEST TETAP TERKUNCI: True")
    print("SAVED:", destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only controlled SNI-MRENet B0-B3 runner"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_MODELS),
        default="all",
        help="Jalankan satu tahap fail-fast atau seluruh ablation.",
    )
    parser.add_argument("--evaluation-split", choices=("val",), default="val")
    args = parser.parse_args()
    run_sni_mrenet_screening(
        args.data_root, args.output_root, args.seeds, stage=args.stage
    )


if __name__ == "__main__":
    main()
