from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.reporting.aggregate_ablation import aggregate
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model


PRESETS = {
    "coffee17": {
        "M0": Path("configs/coffee17/M0_mobilenetv3_gap_source.yaml"),
        "M1": Path("configs/coffee17/M1_mobilenetv3_hbp_source.yaml"),
        "D1": Path("configs/coffee17/D1_mobilenetv3_decoupled_gap_hbp_fixed_source.yaml"),
        "D2": Path("configs/coffee17/D2_mobilenetv3_decoupled_gap_hbp_learned_source.yaml"),
    },
    "cbd": {
        "CBD0": Path("configs/cbd/CBD0_mobilenetv3_gap_source.yaml"),
        "CBD1": Path("configs/cbd/CBD1_mobilenetv3_hbp_source.yaml"),
        "CBDC1": Path("configs/cbd/CBDC1_mobilenetv3_capacity_residual_hbp_source.yaml"),
        "CBDD1": Path("configs/cbd/CBDD1_mobilenetv3_decoupled_gap_hbp_fixed_source.yaml"),
        "CBDD2": Path("configs/cbd/CBDD2_mobilenetv3_decoupled_gap_hbp_learned_source.yaml"),
    },
}

PRESET_ROLES = {
    "coffee17": {"gap": "M0", "hbp": "M1", "fixed": "D1", "learned": "D2"},
    "cbd": {
        "gap": "CBD0",
        "hbp": "CBD1",
        "capacity": "CBDC1",
        "fixed": "CBDD1",
        "learned": "CBDD2",
    },
}


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


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _evaluate(
    checkpoint: Path,
    report_dir: Path,
    data_root: Path,
    split: str,
    prediction_head: str = "fused",
) -> None:
    metrics_path = report_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"SKIP report: {report_dir.name}", flush=True)
        return
    command = [
        sys.executable,
        "-u",
        "-m",
        "bilinear_lmmd.engine.evaluate_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--domain",
        "source",
        "--split",
        split,
        "--data-root",
        str(data_root),
        "--output-dir",
        str(report_dir),
        "--prediction-head",
        prediction_head,
    ]
    _run(command)


def _metric_paths(
    report_root: Path, label: str, seeds: list[int]
) -> list[Path]:
    return [report_root / f"{label}_seed{seed}" / "metrics.json" for seed in seeds]


def _compare(
    report_root: Path,
    baseline: str,
    candidate: str,
    seeds: list[int],
    description: str,
) -> None:
    baseline_paths = _metric_paths(report_root, baseline, seeds)
    candidate_paths = _metric_paths(report_root, candidate, seeds)
    if any(not path.is_file() for path in baseline_paths + candidate_paths):
        return
    result = aggregate(baseline_paths, candidate_paths)
    destination = report_root / f"{baseline}_vs_{candidate}_aggregate.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate}: {description} ===")
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        row = result["summary"][key]
        print(
            f"{label}: {row['baseline_mean']:.2%} -> "
            f"{row['candidate_mean']:.2%} ({row['delta_mean']:+.2%})"
        )
    print(f"SAVED: {destination}")


def _complementarity(
    report_root: Path, model: str, seed: int
) -> dict | None:
    paths = {
        head: report_root / f"{model}_{head}_seed{seed}" / "predictions.csv"
        for head in ("gap", "hbp")
    }
    if any(not path.is_file() for path in paths.values()):
        return None
    rows = {
        head: list(csv.DictReader(path.open(encoding="utf-8")))
        for head, path in paths.items()
    }
    if len(rows["gap"]) != len(rows["hbp"]):
        raise RuntimeError("Jumlah prediksi expert GAP dan HBP berbeda.")
    counts = {"both_correct": 0, "gap_only": 0, "hbp_only": 0, "both_wrong": 0}
    for gap, hbp in zip(rows["gap"], rows["hbp"]):
        if gap["path"] != hbp["path"] or gap["actual"] != hbp["actual"]:
            raise RuntimeError("Urutan prediksi expert GAP dan HBP berbeda.")
        gap_correct = gap["correct"] == "1"
        hbp_correct = hbp["correct"] == "1"
        if gap_correct and hbp_correct:
            counts["both_correct"] += 1
        elif gap_correct:
            counts["gap_only"] += 1
        elif hbp_correct:
            counts["hbp_only"] += 1
        else:
            counts["both_wrong"] += 1
    total = len(rows["gap"])
    agreement = sum(
        gap["predicted"] == hbp["predicted"]
        for gap, hbp in zip(rows["gap"], rows["hbp"])
    )
    return {
        "model": model,
        "seed": seed,
        "total": total,
        **counts,
        "prediction_agreement": agreement / total,
        "oracle_accuracy": (total - counts["both_wrong"]) / total,
    }


def run_decoupled_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    evaluation_split: str,
    preset: str = "coffee17",
    models: list[str] | None = None,
) -> None:
    model_configs = PRESETS[preset]
    roles = PRESET_ROLES[preset]
    selected_models = models or list(model_configs)
    unknown = sorted(set(selected_models).difference(model_configs))
    if unknown:
        raise ValueError(
            f"Model tidak tersedia untuk preset {preset}: {unknown}; "
            f"pilihan={list(model_configs)}"
        )
    report_root = _report_root(output_root, evaluation_split)
    report_root.mkdir(parents=True, exist_ok=True)
    print(f"=== RANCANGAN DECOUPLED GAP-HBP: {preset.upper()} ===")
    for code in selected_models:
        config_path = model_configs[code]
        cfg = load_config(config_path)
        cfg["model"]["pretrained"] = False
        parameters = sum(
            parameter.numel() for parameter in build_model(cfg["model"]).parameters()
        )
        print(f"{code}: head={cfg['model']['head']} params={parameters:,}")

    for code in selected_models:
        config_path = model_configs[code]
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
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
                report_root / f"{code}_seed{seed}",
                data_root,
                evaluation_split,
            )
            if code not in {roles["fixed"], roles["learned"]}:
                continue
            # Expert diagnostics use the same fused-selected checkpoint.
            for head in ("gap", "hbp"):
                _evaluate(
                    run_dir / "best.pt",
                    report_root / f"{code}_{head}_seed{seed}",
                    data_root,
                    evaluation_split,
                    prediction_head=head,
                )
            # Detachable result uses the validation-selected GAP checkpoint.
            _evaluate(
                run_dir / "best_gap.pt",
                report_root / f"{code}_detachable_gap_seed{seed}",
                data_root,
                evaluation_split,
                prediction_head="gap",
            )

    comparisons = [
        (roles["gap"], roles["hbp"], "kontrol efek HBP"),
        (roles["gap"], roles["fixed"], "fixed fusion decoupled"),
        (roles["hbp"], roles["fixed"], "fixed fusion vs HBP"),
        (roles["fixed"], roles["learned"], "learned gate vs fixed fusion"),
        (roles["hbp"], roles["learned"], "learned dual-branch vs HBP"),
        (
            roles["gap"],
            f"{roles['learned']}_detachable_gap",
            "HBP sebagai auxiliary training-only",
        ),
    ]
    if "capacity" in roles:
        comparisons.extend(
            [
                (roles["hbp"], roles["capacity"], "efek kapasitas tambahan pada HBP"),
                (
                    roles["capacity"],
                    roles["learned"],
                    "dual GAP-HBP vs capacity-matched HBP",
                ),
            ]
        )
    for baseline, candidate, description in comparisons:
        _compare(report_root, baseline, candidate, seeds, description)

    audit_path = report_root / "expert_complementarity.json"
    existing_audits = []
    if audit_path.is_file():
        try:
            existing_audits = json.loads(audit_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_audits = []
    audits_by_key = {
        (row.get("model"), row.get("seed")): row
        for row in existing_audits
        if isinstance(row, dict)
    }
    for model in (roles["fixed"], roles["learned"]):
        for seed in seeds:
            result = _complementarity(report_root, model, seed)
            if result is not None:
                audits_by_key[(model, seed)] = result
                print(
                    f"{model} seed {seed}: agreement={result['prediction_agreement']:.2%} "
                    f"GAP-only={result['gap_only']} HBP-only={result['hbp_only']} "
                    f"oracle={result['oracle_accuracy']:.2%}"
                )
    audits = sorted(
        audits_by_key.values(), key=lambda row: (str(row.get("model")), int(row.get("seed", 0)))
    )
    audit_path.write_text(
        json.dumps(audits, indent=2), encoding="utf-8"
    )
    print("\nSCREENING SELESAI. Jangan buka test sebelum kandidat validation dikunci.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoupled dual-branch GAP-HBP")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--preset", choices=tuple(PRESETS), default="coffee17")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subset kode model pada preset, misalnya CBD1 CBDD2.",
    )
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"), default="val"
    )
    args = parser.parse_args()
    run_decoupled_screening(
        args.data_root,
        args.output_root,
        args.seeds,
        args.evaluation_split,
        args.preset,
        args.models,
    )


if __name__ == "__main__":
    main()
