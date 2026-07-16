from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .run_cbd_stacking_confirmation import _metrics, load_prediction_table


KD_CONFIGS = {
    "GAP_KD_CONTROL": (
        Path("configs/CBD4_mobilenetv3_gap_kd_gapcal.yaml"),
        "gap_cal",
    ),
    "STACKING_KD": (
        Path("configs/CBD5_mobilenetv3_gap_kd_stacking.yaml"),
        "stacking",
    ),
}
MODEL_NAMES = ("GAP_RAW", "GAP_KD_CONTROL", "STACKING_KD", "STACKING_TEACHER")
METRICS = ("accuracy", "macro_f1", "defect_f1", "worst_f1")


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


def _prediction_metrics(path: Path) -> dict:
    table = load_prediction_table(path)
    return _metrics(table.labels, table.probabilities.argmax(axis=1), table.classes)


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def aggregate_kd_results(seed_results: dict[int, dict]) -> dict:
    models = {
        model: {
            metric: _mean_std(
                [seed_results[seed][model][metric] for seed in seed_results]
            )
            for metric in METRICS
        }
        for model in MODEL_NAMES
    }
    deltas = {}
    for control in ("GAP_RAW", "GAP_KD_CONTROL"):
        deltas[control] = {}
        for metric in METRICS:
            values = [
                seed_results[seed]["STACKING_KD"][metric]
                - seed_results[seed][control][metric]
                for seed in seed_results
            ]
            deltas[control][metric] = {
                **_mean_std(values),
                "improved_seeds": sum(value > 0 for value in values),
                "total_seeds": len(values),
            }

    student_gain = (
        models["STACKING_KD"]["macro_f1"]["mean"]
        - models["GAP_RAW"]["macro_f1"]["mean"]
    )
    teacher_gain = (
        models["STACKING_TEACHER"]["macro_f1"]["mean"]
        - models["GAP_RAW"]["macro_f1"]["mean"]
    )
    preservation = student_gain / teacher_gain if teacher_gain > 0 else None
    full_confirmation = len(seed_results) >= 3
    criteria = {
        "macro_gain_vs_gap_minimum": student_gain >= 0.003,
        "worst_not_below_gap": (
            models["STACKING_KD"]["worst_f1"]["mean"]
            >= models["GAP_RAW"]["worst_f1"]["mean"]
        ),
        "fusion_teacher_beats_calibration_teacher": (
            models["STACKING_KD"]["macro_f1"]["mean"]
            > models["GAP_KD_CONTROL"]["macro_f1"]["mean"]
        ),
        "macro_improves_vs_gap_on_at_least_2_of_3_seeds": (
            deltas["GAP_RAW"]["macro_f1"]["improved_seeds"] >= 2
            if full_confirmation
            else None
        ),
    }
    passed = full_confirmation and all(value for value in criteria.values())
    return {
        "seeds": list(seed_results),
        "models": models,
        "stacking_kd_deltas": deltas,
        "teacher_gain_preserved": preservation,
        "decision": {
            "status": "PASS" if passed else ("FAIL" if full_confirmation else "SCREEN_ONLY"),
            "criteria": criteria,
            "macro_gain_vs_gap": student_gain,
            "macro_gain_vs_kd_calibration_control": (
                models["STACKING_KD"]["macro_f1"]["mean"]
                - models["GAP_KD_CONTROL"]["macro_f1"]["mean"]
            ),
            "worst_gain_vs_gap": (
                models["STACKING_KD"]["worst_f1"]["mean"]
                - models["GAP_RAW"]["worst_f1"]["mean"]
            ),
        },
    }


def run_kd_confirmation(data_root: Path, output_root: Path, seeds: list[int]) -> dict:
    stacking_summary = output_root / "stacking_reports" / "stacking_confirmation.json"
    if not stacking_summary.is_file():
        raise FileNotFoundError(
            "Jalankan run_cbd_stacking_confirmation terlebih dahulu: "
            f"{stacking_summary}"
        )
    stacking_aggregate = json.loads(stacking_summary.read_text(encoding="utf-8"))
    stacking_status = stacking_aggregate["pre_registered_decision"]["status"]
    available_seeds = set(stacking_aggregate.get("seeds", []))
    if not set(seeds).issubset(available_seeds):
        raise RuntimeError(
            "Ringkasan stacking tidak memuat seluruh seed KD yang diminta."
        )
    if stacking_status != "PASS":
        one_seed_screen = len(seeds) == 1 and stacking_status == "SCREEN_ONLY"
        if not one_seed_screen:
            raise RuntimeError("KD dibatalkan karena teacher stacking belum PASS.")
        print(
            "KD SCREENING: teacher stacking satu seed; hasil bukan konfirmasi.",
            flush=True,
        )

    for seed in seeds:
        gap_checkpoint = output_root / "outputs" / f"CBD0_seed{seed}" / "best.pt"
        hbp_checkpoint = output_root / "outputs" / f"CBD1_seed{seed}" / "best.pt"
        gap_val = output_root / "val_reports" / f"CBD0_seed{seed}" / "predictions.csv"
        hbp_val = output_root / "val_reports" / f"CBD1_seed{seed}" / "predictions.csv"
        required = (gap_checkpoint, hbp_checkpoint, gap_val, hbp_val)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Artefak teacher belum lengkap:\n- " + "\n- ".join(missing))

        for code, (config_path, teacher_kind) in KD_CONFIGS.items():
            epochs = int(load_config(config_path)["training"]["epochs"])
            run_dir = output_root / "kd_outputs" / f"{code}_seed{seed}"
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "bilinear_lmmd.train_cbd_distillation",
                    "--config",
                    str(config_path),
                    "--teacher",
                    teacher_kind,
                    "--gap-checkpoint",
                    str(gap_checkpoint),
                    "--gap-val-predictions",
                    str(gap_val),
                    "--seed",
                    str(seed),
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(run_dir),
                    "--resume",
                ]
                if teacher_kind == "stacking":
                    command.extend(
                        [
                            "--hbp-checkpoint",
                            str(hbp_checkpoint),
                            "--hbp-val-predictions",
                            str(hbp_val),
                        ]
                    )
                _run(command)

            report_dir = output_root / "kd_reports" / f"{code}_seed{seed}"
            if (report_dir / "predictions.csv").is_file():
                print(f"SKIP evaluasi lengkap: {code} seed {seed}", flush=True)
            else:
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.evaluate_checkpoint",
                        "--checkpoint",
                        str(run_dir / "best.pt"),
                        "--domain",
                        "source",
                        "--split",
                        "test",
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    seed_results = {}
    for seed in seeds:
        stacking_metrics_path = (
            output_root / "stacking_reports" / f"seed{seed}" / "metrics.json"
        )
        if not stacking_metrics_path.is_file():
            raise FileNotFoundError(f"Metrics stacking seed {seed} tidak ada.")
        stacking_metrics = json.loads(stacking_metrics_path.read_text(encoding="utf-8"))
        seed_results[seed] = {
            "GAP_RAW": _prediction_metrics(
                output_root / "reports" / f"CBD0_seed{seed}" / "predictions.csv"
            ),
            "GAP_KD_CONTROL": _prediction_metrics(
                output_root
                / "kd_reports"
                / f"GAP_KD_CONTROL_seed{seed}"
                / "predictions.csv"
            ),
            "STACKING_KD": _prediction_metrics(
                output_root
                / "kd_reports"
                / f"STACKING_KD_seed{seed}"
                / "predictions.csv"
            ),
            "STACKING_TEACHER": stacking_metrics["models"]["STACKING"],
        }
        print(f"\n=== KD SEED {seed} ===")
        for model in MODEL_NAMES:
            row = seed_results[seed][model]
            print(
                f"{model:17s} Macro={row['macro_f1']:.2%} "
                f"Defect={row['defect_f1']:.2%} Worst={row['worst_f1']:.2%}"
            )

    aggregate = aggregate_kd_results(seed_results)
    destination = output_root / "kd_reports" / "kd_confirmation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("\n=== AGREGAT KD ===")
    for model in MODEL_NAMES:
        row = aggregate["models"][model]
        print(
            f"{model:17s} Macro={row['macro_f1']['mean']:.2%}±"
            f"{row['macro_f1']['std']:.2%} Worst={row['worst_f1']['mean']:.2%}±"
            f"{row['worst_f1']['std']:.2%}"
        )
    decision = aggregate["decision"]
    print(
        f"Delta STACKING_KD vs GAP: Macro={decision['macro_gain_vs_gap']:+.2%} "
        f"Worst={decision['worst_gain_vs_gap']:+.2%}"
    )
    print(
        "Delta Macro vs KD calibration control: "
        f"{decision['macro_gain_vs_kd_calibration_control']:+.2%}"
    )
    preservation = aggregate["teacher_gain_preserved"]
    print(
        "Teacher gain preserved: "
        + (f"{preservation:.1%}" if preservation is not None else "N/A")
    )
    print(f"KEPUTUSAN: {decision['status']}")
    print(f"SAVED: {destination}")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Konfirmasi KD teacher GAP-HBP stacking")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    run_kd_confirmation(args.data_root, args.output_root, args.seeds)


if __name__ == "__main__":
    main()
