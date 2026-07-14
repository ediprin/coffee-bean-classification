from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .compare_reports import compare
from .config import load_config
from .merge_oof import merge_oof


MODEL_CONFIGS = {
    "M0": Path("configs/M0_mobilenetv3_gap_source.yaml"),
    "M0b": Path("configs/M0b_mobilenetv3_bilinear_source.yaml"),
    "M1": Path("configs/M1_mobilenetv3_hbp_source.yaml"),
}


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_grouped_cv(
    data_root: Path,
    output_root: Path,
    models: list[str],
    folds: int,
    seed: int,
) -> None:
    fold_roots = [data_root / f"fold_{index}" for index in range(1, folds + 1)]
    missing = [str(path) for path in fold_roots if not (path / "source").is_dir()]
    if missing:
        raise FileNotFoundError(
            "Fold belum disiapkan:\n- "
            + "\n- ".join(missing)
            + "\nJalankan bilinear_lmmd.prepare_grouped_folds terlebih dahulu."
        )

    oof_paths: dict[str, Path] = {}
    for model_code in models:
        config_path = MODEL_CONFIGS[model_code]
        cfg = load_config(config_path)
        epochs = int(cfg["training"]["epochs"])
        report_dirs: list[Path] = []
        for fold_index, fold_root in enumerate(fold_roots, start=1):
            label = f"{model_code} | fold {fold_index}/{folds} | seed {seed}"
            run_dir = output_root / "outputs" / f"{model_code}_fold{fold_index}_seed{seed}"
            report_dir = output_root / "reports" / f"{model_code}_fold{fold_index}_seed{seed}"
            report_dirs.append(report_dir)

            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {label}", flush=True)
            else:
                print(f"START training: {label}", flush=True)
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.train",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--data-root",
                        str(fold_root),
                        "--output-dir",
                        str(run_dir),
                    ]
                )

            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {label}", flush=True)
            else:
                print(f"EVALUATE: {label}", flush=True)
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
                        "--output-dir",
                        str(report_dir),
                    ]
                )

        oof_dir = output_root / "oof" / f"{model_code}_seed{seed}"
        print(f"MERGE OOF: {model_code}", flush=True)
        merge_oof(report_dirs, oof_dir)
        oof_paths[model_code] = oof_dir / "metrics.json"

    if "M0" in oof_paths:
        for candidate in ("M0b", "M1"):
            if candidate not in oof_paths:
                continue
            result = compare(oof_paths["M0"], oof_paths[candidate])
            comparison_path = output_root / "oof" / f"M0_vs_{candidate}_seed{seed}.json"
            comparison_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"SAVED: {comparison_path}", flush=True)
    print("\nPASS: seluruh grouped CV yang diminta sudah lengkap.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner grouped CV yang aman dijalankan ulang setelah interupsi"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/coffee_5fold"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIGS), default=["M0", "M1"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_grouped_cv(args.data_root, args.output_root, args.models, args.folds, args.seed)


if __name__ == "__main__":
    main()
