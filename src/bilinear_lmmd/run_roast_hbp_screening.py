from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .aggregate_ablation import aggregate
from .config import load_config
from .models import build_model
from .prepare_roast_coffee import prepare_roast_coffee


MODEL_CONFIGS = {
    "R0": Path("configs/R0_roast_mobilenetv3_gap_source.yaml"),
    "R1": Path("configs/R1_roast_mobilenetv3_hbp_source.yaml"),
}


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    best = run_dir / "best.pt"
    if not history.is_file() or not best.is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _parameter_count(path: Path) -> int:
    cfg = load_config(path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    return sum(parameter.numel() for parameter in model.parameters())


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _ensure_data(raw_root: Path | None, data_root: Path, seed: int) -> dict:
    if not (data_root / "source" / "train").is_dir():
        if raw_root is None:
            raise FileNotFoundError(
                f"Dataset siap training belum ada: {data_root}. Berikan --raw-root."
            )
        prepare_roast_coffee(raw_root, data_root, seed)
    audit_path = data_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit dataset tidak ditemukan: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cross_split = audit.get("cross_split_exact_duplicates", [])
    if cross_split:
        print(
            f"INFO: {len(cross_split)} grup duplikat lintas split sudah "
            "dideduplikasi; salinan test/val diprioritaskan.",
            flush=True,
        )
    if audit.get("cross_class_exact_conflicts"):
        raise RuntimeError(
            "Gambar identik memiliki label berbeda. Periksa audit.json sebelum training."
        )
    return audit


def run_roast_hbp_screening(
    raw_root: Path | None,
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    evaluation_split: str = "val",
) -> None:
    audit = _ensure_data(raw_root, data_root, seeds[0])
    print("\n=== CONTROLLED HBP SCREENING: COFFEE ROAST ===")
    print(f"Data bersih    : {audit['clean_images']}")
    print(f"Strategi split : {audit['split_strategy']}")
    print(f"Evaluasi       : {evaluation_split}")
    if evaluation_split == "test":
        print("Referensi paper: Swin-HSSAM Accuracy=96.50%, F1=96.48%")
    else:
        print("Test paper belum dibandingkan pada tahap validation screening.")
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        print(
            f"{code}: head={cfg['model']['head']} "
            f"params={_parameter_count(config_path):,}"
        )

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = _report_root(output_root, evaluation_split) / f"{code}_seed{seed}"
            if _complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} | seed {seed}", flush=True)
            else:
                print(f"START training: {code} | seed {seed}", flush=True)
                _run(
                    [
                        sys.executable, "-u", "-m", "bilinear_lmmd.train",
                        "--config", str(config_path),
                        "--seed", str(seed),
                        "--data-root", str(data_root),
                        "--output-dir", str(run_dir),
                        "--resume",
                    ]
                )
            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {code} | seed {seed}", flush=True)
            else:
                print(f"EVALUATE: {code} | seed {seed}", flush=True)
                _run(
                    [
                        sys.executable, "-u", "-m", "bilinear_lmmd.evaluate_checkpoint",
                        "--checkpoint", str(run_dir / "best.pt"),
                        "--domain", "source",
                        "--split", evaluation_split,
                        "--data-root", str(data_root),
                        "--output-dir", str(report_dir),
                    ]
                )

    baseline = [
        _report_root(output_root, evaluation_split) / f"R0_seed{seed}" / "metrics.json"
        for seed in seeds
    ]
    candidate = [
        _report_root(output_root, evaluation_split) / f"R1_seed{seed}" / "metrics.json"
        for seed in seeds
    ]
    result = aggregate(baseline, candidate)
    destination = _report_root(output_root, evaluation_split) / "R0_vs_R1_aggregate.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== R0 vs R1: EFEK HBP PADA MOBILENETV3 ===")
    for key, label in (
        ("accuracy", "Accuracy "),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uji terkontrol efek HBP pada Coffee Bean Roast dataset"
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/coffee_roast_prepared")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"), default="val"
    )
    args = parser.parse_args()
    run_roast_hbp_screening(
        args.raw_root,
        args.data_root,
        args.output_root,
        args.seeds,
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
