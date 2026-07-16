from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .aggregate_ablation import aggregate
from .config import load_config
from .models import build_model
from .prepare_cbd_multiclassify import EXPECTED_CLASSES, prepare_cbd_multiclassify


MODEL_CONFIGS = {
    "CBD0": Path("configs/CBD0_mobilenetv3_gap_source.yaml"),
    "CBD1": Path("configs/CBD1_mobilenetv3_hbp_source.yaml"),
    "CBD2": Path("configs/CBD2_mobilenetv3_gap_balanced_softmax.yaml"),
    "CBD3": Path("configs/CBD3_mobilenetv3_hbp_balanced_softmax.yaml"),
}
COMPARISONS = (
    ("CBD0", "CBD1", "efek HBP dengan CE"),
    ("CBD0", "CBD2", "efek Balanced Softmax pada GAP"),
    ("CBD1", "CBD3", "efek Balanced Softmax pada HBP"),
    ("CBD2", "CBD3", "efek HBP dengan Balanced Softmax"),
)


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
    return sum(parameter.numel() for parameter in build_model(cfg["model"]).parameters())


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _ensure_data(raw_root: Path | None, data_root: Path, seed: int) -> dict:
    if not (data_root / "source" / "train").is_dir():
        if raw_root is None:
            raise FileNotFoundError(
                f"Dataset siap training belum ada: {data_root}. Berikan --raw-root."
            )
        prepare_cbd_multiclassify(raw_root, data_root, seed)
    audit_path = data_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit cbd-multiclassify tidak ada: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("classes") != list(EXPECTED_CLASSES):
        raise RuntimeError("Urutan/daftar kelas audit tidak sesuai konfigurasi model.")
    if audit.get("generated_cross_split_identity_count") != 0:
        raise RuntimeError("Identity leakage ditemukan pada split hasil preparasi.")
    return audit


def run_cbd_multiclassify_screening(
    raw_root: Path | None,
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    evaluation_split: str,
) -> None:
    audit = _ensure_data(raw_root, data_root, seeds[0])
    print("\n=== CBD-MULTICLASSIFY: GAP vs HBP ===")
    print(f"Data bersih : {audit['clean_labeled_images']}")
    print(f"Kelas       : {', '.join(audit['classes'])}")
    print(f"Split       : {audit['split_strategy']}")
    print(f"Evaluasi    : {evaluation_split}")
    print("KLAIM       : benchmark label kasar terpisah, bukan Coffee-17 external test")
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
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.train",
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
            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {code} | seed {seed}", flush=True)
            else:
                print(f"EVALUATE: {code} | seed {seed}", flush=True)
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
                        evaluation_split,
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    report_root = _report_root(output_root, evaluation_split)
    for baseline_code, candidate_code, description in COMPARISONS:
        baseline = [
            report_root / f"{baseline_code}_seed{seed}" / "metrics.json"
            for seed in seeds
        ]
        candidate = [
            report_root / f"{candidate_code}_seed{seed}" / "metrics.json"
            for seed in seeds
        ]
        result = aggregate(baseline, candidate)
        destination = (
            report_root / f"{baseline_code}_vs_{candidate_code}_aggregate.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"\n=== {baseline_code} vs {candidate_code}: {description} ==="
        )
        for key, label in (
            ("accuracy", "Accuracy "),
            ("macro_f1", "Macro-F1"),
            ("hard_class_f1", "Defect-F1"),
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
        description="Grouped screening GAP vs HBP pada Roboflow cbd-multiclassify"
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/cbd_multiclassify_prepared")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"), default="val"
    )
    args = parser.parse_args()
    run_cbd_multiclassify_screening(
        args.raw_root,
        args.data_root,
        args.output_root,
        args.seeds,
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
