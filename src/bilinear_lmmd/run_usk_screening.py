from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .aggregate_ablation import aggregate
from .config import load_config
from .models import build_model
from .prepare_usk_coffee import prepare_usk_coffee


MODEL_CONFIGS = {
    "U0": Path("configs/U0_usk_resnet18_gap_source.yaml"),
    "U1": Path("configs/U1_usk_mobilenetv2_gap_source.yaml"),
    "U2": Path("configs/U2_usk_mobilenetv3_gap_source.yaml"),
    "U3": Path("configs/U3_usk_mobilenetv3_hbp_source.yaml"),
}
STAGES = {
    "paper": ["U0", "U1"],
    "quick": ["U1", "U2", "U3"],
    "all": ["U0", "U1", "U2", "U3"],
}
COMPARISONS = (
    ("U0", "U3", "HBP MobileNetV3 vs ResNet-18"),
    ("U1", "U3", "HBP MobileNetV3 vs MobileNetV2 paper"),
    ("U2", "U3", "efek HBP pada MobileNetV3"),
    ("U1", "U2", "MobileNetV3 vs MobileNetV2 dengan GAP"),
)


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _parameter_count(config_path: Path) -> int:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    return sum(parameter.numel() for parameter in build_model(cfg["model"]).parameters())


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _metrics_paths(
    output_root: Path, model: str, seeds: list[int], split: str
) -> list[Path]:
    return [
        _report_root(output_root, split) / f"{model}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]


def _print_comparison(
    baseline: str,
    candidate: str,
    description: str,
    output_root: Path,
    seeds: list[int],
    split: str,
) -> None:
    baseline_paths = _metrics_paths(output_root, baseline, seeds, split)
    candidate_paths = _metrics_paths(output_root, candidate, seeds, split)
    if any(not path.is_file() for path in baseline_paths + candidate_paths):
        return
    result = aggregate(baseline_paths, candidate_paths)
    destination = (
        _report_root(output_root, split)
        / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate}: {description} ===")
    for key, label in (
        ("accuracy", "Accuracy "),
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        item = result["summary"][key]
        print(
            f"{label}: {item['baseline_mean']:.2%} -> "
            f"{item['candidate_mean']:.2%} ({item['delta_mean']:+.2%})"
        )
    print(f"SAVED: {destination}")


def _ensure_dataset(
    raw_root: Path | None,
    data_root: Path,
    seed: int,
    allow_pair_leakage: bool,
) -> dict:
    if not (data_root / "source" / "train").is_dir():
        if raw_root is None:
            raise FileNotFoundError(
                f"Dataset siap training belum ada: {data_root}. Berikan --raw-root."
            )
        prepare_usk_coffee(raw_root, data_root, seed)
    audit_path = data_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit USK tidak ditemukan: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    leaked_pairs = audit.get("cross_split_view_pairs", {})
    if leaked_pairs and not allow_pair_leakage:
        raise RuntimeError(
            f"Terdeteksi {len(leaked_pairs)} pasangan sisi biji lintas split. "
            "Perbaiki split; --allow-pair-leakage hanya untuk reproduksi terpisah."
        )
    return audit


def run_usk_screening(
    raw_root: Path | None,
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    models: list[str],
    evaluation_split: str,
    allow_pair_leakage: bool = False,
) -> None:
    audit = _ensure_dataset(
        raw_root, data_root, seeds[0], allow_pair_leakage
    )
    print("\n=== PROTOKOL USK-COFFEE ===")
    print(f"Data bersih     : {audit['clean_images']}")
    print(f"Strategi split  : {audit['split_strategy']}")
    print(f"Evaluasi        : {evaluation_split}")
    print("Paper reference : ResNet-18=81.13%, MobileNetV2=81.31% test accuracy")
    if evaluation_split != "test":
        print("CATATAN         : validation screening, belum dibandingkan ke angka test paper")
    elif audit["split_strategy"] != "preserved_from_archive":
        print("CATATAN         : split dibuat ulang; angka paper hanya konteks, bukan head-to-head")

    print("\n=== MODEL ===")
    for code in models:
        cfg = load_config(MODEL_CONFIGS[code])
        print(
            f"{code}: backbone={cfg['model']['backbone']} "
            f"head={cfg['model']['head']} params={_parameter_count(MODEL_CONFIGS[code]):,}"
        )

    for model_code in models:
        config_path = MODEL_CONFIGS[model_code]
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            label = f"{model_code} | seed {seed}"
            run_dir = output_root / "outputs" / f"{model_code}_seed{seed}"
            report_dir = (
                _report_root(output_root, evaluation_split)
                / f"{model_code}_seed{seed}"
            )
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
                        str(data_root),
                        "--output-dir",
                        str(run_dir),
                        "--resume",
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
                        evaluation_split,
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    selected = set(models)
    for baseline, candidate, description in COMPARISONS:
        if {baseline, candidate}.issubset(selected):
            _print_comparison(
                baseline,
                candidate,
                description,
                output_root,
                seeds,
                evaluation_split,
            )
    print("\nPASS: screening USK selesai.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dan screening baseline paper vs HBP pada USK-Coffee"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Root dataset mentah dari /kaggle/input; tidak perlu setelah preparasi.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/usk_coffee_prepared")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--stage", choices=tuple(STAGES), default="quick"
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIGS)
    )
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"), default="val"
    )
    parser.add_argument(
        "--allow-pair-leakage",
        action="store_true",
        help="Hanya untuk reproduksi; jangan dipakai untuk klaim utama.",
    )
    args = parser.parse_args()
    run_usk_screening(
        args.raw_root,
        args.data_root,
        args.output_root,
        args.seeds,
        args.models or STAGES[args.stage],
        args.evaluation_split,
        args.allow_pair_leakage,
    )


if __name__ == "__main__":
    main()
