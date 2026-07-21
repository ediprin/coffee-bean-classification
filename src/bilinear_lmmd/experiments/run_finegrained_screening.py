from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.reporting.aggregate_ablation import aggregate
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model


MODEL_CONFIGS = {
    "M0": Path("configs/coffee17/M0_mobilenetv3_gap_source.yaml"),
    "H0": Path("configs/coffee17/H0_mobilenetv3_gap_hierarchical_source.yaml"),
    "H0f": Path("configs/coffee17/H0f_mobilenetv3_gap_full_family_hierarchical_source.yaml"),
    "M1": Path("configs/coffee17/M1_mobilenetv3_hbp_source.yaml"),
    "O0": Path("configs/coffee17/O0_mobilenetv3_gap_object_crop_source.yaml"),
    "O1": Path("configs/coffee17/O1_mobilenetv3_hbp_object_crop_source.yaml"),
    "M1e": Path("configs/coffee17/M1e_mobilenetv3_hbp_ema_source.yaml"),
    "H1": Path("configs/coffee17/H1_mobilenetv3_hbp_hierarchical_source.yaml"),
    "S0": Path("configs/coffee17/S0_mobilenetv3_sppf_attention_gap_source.yaml"),
    "S1": Path("configs/coffee17/S1_mobilenetv3_sppf_attention_hbp_source.yaml"),
    "C1": Path("configs/coffee17/C1_mobilenetv3_capacity_residual_hbp_source.yaml"),
    "M1s": Path("configs/coffee17/M1s_mobilenetv3_sp_hbp_source.yaml"),
    "E1": Path("configs/coffee17/E1_mobilenetv3_hbp_local_moe_source.yaml"),
    "A2": Path("configs/coffee17/A2_mobilenetv3_gap_224_arcface_source.yaml"),
    "A3": Path("configs/coffee17/A3_mobilenetv3_hbp_224_arcface_source.yaml"),
    "F0": Path("configs/coffee17/F0_mobilenetv3_gap_320_ce_source.yaml"),
    "F1": Path("configs/coffee17/F1_mobilenetv3_hbp_320_ce_source.yaml"),
    "F2": Path("configs/coffee17/F2_mobilenetv3_gap_320_arcface_source.yaml"),
    "F3": Path("configs/coffee17/F3_mobilenetv3_hbp_320_arcface_source.yaml"),
}

STAGE_MODELS = {
    "spatial": ["M1", "M1s"],
    "object_crop": ["M0", "M1", "O0", "O1"],
    "hierarchy": ["M1", "H1"],
    "gap_hierarchy": ["M0", "H0"],
    "gap_hierarchy_full": ["M0", "H0f"],
    "gap_hierarchy_mapping": ["M0", "H0", "H0f"],
    "hierarchy_factorial": ["M0", "H0", "M1", "H1"],
    "sppf": ["M1", "S1"],
    "sppf_factorial": ["M0", "M1", "S0", "S1"],
    "sppf_control": ["M1", "C1", "S1"],
    "moe": ["M1", "E1"],
    "resolution": ["M1", "F1"],
    "arcface224": ["M0", "M1", "A2", "A3"],
    "ablation": ["F0", "F1", "F2", "F3"],
    "all": ["M0", "H0", "H0f", "M1", "O0", "O1", "H1", "C1", "S0", "S1", "M1s", "E1", "A2", "A3", "F0", "F1", "F2", "F3"],
}

COMPARISONS = (
    ("M0", "M1", "efek HBP pada 224"),
    ("M0", "O0", "efek object-centric crop pada GAP"),
    ("M1", "O1", "efek object-centric crop pada HBP"),
    ("O0", "O1", "efek HBP setelah object-centric crop"),
    ("M1", "M1e", "efek EMA pada HBP"),
    ("M1", "H1", "efek hierarchical coarse-to-fine supervision"),
    ("M0", "H0", "efek auxiliary hierarchy pada GAP tanpa HBP"),
    ("M0", "H0f", "efek full-family auxiliary hierarchy pada GAP"),
    ("H0", "H0f", "efek mapping singleton-pair menjadi tujuh keluarga penuh"),
    ("H0", "H1", "efek HBP ketika auxiliary hierarchy aktif"),
    ("M1", "S1", "efek SPPF-Attention sebelum HBP"),
    ("M0", "S0", "efek SPPF-Attention sebelum GAP"),
    ("S0", "S1", "efek HBP ketika memakai SPPF-Attention"),
    ("M1", "C1", "efek capacity-matched deep residual refinement"),
    ("C1", "S1", "SPPF-Attention vs capacity-matched control"),
    ("M1", "M1s", "efek preservasi grid HBP 7x7 -> 14x14"),
    ("M1", "E1", "efek global-local HBP mixture-of-experts"),
    ("M0", "A2", "efek ArcFace pada GAP 224"),
    ("M1", "A3", "efek ArcFace pada HBP 224"),
    ("A2", "A3", "efek HBP pada 224 + ArcFace"),
    ("M1", "F1", "efek resolusi HBP 224 -> 320"),
    ("F0", "F1", "efek HBP pada 320 + CE"),
    ("F0", "F2", "efek ArcFace pada GAP 320"),
    ("F1", "F3", "efek ArcFace pada HBP 320"),
    ("F2", "F3", "efek HBP pada 320 + ArcFace"),
)


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


def _parameter_count(config_path: Path) -> int:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    return sum(parameter.numel() for parameter in model.parameters())


def _report_root(output_root: Path, evaluation_split: str) -> Path:
    if evaluation_split == "test":
        return output_root / "reports"
    return output_root / f"{evaluation_split}_reports"


def _report_paths(
    output_root: Path,
    model: str,
    seeds: list[int],
    evaluation_split: str,
) -> list[Path]:
    root = _report_root(output_root, evaluation_split)
    return [
        root / f"{model}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]


def _print_comparison(
    baseline: str,
    candidate: str,
    description: str,
    output_root: Path,
    seeds: list[int],
    evaluation_split: str,
) -> None:
    baseline_paths = _report_paths(
        output_root, baseline, seeds, evaluation_split
    )
    candidate_paths = _report_paths(
        output_root, candidate, seeds, evaluation_split
    )
    missing = [
        str(path)
        for path in baseline_paths + candidate_paths
        if not path.is_file()
    ]
    if missing:
        return

    result = aggregate(baseline_paths, candidate_paths)
    destination = (
        _report_root(output_root, evaluation_split)
        / f"{baseline}_vs_{candidate}_aggregate.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {baseline} vs {candidate}: {description} ===")
    for key, label in (
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


def run_finegrained_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    models: list[str],
    evaluation_split: str = "test",
) -> None:
    print("=== RANCANGAN EKSPERIMEN ===")
    for code in models:
        config_path = MODEL_CONFIGS[code]
        cfg = load_config(config_path)
        count = _parameter_count(config_path)
        print(
            f"{code}: image={cfg['data']['image_size']} "
            f"object_crop={cfg['data'].get('object_crop', False)} "
            f"head={cfg['model']['head']} "
            f"classifier={cfg['model'].get('classifier', 'linear')} "
            f"params={count:,}"
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

            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {label}", flush=True)
            else:
                print(f"EVALUATE: {label}", flush=True)
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.engine.evaluate_checkpoint",
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
    print(
        f"\nPASS: screening fine-grained {', '.join(models)} selesai.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screening fine-grained HBP, MoE, resolusi, dan ArcFace"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/coffee_clean/folds/fold_1"),
        help="Root satu split yang berisi source/train, source/val, source/test.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="test",
        help="Gunakan val untuk screening; test hanya setelah kandidat dikunci.",
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_MODELS),
        default="spatial",
        help=(
            "spatial=M1/M1s, hierarchy=M1/H1, "
            "gap_hierarchy=M0/H0, hierarchy_factorial=M0/H0/M1/H1, "
            "gap_hierarchy_full=M0/H0f, gap_hierarchy_mapping=M0/H0/H0f, "
            "sppf=M1/S1, "
            "object_crop=M0/M1/O0/O1, "
            "sppf_factorial=M0/M1/S0/S1, "
            "sppf_control=M1/C1/S1, "
            "moe=M1/E1, resolution=M1/F1, "
            "arcface224=M0/M1/A2/A3, ablation=F0-F3, all=semuanya."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        help="Override daftar model dari --stage.",
    )
    args = parser.parse_args()
    models = args.models or STAGE_MODELS[args.stage]
    run_finegrained_screening(
        args.data_root,
        args.output_root,
        args.seeds,
        models,
        args.evaluation_split,
    )


if __name__ == "__main__":
    main()
