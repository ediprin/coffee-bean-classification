from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.reporting.aggregate_ablation import aggregate
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.data.preparation.prepare_paper_protocol import prepare_paper_protocol


MODEL_CONFIGS = {
    "P0": Path("configs/paper/P0_paper_mobilenetv3_gap_ce.yaml"),
    "P1": Path("configs/paper/P1_paper_mobilenetv3_hbp_ce.yaml"),
}


def _complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    if not history_path.is_file() or not (run_dir / "best.pt").is_file():
        return False
    try:
        return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _model_counts(config_path: Path) -> tuple[int, int]:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    if cfg["training"].get("freeze_backbone", False):
        model.encoder.requires_grad_(False)
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


def run_paper_reproduction(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    raw_root: Path | None = None,
) -> None:
    if not (data_root / "source/train").is_dir():
        if raw_root is None:
            raise FileNotFoundError(
                f"Paper-protocol dataset tidak ditemukan: {data_root}. "
                "Berikan --raw-root untuk menyiapkannya."
            )
        prepare_paper_protocol(raw_root, data_root, seed=seeds[0])

    audit_path = data_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit paper-protocol tidak ditemukan: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("augmented_count") != 6853 or audit.get("split_counts", {}).get("test") != 686:
        raise ValueError("Dataset tidak cocok dengan dukungan reproduksi paper 6853/686.")

    print("=== REPRODUKSI ARWATCHANANUKUL ET AL. (LEAKAGE-PRONE) ===")
    print("Paper reference: Accuracy=88.63% Macro-F1=89.04% Test n=686")
    print(
        "Identity leakage: "
        f"{audit['identity_overlap']['originals_crossing_splits']}/"
        f"{audit['identity_overlap']['unique_originals']} original"
    )
    for code, config_path in MODEL_CONFIGS.items():
        cfg = load_config(config_path)
        total, trainable = _model_counts(config_path)
        print(
            f"{code}: head={cfg['model']['head']} epochs={cfg['training']['epochs']} "
            f"lr={cfg['training']['lr']} total={total:,} trainable={trainable:,}"
        )

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = output_root / "reports" / f"{code}_seed{seed}"
            if _complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}")
            else:
                _run([
                    sys.executable, "-u", "-m", "bilinear_lmmd.engine.train",
                    "--config", str(config_path),
                    "--seed", str(seed),
                    "--data-root", str(data_root),
                    "--output-dir", str(run_dir),
                    "--resume",
                ])
            if (report_dir / "metrics.json").is_file():
                print(f"SKIP evaluasi lengkap: {code} seed {seed}")
            else:
                _run([
                    sys.executable, "-u", "-m", "bilinear_lmmd.engine.evaluate_checkpoint",
                    "--checkpoint", str(run_dir / "best.pt"),
                    "--domain", "source",
                    "--split", "test",
                    "--data-root", str(data_root),
                    "--output-dir", str(report_dir),
                ])

    baseline = [
        output_root / "reports" / f"P0_seed{seed}" / "metrics.json"
        for seed in seeds
    ]
    candidate = [
        output_root / "reports" / f"P1_seed{seed}" / "metrics.json"
        for seed in seeds
    ]
    result = aggregate(baseline, candidate)
    destination = output_root / "reports" / "P0_vs_P1_aggregate.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== P0 vs P1: EFEK HBP PADA PROTOKOL PAPER ===")
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
    print("KLAIM: keterbandingan paper saja; hasil bersih M0/M1 tetap bukti utama.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduksi MobileNetV3 paper vs HBP+CE pada split paper"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    run_paper_reproduction(
        args.data_root,
        args.output_root,
        args.seeds,
        raw_root=args.raw_root,
    )


if __name__ == "__main__":
    main()
