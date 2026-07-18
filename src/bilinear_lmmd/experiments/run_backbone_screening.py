from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from bilinear_lmmd.reporting.aggregate_ablation import aggregate
from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model


BACKBONES = {
    "MV3": {
        "label": "MobileNetV3-Large",
        "gap": ("M0", Path("configs/coffee17/M0_mobilenetv3_gap_source.yaml")),
        "hierarchical_gap": (
            "BM3P",
            Path("configs/backbones/BM3P_mobilenetv3_hierarchical_gap_source.yaml"),
        ),
        "hbp": ("M1", Path("configs/coffee17/M1_mobilenetv3_hbp_source.yaml")),
        "hbp_linear": (
            "M1L",
            Path("configs/coffee17/M1L_mobilenetv3_hbp_linear_source.yaml"),
        ),
    },
    "MV4": {
        "label": "MobileNetV4-Conv-Medium",
        "gap": ("BV4G", Path("configs/backbones/BV4G_mobilenetv4_gap_source.yaml")),
        "hierarchical_gap": (
            "BV4P",
            Path("configs/backbones/BV4P_mobilenetv4_hierarchical_gap_source.yaml"),
        ),
        "hbp": ("BV4H", Path("configs/backbones/BV4H_mobilenetv4_hbp_source.yaml")),
        "hbp_linear": (
            "BV4L",
            Path("configs/backbones/BV4L_mobilenetv4_hbp_linear_source.yaml"),
        ),
    },
    "EV2": {
        "label": "EfficientNetV2-B0",
        "gap": ("BE2G", Path("configs/backbones/BE2G_efficientnetv2_gap_source.yaml")),
        "hierarchical_gap": (
            "BE2P",
            Path("configs/backbones/BE2P_efficientnetv2_hierarchical_gap_source.yaml"),
        ),
        "hbp": ("BE2H", Path("configs/backbones/BE2H_efficientnetv2_hbp_source.yaml")),
        "hbp_linear": (
            "BE2L",
            Path("configs/backbones/BE2L_efficientnetv2_hbp_linear_source.yaml"),
        ),
    },
    "CV2": {
        "label": "ConvNeXtV2-Atto",
        "gap": ("BC2G", Path("configs/backbones/BC2G_convnextv2_gap_source.yaml")),
        "hierarchical_gap": (
            "BC2P",
            Path("configs/backbones/BC2P_convnextv2_hierarchical_gap_source.yaml"),
        ),
        "hbp": ("BC2H", Path("configs/backbones/BC2H_convnextv2_hbp_source.yaml")),
        "hbp_linear": (
            "BC2L",
            Path("configs/backbones/BC2L_convnextv2_hbp_linear_source.yaml"),
        ),
    },
    "PV2": {
        "label": "PVTv2-B0",
        "gap": ("BP2G", Path("configs/backbones/BP2G_pvtv2_gap_source.yaml")),
        "hierarchical_gap": (
            "BP2P",
            Path("configs/backbones/BP2P_pvtv2_hierarchical_gap_source.yaml"),
        ),
        "hbp": ("BP2H", Path("configs/backbones/BP2H_pvtv2_hbp_source.yaml")),
        "hbp_linear": (
            "BP2L",
            Path("configs/backbones/BP2L_pvtv2_hbp_linear_source.yaml"),
        ),
    },
    "SHV": {
        "label": "SHViT-S1",
        "gap": ("BSHG", Path("configs/backbones/BSHG_shvit_gap_source.yaml")),
        "hierarchical_gap": (
            "BSHP",
            Path("configs/backbones/BSHP_shvit_hierarchical_gap_source.yaml"),
        ),
        "hbp": ("BSHH", Path("configs/backbones/BSHH_shvit_hbp_source.yaml")),
        "hbp_linear": (
            "BSHL",
            Path("configs/backbones/BSHL_shvit_hbp_linear_source.yaml"),
        ),
    },
}

DEFAULT_BACKBONES = ("MV4", "EV2", "CV2", "PV2")

HIERARCHICAL_HEADS = frozenset(("hierarchical_gap", "hbp", "hbp_linear"))
COMPARABLE_HIERARCHY_REDUCTIONS = (4, 16, 32)

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return len(history) >= epochs


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _report_root(output_root: Path, split: str) -> Path:
    return output_root / ("reports" if split == "test" else f"{split}_reports")


def _model_summary(config_path: Path) -> dict[str, object]:
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    feature_info = model.encoder.feature_info
    encoder_parameters = sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    pool_parameters = sum(parameter.numel() for parameter in model.pool.parameters())
    return {
        "backbone": cfg["model"]["backbone"],
        "head": cfg["model"]["head"],
        "out_indices": list(cfg["model"]["out_indices"]),
        "channels": list(feature_info.channels()),
        "reductions": list(feature_info.reduction()),
        "image_size": int(cfg["data"]["image_size"]),
        "embedding_dim": int(model.pool.output_dim),
        "encoder_parameters": encoder_parameters,
        "pool_parameters": pool_parameters,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def assess_hierarchy_compatibility(
    head: str,
    reductions: list[int] | tuple[int, ...],
) -> dict[str, object]:
    """Assess whether a hierarchical head is comparable in this benchmark."""

    actual = tuple(int(value) for value in reductions)
    if head not in HIERARCHICAL_HEADS:
        return {
            "required": False,
            "comparable": None,
            "expected_reductions": list(COMPARABLE_HIERARCHY_REDUCTIONS),
            "actual_reductions": list(actual),
            "reason": "Head tidak memakai tiga level feature hierarchy.",
        }
    comparable = actual == COMPARABLE_HIERARCHY_REDUCTIONS
    reason = (
        "Feature hierarchy cocok dengan protokol [4, 16, 32]."
        if comparable
        else (
            "Feature hierarchy tidak sebanding: protokol membutuhkan "
            f"{list(COMPARABLE_HIERARCHY_REDUCTIONS)}, didapat {list(actual)}."
        )
    )
    return {
        "required": True,
        "comparable": comparable,
        "expected_reductions": list(COMPARABLE_HIERARCHY_REDUCTIONS),
        "actual_reductions": list(actual),
        "reason": reason,
    }


def _audit_selected_models(
    selected: list[tuple[str, str, str, Path]],
    output_root: Path,
    allow_incompatible_hierarchy: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    incompatible: list[dict[str, object]] = []
    print("\n=== AUDIT KOMPATIBILITAS HIERARKI ===")
    for family, head, code, config_path in selected:
        summary = _model_summary(config_path)
        compatibility = assess_hierarchy_compatibility(
            head,
            summary["reductions"],
        )
        row = {
            "code": code,
            "family": family,
            **summary,
            "hierarchy": compatibility,
        }
        rows.append(row)
        status = (
            "N/A"
            if compatibility["comparable"] is None
            else ("PASS" if compatibility["comparable"] else "FAIL")
        )
        print(
            f"{code}: {status} | head={head} "
            f"reductions={summary['reductions']} channels={summary['channels']} "
            f"pool={int(summary['pool_parameters']):,} params"
        )
        if compatibility["comparable"] is False:
            incompatible.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "backbone_compatibility.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"SAVED: {destination}")
    if incompatible and not allow_incompatible_hierarchy:
        codes = ", ".join(str(row["code"]) for row in incompatible)
        raise ValueError(
            "Head hierarkis tidak sebanding untuk: "
            f"{codes}. Pilih backbone lain, jalankan GAP saja, atau gunakan "
            "--allow-incompatible-hierarchy hanya untuk eksplorasi berlabel."
        )
    return rows


def _selected_models(
    backbones: list[str], heads: list[str]
) -> list[tuple[str, str, str, Path]]:
    selected: list[tuple[str, str, str, Path]] = []
    for backbone in backbones:
        spec = BACKBONES[backbone]
        for head in heads:
            code, config_path = spec[head]
            selected.append((backbone, head, code, config_path))
    return selected


def _aggregate_pair(
    backbone: str,
    baseline_head: str,
    candidate_head: str,
    output_root: Path,
    seeds: list[int],
    split: str,
) -> dict | None:
    spec = BACKBONES[backbone]
    baseline_code = spec[baseline_head][0]
    candidate_code = spec[candidate_head][0]
    report_root = _report_root(output_root, split)
    baseline_paths = [
        report_root / f"{baseline_code}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]
    candidate_paths = [
        report_root / f"{candidate_code}_seed{seed}" / "metrics.json"
        for seed in seeds
    ]
    if any(not path.is_file() for path in baseline_paths + candidate_paths):
        return None

    result = aggregate(baseline_paths, candidate_paths)
    destination = (
        report_root / f"{baseline_code}_vs_{candidate_code}_aggregate.json"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"\n=== {spec['label']}: "
        f"{baseline_head.upper()} vs {candidate_head.upper()} ==="
    )
    for key, label in (
        ("macro_f1", "Macro-F1"),
        ("hard_class_f1", "Hard-F1 "),
        ("worst_class_f1", "Worst-F1"),
    ):
        item = result["summary"][key]
        print(
            f"{label}: {item['baseline_mean']:.2%} -> "
            f"{item['candidate_mean']:.2%} "
            f"({item['delta_mean']:+.2%})"
        )
    print(f"SAVED: {destination}")
    return result


def _leaderboard(
    selected: list[tuple[str, str, str, Path]],
    output_root: Path,
    seeds: list[int],
    split: str,
) -> list[dict[str, object]]:
    report_root = _report_root(output_root, split)
    rows: list[dict[str, object]] = []
    for backbone, head, code, config_path in selected:
        paths = [report_root / f"{code}_seed{seed}" / "metrics.json" for seed in seeds]
        if any(not path.is_file() for path in paths):
            continue
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        summary = _model_summary(config_path)
        row: dict[str, object] = {
            "code": code,
            "family": backbone,
            "backbone_label": BACKBONES[backbone]["label"],
            "backbone": summary["backbone"],
            "head": head,
            "parameters": summary["parameters"],
            "seeds": seeds,
        }
        for metric in METRICS:
            values = [float(report[metric]) for report in reports]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)

    rows.sort(key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    json_path = report_root / "backbone_leaderboard.json"
    csv_path = report_root / "backbone_leaderboard.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\n=== LEADERBOARD BACKBONE ===")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}. {row['code']:4s} {row['backbone_label']:27s} "
            f"{str(row['head']).upper():3s} "
            f"Macro={float(row['macro_f1_mean']):.2%} "
            f"Hard={float(row['hard_class_f1_mean']):.2%} "
            f"Worst={float(row['worst_class_f1_mean']):.2%} "
            f"Params={int(row['parameters']):,}"
        )
    print(f"SAVED: {json_path}")
    print(f"SAVED: {csv_path}")
    return rows


def run_backbone_screening(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
    backbones: list[str],
    heads: list[str],
    evaluation_split: str = "val",
    hf_repo: str | None = None,
    hf_sync_every: int = 5,
    allow_incompatible_hierarchy: bool = False,
    audit_only: bool = False,
) -> None:
    if hf_sync_every <= 0:
        raise ValueError("hf_sync_every harus lebih besar dari nol.")
    selected = _selected_models(backbones, heads)
    audit_rows = _audit_selected_models(
        selected,
        output_root,
        allow_incompatible_hierarchy=(allow_incompatible_hierarchy or audit_only),
    )
    if audit_only:
        print("\nPASS: audit selesai tanpa training.", flush=True)
        return
    incompatible_codes = {
        str(row["code"])
        for row in audit_rows
        if row["hierarchy"]["comparable"] is False
    }
    if hf_repo:
        try:
            ensure_artifact_repo(hf_repo, private=True)
        except Exception as exc:
            raise RuntimeError(
                "Repo checkpoint Hugging Face tidak dapat dibuat/diakses. "
                "Pastikan HF_TOKEN memiliki izin write."
            ) from exc
    total = len(selected) * len(seeds)
    print("=== PROTOKOL BACKBONE ===")
    print(f"Dataset : {data_root}")
    print(f"Split   : {evaluation_split}")
    print(f"Seeds   : {seeds}")
    print(f"Runs    : {total}")
    print(f"HF repo : {hf_repo or 'OFF'}")
    if hf_repo:
        print(f"HF sync : setiap {hf_sync_every} epoch")
    print("Klaim   : transfer-learning benchmark; resep pretraining dicatat per config")

    print("\n=== RANCANGAN MODEL ===")
    for _, _, code, config_path in selected:
        summary = _model_summary(config_path)
        print(
            f"{code}: backbone={summary['backbone']} head={summary['head']} "
            f"out={summary['out_indices']} reductions={summary['reductions']} "
            f"params={int(summary['parameters']):,}"
        )

    progress = 0
    for _, _, code, config_path in selected:
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            progress += 1
            label = f"{code} | seed {seed}"
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            report_dir = _report_root(output_root, evaluation_split) / f"{code}_seed{seed}"
            artifact_run_path = f"outputs/{code}_seed{seed}"
            artifact_report_path = f"{_report_root(Path('.'), evaluation_split).name}/{code}_seed{seed}"
            print(f"\n[{progress}/{total}] {label}", flush=True)
            if hf_repo:
                try:
                    restored = restore_artifacts(
                        hf_repo,
                        artifact_run_path,
                        run_dir,
                        overwrite=False,
                    )
                    restored += restore_artifacts(
                        hf_repo,
                        artifact_report_path,
                        report_dir,
                        filenames=(
                            "metrics.json",
                            "confusion_matrix.csv",
                            "predictions.csv",
                        ),
                        overwrite=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Restore artefak HF gagal untuk {label}."
                    ) from exc
                print(f"HF RESTORE: {len(restored)} file", flush=True)
            if _training_complete(run_dir, epochs):
                print("SKIP training lengkap", flush=True)
            else:
                print("START / RESUME training", flush=True)
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
                        *(
                            [
                                "--artifact-repo",
                                hf_repo,
                                "--artifact-path",
                                artifact_run_path,
                                "--artifact-sync-every",
                                str(hf_sync_every),
                            ]
                            if hf_repo
                            else []
                        ),
                    ]
                )

            if (report_dir / "metrics.json").is_file():
                print("SKIP evaluasi lengkap", flush=True)
            else:
                print("EVALUATE", flush=True)
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
                if hf_repo:
                    report_files = [
                        path.name for path in report_dir.iterdir() if path.is_file()
                    ]
                    sync_artifacts(
                        hf_repo,
                        artifact_report_path,
                        report_dir,
                        filenames=report_files,
                        commit_message=f"Evaluation {label} on {evaluation_split}",
                    )

    for baseline_head, candidate_head in (
        ("gap", "hbp"),
        ("gap", "hierarchical_gap"),
        ("hierarchical_gap", "hbp"),
        ("hbp", "hbp_linear"),
        ("gap", "hbp_linear"),
    ):
        if {baseline_head, candidate_head}.issubset(heads):
            for backbone in backbones:
                baseline_code = BACKBONES[backbone][baseline_head][0]
                candidate_code = BACKBONES[backbone][candidate_head][0]
                if {baseline_code, candidate_code} & incompatible_codes:
                    print(
                        f"SKIP agregasi {baseline_code} vs {candidate_code}: "
                        "feature hierarchy tidak sebanding."
                    )
                    continue
                _aggregate_pair(
                    backbone,
                    baseline_head,
                    candidate_head,
                    output_root,
                    seeds,
                    evaluation_split,
                )
    comparable_selected = [
        item for item in selected if item[2] not in incompatible_codes
    ]
    _leaderboard(comparable_selected, output_root, seeds, evaluation_split)
    if hf_repo:
        report_root = _report_root(output_root, evaluation_split)
        summary_files = [
            path.name for path in report_root.iterdir() if path.is_file()
        ]
        if summary_files:
            sync_artifacts(
                hf_repo,
                _report_root(Path("."), evaluation_split).name,
                report_root,
                filenames=summary_files,
                commit_message=f"Aggregate backbone benchmark on {evaluation_split}",
            )
    print("\nPASS: benchmark backbone selesai.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark terkontrol GAP/HBP lintas CNN, Transformer, dan hybrid"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root split yang berisi source/train, source/val, dan source/test.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=tuple(BACKBONES),
        default=list(DEFAULT_BACKBONES),
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=("gap", "hierarchical_gap", "hbp", "hbp_linear"),
        default=["gap", "hierarchical_gap", "hbp"],
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="val",
        help="Screening harus memakai val; test hanya setelah kandidat dikunci.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Konfirmasi eksplisit bahwa kandidat sudah dikunci sebelum membuka test.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Tulis laporan kompatibilitas backbone tanpa menjalankan training.",
    )
    parser.add_argument(
        "--allow-incompatible-hierarchy",
        action="store_true",
        help=(
            "Izinkan feature reductions selain [4,16,32]. Hasil wajib "
            "dilabeli eksploratif dan tidak boleh dibandingkan langsung."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        help="Repo model Hugging Face private untuk checkpoint lintas runtime.",
    )
    parser.add_argument(
        "--hf-sync-every",
        type=int,
        default=5,
        help="Upload last.pt dan metadata setiap N epoch (default: 5).",
    )
    args = parser.parse_args()
    if args.evaluation_split == "test" and not args.allow_test:
        parser.error("--evaluation-split test membutuhkan --allow-test.")
    run_backbone_screening(
        data_root=args.data_root,
        output_root=args.output_root,
        seeds=args.seeds,
        backbones=args.backbones,
        heads=args.heads,
        evaluation_split=args.evaluation_split,
        hf_repo=args.hf_repo,
        hf_sync_every=args.hf_sync_every,
        allow_incompatible_hierarchy=args.allow_incompatible_hierarchy,
        audit_only=args.audit_only,
    )


if __name__ == "__main__":
    main()
