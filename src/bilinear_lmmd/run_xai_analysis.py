from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

from tqdm import tqdm

from .xai import (
    analyze_explanation,
    load_checkpoint_model,
    load_eval_foreground_mask,
    load_eval_image,
    render_comparison_panel,
)


MODELS = ("M1", "M5w01")
OUTCOMES = ("rescued", "negative_transfer", "both_correct", "both_wrong")
METRIC_NAMES = (
    "foreground_mass",
    "background_leakage",
    "foreground_lift",
    "top20_iou",
    "target_confidence_drop",
    "relative_confidence_drop",
)


def _read_predictions(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions belum ada: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV kosong: {path}")
        classes = [name.removeprefix("prob::") for name in reader.fieldnames if name.startswith("prob::")]
        rows = {}
        for row in reader:
            key = (row["actual"], Path(row["path"]).name)
            if key in rows:
                raise ValueError(f"Sampel duplikat di {path}: {key}")
            rows[key] = row
    return classes, rows


def pair_prediction_rows(
    m1_path: Path, m5_path: Path
) -> tuple[list[str], list[dict[str, str]]]:
    classes_m1, m1 = _read_predictions(m1_path)
    classes_m5, m5 = _read_predictions(m5_path)
    if classes_m1 != classes_m5:
        raise ValueError("Urutan kelas predictions M1 dan M5w01 berbeda.")
    if m1.keys() != m5.keys():
        raise ValueError("Daftar sampel predictions M1 dan M5w01 berbeda.")
    paired = []
    for key in m1:
        m1_correct = m1[key]["correct"] == "1"
        m5_correct = m5[key]["correct"] == "1"
        if not m1_correct and m5_correct:
            outcome = "rescued"
        elif m1_correct and not m5_correct:
            outcome = "negative_transfer"
        elif m1_correct and m5_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        paired.append(
            {
                "actual": key[0],
                "filename": key[1],
                "outcome": outcome,
                "M1_predicted": m1[key]["predicted"],
                "M5w01_predicted": m5[key]["predicted"],
            }
        )
    return classes_m1, paired


def select_rows(
    rows: list[dict[str, str]], samples_per_outcome: int, selection_seed: int
) -> list[dict[str, str]]:
    selected = []
    for outcome in OUTCOMES:
        candidates = [row for row in rows if row["outcome"] == outcome]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{selection_seed}:{outcome}:{row['actual']}:{row['filename']}".encode()
            ).hexdigest()
        )
        selected.extend(candidates[:samples_per_outcome])
    return selected


def _experiment_paths(
    domain: str,
    seed: int,
    illumination_root: Path,
    cross_shift_root: Path,
    evaluation_domain: str,
) -> tuple[dict[str, Path], dict[str, Path]]:
    if domain == "illumination":
        checkpoints = {
            model: illumination_root / "outputs" / f"{model}_seed{seed}" / "best.pt"
            for model in MODELS
        }
        predictions = {
            model: illumination_root / "reports" / f"{model}_seed{seed}" / evaluation_domain / "predictions.csv"
            for model in MODELS
        }
    else:
        checkpoints = {
            "M1": illumination_root / "outputs" / f"M1_seed{seed}" / "best.pt",
            "M5w01": cross_shift_root / "outputs" / domain / f"M5w01_seed{seed}" / "best.pt",
        }
        predictions = {
            model: cross_shift_root / "reports" / domain / f"{model}_seed{seed}" / evaluation_domain / "predictions.csv"
            for model in MODELS
        }
    for path in (*checkpoints.values(), *predictions.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Artefak eksperimen belum ada: {path}")
    return checkpoints, predictions


def _image_paths(
    domain_root: Path,
    domain: str,
    evaluation_domain: str,
    actual: str,
    filename: str,
) -> tuple[Path, Path]:
    image = domain_root / evaluation_domain / "test" / actual / filename
    if not image.is_file():
        raise FileNotFoundError(f"Gambar evaluasi tidak ditemukan: {image}")
    if evaluation_domain == "source":
        return image, image
    suffix = f"__{domain}"
    stem = image.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Nama target tidak mengikuti protokol sintetis: {image.name}")
    original_stem = stem[: -len(suffix)]
    matches = sorted((domain_root / "source" / "test" / actual).glob(f"{original_stem}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Pasangan source harus tepat satu untuk {image}; ditemukan {len(matches)}."
        )
    return image, matches[0]


def _safe_name(row: dict[str, str]) -> str:
    digest = hashlib.sha256(f"{row['actual']}:{row['filename']}".encode()).hexdigest()[:10]
    stem = "".join(character if character.isalnum() else "_" for character in Path(row["filename"]).stem)
    return f"{stem}_{digest}"


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _aggregate_selected(samples: list[dict]) -> dict:
    model_summary = {}
    for model in MODELS:
        model_summary[model] = {}
        for method in ("layercam", "finer_layercam"):
            model_summary[model][method] = {
                metric: _mean(
                    [
                        float(row["models"][model]["metrics"][method][metric])
                        for row in samples
                    ]
                )
                for metric in METRIC_NAMES
            }
    deltas = {
        method: {
            metric: (
                model_summary["M5w01"][method][metric]
                - model_summary["M1"][method][metric]
                if model_summary["M1"][method][metric] is not None
                else None
            )
            for metric in METRIC_NAMES
        }
        for method in ("layercam", "finer_layercam")
    }
    return {
        "selected_samples": len(samples),
        "models": model_summary,
        "delta_M5w01_vs_M1": deltas,
    }


def summarize_xai(output_root: Path, protocol: dict) -> dict:
    sample_files = sorted((output_root / "samples").glob("**/*.json"))
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in sample_files]
    aggregates: dict[str, dict] = {}
    for domain in protocol["domains"]:
        aggregates[domain] = {}
        for evaluation_domain in protocol["evaluation_domains"]:
            subset = [
                row for row in samples
                if row["domain"] == domain and row["evaluation_domain"] == evaluation_domain
            ]
            outcome_counts = {
                outcome: sum(row["outcome"] == outcome for row in subset)
                for outcome in OUTCOMES
            }
            aggregates[domain][evaluation_domain] = {
                **_aggregate_selected(subset),
                "outcome_counts": outcome_counts,
                "by_outcome": {
                    outcome: _aggregate_selected(
                        [row for row in subset if row["outcome"] == outcome]
                    )
                    for outcome in OUTCOMES
                },
            }

    summary = {
        "protocol": protocol,
        "scope": (
            "post-hoc diagnostic on deterministically selected controlled-synthetic "
            "samples; not a population estimate or real-world validation"
        ),
        "completed_samples": len(samples),
        "aggregates": aggregates,
    }
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "xai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (report_dir / "xai_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "domain", "evaluation_domain", "seed", "outcome", "actual", "filename",
            "model", "predicted", "references", "method", *METRIC_NAMES,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            for model in MODELS:
                model_row = row["models"][model]
                for method in ("layercam", "finer_layercam"):
                    writer.writerow(
                        {
                            "domain": row["domain"],
                            "evaluation_domain": row["evaluation_domain"],
                            "seed": row["seed"],
                            "outcome": row["outcome"],
                            "actual": row["actual"],
                            "filename": row["filename"],
                            "model": model,
                            "predicted": model_row["predicted"],
                            "references": " | ".join(model_row["references"]),
                            "method": method,
                            **{metric: model_row["metrics"][method][metric] for metric in METRIC_NAMES},
                        }
                    )
    return summary


def _print_summary(summary: dict) -> None:
    print("\n=== XAI M1 vs M5w01 ===")
    print(f"Sampel diagnostik selesai: {summary['completed_samples']}")
    for domain, evaluations in summary["aggregates"].items():
        for evaluation_domain, row in evaluations.items():
            finer = row["delta_M5w01_vs_M1"]["finer_layercam"]
            print(
                f"{domain:<12} {evaluation_domain:<6} n={row['selected_samples']:>2} | "
                f"Delta foreground={finer['foreground_mass']:+.2%} "
                f"leakage={finer['background_leakage']:+.2%} "
                f"relative-drop={finer['relative_confidence_drop']:+.4f}"
                if finer["foreground_mass"] is not None
                else f"{domain:<12} {evaluation_domain:<6} n=0"
            )


def run_xai_analysis(
    data_root: Path,
    illumination_root: Path,
    cross_shift_root: Path,
    output_root: Path,
    domains: list[str],
    evaluation_domains: list[str],
    seeds: list[int],
    samples_per_outcome: int,
    gamma: float,
    top_k: int,
    deletion_fraction: float,
    device: str,
) -> dict:
    protocol = {
        "method": "multilayer LayerCAM and Finer-CAM class-logit comparison",
        "domains": domains,
        "evaluation_domains": evaluation_domains,
        "seeds": seeds,
        "samples_per_outcome": samples_per_outcome,
        "selection": "deterministic SHA-256 ranking within each outcome",
        "gamma": gamma,
        "top_k_references": top_k,
        "deletion_fraction": deletion_fraction,
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise FileExistsError(
                f"Protokol {protocol_path} berbeda. Gunakan output-root baru agar hasil tidak tercampur."
            )
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"{output_root} sudah berisi file tanpa protocol.json.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for domain in domains:
        domain_root = data_root / domain
        if not domain_root.is_dir():
            raise FileNotFoundError(f"Domain sintetis tidak ditemukan: {domain_root}")
        for evaluation_domain in evaluation_domains:
            for seed in seeds:
                checkpoints, predictions = _experiment_paths(
                    domain, seed, illumination_root, cross_shift_root, evaluation_domain
                )
                classes, paired = pair_prediction_rows(
                    predictions["M1"], predictions["M5w01"]
                )
                selected = select_rows(paired, samples_per_outcome, selection_seed=seed)
                pending = []
                for row in selected:
                    base = (
                        output_root
                        / "samples"
                        / domain
                        / evaluation_domain
                        / f"seed{seed}"
                        / row["outcome"]
                        / _safe_name(row)
                    )
                    if not (
                        base.with_suffix(".json").is_file()
                        and base.with_suffix(".png").is_file()
                    ):
                        pending.append((row, base))
                if not pending:
                    print(
                        f"SKIP XAI lengkap: {domain} {evaluation_domain} seed {seed}",
                        flush=True,
                    )
                    continue

                print(
                    f"\nXAI {domain} | {evaluation_domain} | seed {seed}: "
                    f"{len(pending)}/{len(selected)} sampel pending",
                    flush=True,
                )
                print("  Memuat checkpoint M1 dan M5w01...", flush=True)
                load_started = time.perf_counter()
                loaded = {
                    model: load_checkpoint_model(path, device_name=device)
                    for model, path in checkpoints.items()
                }
                devices = sorted({str(model_device) for _, _, model_device in loaded.values()})
                print(
                    f"  Checkpoint siap dalam {time.perf_counter() - load_started:.1f}s "
                    f"| device={','.join(devices)}",
                    flush=True,
                )
                if devices == ["cpu"]:
                    print(
                        "  WARNING: XAI berjalan di CPU; backward pass per sampel "
                        "akan jauh lebih lambat. Aktifkan GPU Kaggle.",
                        flush=True,
                    )
                image_sizes = {
                    int(cfg["data"]["image_size"])
                    for _, cfg, _ in loaded.values()
                }
                if len(image_sizes) != 1:
                    raise ValueError("Resolusi checkpoint M1 dan M5w01 berbeda.")
                image_size = image_sizes.pop()

                progress = tqdm(
                    pending,
                    desc=f"{domain}/{evaluation_domain}/seed{seed}",
                    unit="sample",
                    dynamic_ncols=True,
                )
                for row, base in progress:
                    sample_started = time.perf_counter()
                    progress.set_postfix_str(
                        f"{row['outcome']} | menyiapkan gambar", refresh=True
                    )
                    image_path, source_path = _image_paths(
                        domain_root,
                        domain,
                        evaluation_domain,
                        row["actual"],
                        row["filename"],
                    )
                    image_cpu, display = load_eval_image(image_path, image_size)
                    foreground = load_eval_foreground_mask(source_path, image_size)
                    actual_index = classes.index(row["actual"])
                    explanations = {}
                    model_results = {}
                    for model_name in MODELS:
                        model_started = time.perf_counter()
                        progress.set_postfix_str(
                            f"{row['outcome']} | {model_name}: CAM+deletion",
                            refresh=True,
                        )
                        model, _, model_device = loaded[model_name]
                        explanation, metrics = analyze_explanation(
                            model,
                            image_cpu.to(model_device),
                            foreground,
                            target=actual_index,
                            top_k=top_k,
                            gamma=gamma,
                            deletion_fraction=deletion_fraction,
                        )
                        observed_prediction = classes[explanation.prediction]
                        expected_prediction = row[f"{model_name}_predicted"]
                        if observed_prediction != expected_prediction:
                            raise RuntimeError(
                                "Prediksi XAI tidak mereproduksi predictions.csv: "
                                f"{model_name} {image_path}, "
                                f"expected={expected_prediction}, "
                                f"observed={observed_prediction}."
                            )
                        explanations[model_name] = explanation
                        model_results[model_name] = {
                            "predicted": observed_prediction,
                            "actual_probability": float(
                                explanation.probabilities[actual_index]
                            ),
                            "references": [
                                classes[value] for value in explanation.references
                            ],
                            "metrics": metrics,
                        }
                        tqdm.write(
                            f"    {model_name} selesai: "
                            f"{time.perf_counter() - model_started:.1f}s"
                        )
                    result = {
                        "domain": domain,
                        "evaluation_domain": evaluation_domain,
                        "seed": seed,
                        "outcome": row["outcome"],
                        "actual": row["actual"],
                        "filename": row["filename"],
                        "image": str(image_path),
                        "foreground_mask_source": str(source_path),
                        "models": model_results,
                    }
                    render_comparison_panel(
                        base.with_suffix(".png"),
                        display,
                        foreground,
                        classes,
                        actual_index,
                        row["outcome"],
                        f"{domain}/{evaluation_domain}/seed{seed}",
                        explanations,
                    )
                    # JSON is the completion marker and is deliberately written last.
                    base.with_suffix(".json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                    progress.set_postfix_str(
                        f"tersimpan | {time.perf_counter() - sample_started:.1f}s",
                        refresh=True,
                    )
                del loaded

    summary = summarize_xai(output_root, protocol)
    _print_summary(summary)
    print(f"\nSAVED: {output_root / 'reports' / 'xai_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAI M1 vs M5w01 lintas synthetic shift"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--illumination-root", type=Path, required=True)
    parser.add_argument("--cross-shift-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--domains", nargs="+", default=["illumination", "sensor", "background"]
    )
    parser.add_argument(
        "--evaluation-domains",
        nargs="+",
        default=["target", "source"],
        choices=["source", "target"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--samples-per-outcome", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--deletion-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples_per_outcome < 1:
        parser.error("--samples-per-outcome minimal 1.")
    run_xai_analysis(
        data_root=args.data_root,
        illumination_root=args.illumination_root,
        cross_shift_root=args.cross_shift_root,
        output_root=args.output_root,
        domains=list(dict.fromkeys(args.domains)),
        evaluation_domains=list(dict.fromkeys(args.evaluation_domains)),
        seeds=list(dict.fromkeys(args.seeds)),
        samples_per_outcome=args.samples_per_outcome,
        gamma=args.gamma,
        top_k=args.top_k,
        deletion_fraction=args.deletion_fraction,
        device=args.device,
    )


if __name__ == "__main__":
    main()
