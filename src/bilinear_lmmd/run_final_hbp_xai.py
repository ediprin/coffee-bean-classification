from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .run_xai_analysis import METRIC_NAMES
from .xai import (
    analyze_explanation,
    load_checkpoint_model,
    load_eval_foreground_mask,
    load_eval_image,
    render_cam_heatmap_panel,
)


MODELS = ("M0", "M1")
OUTCOMES = ("rescued_by_hbp", "harmed_by_hbp", "both_correct", "both_wrong")


def _read_predictions(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions belum ada: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV kosong: {path}")
        classes = [
            name.removeprefix("prob::")
            for name in reader.fieldnames
            if name.startswith("prob::")
        ]
        rows = {}
        for row in reader:
            key = (row["actual"], Path(row["path"]).name)
            if key in rows:
                raise ValueError(f"Sampel duplikat di {path}: {key}")
            rows[key] = row
    return classes, rows


def pair_final_predictions(
    m0_path: Path, m1_path: Path
) -> tuple[list[str], list[dict[str, str]]]:
    classes_m0, m0 = _read_predictions(m0_path)
    classes_m1, m1 = _read_predictions(m1_path)
    if classes_m0 != classes_m1:
        raise ValueError("Urutan kelas predictions M0 dan M1 berbeda.")
    if m0.keys() != m1.keys():
        raise ValueError("Daftar sampel predictions M0 dan M1 berbeda.")
    paired = []
    for actual, filename in m0:
        m0_correct = m0[(actual, filename)]["correct"] == "1"
        m1_correct = m1[(actual, filename)]["correct"] == "1"
        if not m0_correct and m1_correct:
            outcome = "rescued_by_hbp"
        elif m0_correct and not m1_correct:
            outcome = "harmed_by_hbp"
        elif m0_correct and m1_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        paired.append(
            {
                "actual": actual,
                "filename": filename,
                "outcome": outcome,
                "M0_predicted": m0[(actual, filename)]["predicted"],
                "M1_predicted": m1[(actual, filename)]["predicted"],
            }
        )
    return classes_m0, paired


def select_final_xai_rows(
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


def _safe_name(row: dict[str, str]) -> str:
    digest = hashlib.sha256(
        f"{row['actual']}:{row['filename']}".encode()
    ).hexdigest()[:10]
    stem = "".join(
        character if character.isalnum() else "_"
        for character in Path(row["filename"]).stem
    )
    return f"{stem}_{digest}"


def _aggregate(samples: list[dict]) -> dict:
    models = {}
    for model in MODELS:
        models[model] = {}
        for method in ("layercam", "finer_layercam"):
            models[model][method] = {
                metric: statistics.mean(
                    row["models"][model]["metrics"][method][metric]
                    for row in samples
                )
                if samples
                else None
                for metric in METRIC_NAMES
            }
    delta = {
        method: {
            metric: (
                models["M1"][method][metric] - models["M0"][method][metric]
                if models["M0"][method][metric] is not None
                else None
            )
            for metric in METRIC_NAMES
        }
        for method in ("layercam", "finer_layercam")
    }
    return {"selected_samples": len(samples), "models": models, "delta_M1_vs_M0": delta}


def _write_gallery(paths: list[Path], destination: Path) -> None:
    if not paths:
        return
    opened = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in opened)
    height = sum(image.height for image in opened)
    gallery = Image.new("RGB", (width, height), "white")
    y = 0
    for image in opened:
        gallery.paste(image, (0, y))
        y += image.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(destination)
    for image in opened:
        image.close()


def run_final_hbp_xai(
    data_root: Path,
    experiment_root: Path,
    output_root: Path,
    seeds: list[int],
    samples_per_outcome: int,
    gamma: float,
    top_k: int,
    deletion_fraction: float,
    device: str,
) -> dict:
    protocol = {
        "method": "multilayer LayerCAM and Finer-LayerCAM",
        "comparison": "M0 GAP vs M1 HBP on locked Coffee-17 test",
        "seeds": seeds,
        "samples_per_outcome": samples_per_outcome,
        "selection": "deterministic SHA-256 ranking within each prediction outcome",
        "gamma": gamma,
        "top_k_references": top_k,
        "deletion_fraction": deletion_fraction,
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise FileExistsError("Protokol XAI berbeda; gunakan output-root baru.")
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"{output_root} sudah berisi file tanpa protocol.json.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    all_samples = []
    for seed in seeds:
        checkpoints = {
            model: experiment_root / "outputs" / f"{model}_seed{seed}" / "best.pt"
            for model in MODELS
        }
        predictions = {
            model: experiment_root
            / "reports"
            / f"{model}_seed{seed}"
            / "predictions.csv"
            for model in MODELS
        }
        for path in (*checkpoints.values(), *predictions.values()):
            if not path.is_file():
                raise FileNotFoundError(f"Artefak final belum ada: {path}")

        classes, paired = pair_final_predictions(predictions["M0"], predictions["M1"])
        selected = select_final_xai_rows(paired, samples_per_outcome, seed)
        loaded = {
            model: load_checkpoint_model(path, device_name=device)
            for model, path in checkpoints.items()
        }
        image_sizes = {int(cfg["data"]["image_size"]) for _, cfg, _ in loaded.values()}
        if len(image_sizes) != 1:
            raise ValueError("Resolusi checkpoint M0 dan M1 berbeda.")
        image_size = image_sizes.pop()
        panel_paths = []

        progress = tqdm(selected, desc=f"XAI final seed {seed}", unit="sample")
        for row in progress:
            base = (
                output_root
                / "samples"
                / f"seed{seed}"
                / row["outcome"]
                / _safe_name(row)
            )
            json_path = base.with_suffix(".json")
            panel_path = base.with_suffix(".png")
            panel_paths.append(panel_path)
            if json_path.is_file() and panel_path.is_file():
                all_samples.append(json.loads(json_path.read_text(encoding="utf-8")))
                continue

            image_path = (
                data_root / "source" / "test" / row["actual"] / row["filename"]
            )
            if not image_path.is_file():
                raise FileNotFoundError(f"Gambar test tidak ditemukan: {image_path}")
            image_cpu, display = load_eval_image(image_path, image_size)
            foreground = load_eval_foreground_mask(image_path, image_size)
            actual_index = classes.index(row["actual"])
            explanations = {}
            model_results = {}
            for model_name in MODELS:
                progress.set_postfix_str(f"{row['outcome']} | {model_name}")
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
                predicted = classes[explanation.prediction]
                if predicted != row[f"{model_name}_predicted"]:
                    raise RuntimeError(
                        f"Prediksi XAI berbeda untuk {model_name} {image_path}: "
                        f"{predicted} != {row[f'{model_name}_predicted']}"
                    )
                explanations[model_name] = explanation
                model_results[model_name] = {
                    "predicted": predicted,
                    "actual_probability": float(explanation.probabilities[actual_index]),
                    "references": [classes[index] for index in explanation.references],
                    "metrics": metrics,
                }
            result = {
                "seed": seed,
                "outcome": row["outcome"],
                "actual": row["actual"],
                "filename": row["filename"],
                "image": str(image_path),
                "models": model_results,
            }
            render_cam_heatmap_panel(
                panel_path,
                display,
                foreground,
                classes,
                actual_index,
                row["outcome"],
                f"Coffee-17 test | seed={seed}",
                explanations,
            )
            json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            all_samples.append(result)

        _write_gallery(
            [path for path in panel_paths if path.is_file()],
            output_root / f"gallery_seed{seed}.png",
        )
        del loaded

    summary = {
        "protocol": protocol,
        "scope": "deterministic post-hoc diagnostic sample; not a population estimate",
        "outcome_counts": {
            outcome: sum(row["outcome"] == outcome for row in all_samples)
            for outcome in OUTCOMES
        },
        "aggregate": _aggregate(all_samples),
        "samples": all_samples,
    }
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final_hbp_xai_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== FINAL HBP XAI ===")
    print(f"Sampel: {len(all_samples)} | Outcome: {summary['outcome_counts']}")
    for method in ("layercam", "finer_layercam"):
        delta = summary["aggregate"]["delta_M1_vs_M0"][method]
        print(
            f"{method}: Delta foreground={delta['foreground_mass']:+.2%} "
            f"Delta leakage={delta['background_leakage']:+.2%} "
            f"Delta relative-drop={delta['relative_confidence_drop']:+.4f}"
        )
    print(f"SAVED: {output_root}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LayerCAM/Finer-CAM final M0 GAP vs M1 HBP pada test Coffee-17."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--samples-per-outcome", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--deletion-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples_per_outcome < 1:
        parser.error("--samples-per-outcome minimal 1.")
    run_final_hbp_xai(
        args.data_root,
        args.experiment_root,
        args.output_root,
        list(dict.fromkeys(args.seeds)),
        args.samples_per_outcome,
        args.gamma,
        args.top_k,
        args.deletion_fraction,
        args.device,
    )


if __name__ == "__main__":
    main()
