from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from bilinear_lmmd.analysis.open_set import (
    fit_class_prototypes,
    fit_openmax,
    fit_vim,
    open_set_metrics,
    openmax_knownness,
    prototype_knownness,
    standard_knownness_scores,
    threshold_from_known_validation,
    vim_knownness,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.loaders import build_image_transform
from bilinear_lmmd.data.preparation.prepare_osr_splits import prepare_osr_splits
from bilinear_lmmd.engine.train import resolve_device
from bilinear_lmmd.modeling.models import build_model


AGGREGATE_METRICS = (
    "known_macro_f1",
    "oscr",
    "macro_oscr_research",
    "auroc",
    "aupr_in",
    "aupr_out",
    "fpr95",
    "unknown_rejection",
)


@dataclass(frozen=True)
class FeatureBundle:
    class_names: list[str]
    labels: np.ndarray
    logits: np.ndarray
    embeddings: np.ndarray
    paths: list[str]

    @property
    def predictions(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    def subset(self, indices: np.ndarray) -> "FeatureBundle":
        return FeatureBundle(
            class_names=self.class_names,
            labels=self.labels[indices],
            logits=self.logits[indices],
            embeddings=self.embeddings[indices],
            paths=[self.paths[int(index)] for index in indices],
        )


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _write_split_config(
    base_config: Path,
    split_root: Path,
    output_dir: Path,
    destination: Path,
) -> None:
    cfg = load_config(base_config)
    cfg["data"]["root"] = str(split_root)
    cfg["model"]["num_classes"] = 14
    cfg["training"]["output_dir"] = str(output_dir)
    cfg["evaluation"]["hard_groups"] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def _make_dataset(root: Path, cfg: dict) -> ImageFolder:
    data_cfg = cfg["data"]
    return ImageFolder(
        root,
        transform=build_image_transform(
            image_size=int(data_cfg["image_size"]),
            train=False,
            rotation_angles=[0],
            object_crop=bool(data_cfg.get("object_crop", False)),
            object_crop_margin=float(data_cfg.get("object_crop_margin", 0.10)),
            augmentation_mode=str(data_cfg.get("augmentation_mode", "standard")),
        ),
    )


@torch.no_grad()
def _collect(
    model: torch.nn.Module,
    dataset: ImageFolder,
    cfg: dict,
    description: str,
) -> FeatureBundle:
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("workers", 4)),
        pin_memory=True,
    )
    labels: list[int] = []
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    model.eval()
    for images, targets in tqdm(loader, desc=description, leave=False):
        output = model(images.to(device))
        labels.extend(targets.tolist())
        logits.append(output.logits.cpu().numpy())
        embeddings.append(output.embedding.cpu().numpy())
    return FeatureBundle(
        class_names=list(dataset.classes),
        labels=np.asarray(labels, dtype=np.int64),
        logits=np.concatenate(logits),
        embeddings=np.concatenate(embeddings),
        paths=[str(Path(path).resolve()) for path, _ in dataset.samples],
    )


def _balanced_indices(
    bundle: FeatureBundle,
    tier_root: Path,
    manifest_rows: list[dict],
) -> np.ndarray:
    by_path = {path: index for index, path in enumerate(bundle.paths)}
    expected = [str((tier_root / row["path"]).resolve()) for row in manifest_rows]
    missing = [path for path in expected if path not in by_path]
    if missing:
        raise FileNotFoundError(f"Balanced manifest tidak cocok: {missing[:3]}")
    return np.asarray([by_path[path] for path in expected], dtype=np.int64)


def _score_sets(
    train: FeatureBundle,
    validation: FeatureBundle,
    known_test: FeatureBundle,
    unknown_test: FeatureBundle,
    protocol: dict,
    classifier_weight: np.ndarray | None = None,
    classifier_bias: np.ndarray | None = None,
    vim_principal_dimension: int | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    temperature = float(protocol["scores"]["energy_temperature"])
    scores = {
        "validation": standard_knownness_scores(validation.logits, temperature),
        "known_test": standard_knownness_scores(known_test.logits, temperature),
        "unknown_test": standard_knownness_scores(unknown_test.logits, temperature),
    }
    prototypes = fit_class_prototypes(
        train.embeddings,
        train.labels,
        train.predictions,
        len(train.class_names),
    )
    for name, bundle in (
        ("validation", validation),
        ("known_test", known_test),
        ("unknown_test", unknown_test),
    ):
        scores[name]["prototype"] = prototype_knownness(
            bundle.embeddings, prototypes
        )

    openmax_models = fit_openmax(
        train.logits,
        train.labels,
        train.predictions,
        len(train.class_names),
        tail_size=int(protocol["scores"]["openmax_tail_size"]),
        distance_metric=str(protocol["scores"]["openmax_distance"]),
    )
    for name, bundle in (
        ("validation", validation),
        ("known_test", known_test),
        ("unknown_test", unknown_test),
    ):
        scores[name]["openmax"] = openmax_knownness(
            bundle.logits,
            openmax_models,
            alpha_rank=int(protocol["scores"]["openmax_alpha_rank"]),
            distance_metric=str(protocol["scores"]["openmax_distance"]),
        )
    fit_metadata = {
        "prototype_training_samples": int(
            np.sum(train.labels == train.predictions)
        ),
        "openmax_tail_count_by_class": {
            train.class_names[index]: model.tail_count
            for index, model in enumerate(openmax_models)
        },
        "openmax_distance": str(protocol["scores"]["openmax_distance"]),
    }
    if (classifier_weight is None) != (classifier_bias is None):
        raise ValueError("Weight dan bias classifier ViM harus diberikan bersama.")
    if classifier_weight is not None and classifier_bias is not None:
        vim_model = fit_vim(
            train.embeddings,
            train.logits,
            classifier_weight,
            classifier_bias,
            principal_dimension=vim_principal_dimension,
        )
        for name, bundle in (
            ("validation", validation),
            ("known_test", known_test),
            ("unknown_test", unknown_test),
        ):
            scores[name]["vim"] = vim_knownness(
                bundle.embeddings,
                bundle.logits,
                vim_model,
            )
        fit_metadata["vim"] = {
            "fit_population": "known_train_only",
            "principal_dimension": vim_model.principal_dimension,
            "feature_dimension": int(train.embeddings.shape[1]),
            "residual_dimension": int(vim_model.residual_basis.shape[1]),
            "alpha": vim_model.alpha,
            "dimension_rule": (
                "official_feature_dimension_heuristic"
                if vim_principal_dimension is None
                else "explicit_pre_registered_value"
            ),
        }
    return scores, fit_metadata


def _metric_views(
    known: FeatureBundle,
    unknown: FeatureBundle,
    score_sets: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    known_indices: np.ndarray | None = None,
    unknown_indices: np.ndarray | None = None,
) -> dict[str, dict]:
    known_indices = (
        np.arange(len(known.labels)) if known_indices is None else known_indices
    )
    unknown_indices = (
        np.arange(len(unknown.labels)) if unknown_indices is None else unknown_indices
    )
    reports = {}
    for method in sorted(score_sets["known_test"]):
        reports[method] = open_set_metrics(
            known.labels[known_indices],
            known.predictions[known_indices],
            score_sets["known_test"][method][known_indices],
            unknown.labels[unknown_indices],
            score_sets["unknown_test"][method][unknown_indices],
            thresholds[method],
            known_class_names=known.class_names,
            unknown_class_names=unknown.class_names,
        )
    return reports


def _write_predictions(
    destination: Path,
    known: FeatureBundle,
    unknown: FeatureBundle,
    score_sets: dict[str, dict[str, np.ndarray]],
) -> None:
    methods = sorted(score_sets["known_test"])
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["population", "path", "actual", "predicted", "correct", *methods]
        )
        for population, bundle, score_key in (
            ("known", known, "known_test"),
            ("unknown", unknown, "unknown_test"),
        ):
            for index, path in enumerate(bundle.paths):
                actual = bundle.class_names[int(bundle.labels[index])]
                predicted = known.class_names[int(bundle.predictions[index])]
                writer.writerow(
                    [
                        population,
                        path,
                        actual,
                        predicted,
                        int(population == "known" and actual == predicted),
                        *[score_sets[score_key][method][index] for method in methods],
                    ]
                )


def evaluate_split(
    checkpoint_path: Path,
    tier_root: Path,
    protocol: dict,
    report_root: Path,
    include_vim: bool = False,
    vim_principal_dimension: int | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["model"]["pretrained"] = False
    device = resolve_device(cfg["device"])
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])

    datasets = {
        "train": _make_dataset(tier_root / "source" / "train", cfg),
        "validation": _make_dataset(tier_root / "source" / "val", cfg),
        "known_test": _make_dataset(tier_root / "source" / "test", cfg),
        "unknown_test": _make_dataset(tier_root / "unknown" / "test", cfg),
    }
    known_classes = datasets["train"].classes
    for name in ("validation", "known_test"):
        if datasets[name].classes != known_classes:
            raise ValueError(f"Pemetaan known class berbeda pada {name}.")
    bundles = {
        name: _collect(model, dataset, cfg, name)
        for name, dataset in datasets.items()
    }
    classifier_weight = None
    classifier_bias = None
    if include_vim:
        if not isinstance(model.classifier, torch.nn.Linear):
            raise TypeError("ViM runner memerlukan classifier linear dengan bias.")
        if model.classifier.bias is None:
            raise TypeError("ViM runner memerlukan bias classifier.")
        classifier_weight = model.classifier.weight.detach().cpu().numpy()
        classifier_bias = model.classifier.bias.detach().cpu().numpy()
    score_sets, fit_metadata = _score_sets(
        bundles["train"],
        bundles["validation"],
        bundles["known_test"],
        bundles["unknown_test"],
        protocol,
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        vim_principal_dimension=vim_principal_dimension,
    )
    acceptance_target = float(protocol["dataset"]["known_acceptance_target"])
    thresholds = {
        method: threshold_from_known_validation(values, acceptance_target)
        for method, values in score_sets["validation"].items()
    }
    full = _metric_views(
        bundles["known_test"], bundles["unknown_test"], score_sets, thresholds
    )
    manifest = json.loads(
        (tier_root / "balanced_test_manifest.json").read_text(encoding="utf-8")
    )
    known_indices = _balanced_indices(
        bundles["known_test"], tier_root, manifest["known"]
    )
    unknown_indices = _balanced_indices(
        bundles["unknown_test"], tier_root, manifest["unknown"]
    )
    balanced = _metric_views(
        bundles["known_test"],
        bundles["unknown_test"],
        score_sets,
        thresholds,
        known_indices,
        unknown_indices,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "known_classes": bundles["known_test"].class_names,
        "unknown_classes": bundles["unknown_test"].class_names,
        "threshold_policy": {
            "source": "known_validation_only",
            "target_acceptance": acceptance_target,
            "thresholds": thresholds,
        },
        "fit_metadata": fit_metadata,
        "primary_balanced": balanced,
        "diagnostic_full": full,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_predictions(
        report_root / "predictions.csv",
        bundles["known_test"],
        bundles["unknown_test"],
        score_sets,
    )
    return report


def run_osr_baselines(
    data_root: Path,
    output_root: Path,
    prepared_root: Path | None,
    protocol_path: Path,
    base_config: Path,
    seed: int,
    resume: bool,
    skip_training: bool,
    artifact_repo: str | None,
    artifact_sync_every: int,
) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    prepared_root = prepared_root or output_root / "prepared"
    prepare_osr_splits(data_root, prepared_root, protocol_path)
    reports = {}
    for tier in ("near", "medium", "far"):
        print(f"\n=== OSR {tier.upper()} | SEED {seed} ===", flush=True)
        tier_root = prepared_root / tier
        run_root = output_root / "outputs" / f"OSR0_{tier}_seed{seed}"
        checkpoint = run_root / "best.pt"
        generated_config = (
            output_root / "resolved_configs" / f"OSR0_{tier}_seed{seed}.yaml"
        )
        _write_split_config(base_config, tier_root, run_root, generated_config)
        if not checkpoint.is_file() and not skip_training:
            command = [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.engine.train",
                "--config",
                str(generated_config),
                "--seed",
                str(seed),
                "--output-dir",
                str(run_root),
            ]
            if resume:
                command.append("--resume")
            if artifact_repo:
                command.extend(
                    [
                        "--artifact-repo",
                        artifact_repo,
                        "--artifact-path",
                        f"osr-v1/OSR0_{tier}_seed{seed}",
                        "--artifact-sync-every",
                        str(artifact_sync_every),
                    ]
                )
            _run(command)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint belum tersedia: {checkpoint}. Jalankan tanpa --skip-training."
            )
        report = evaluate_split(
            checkpoint,
            tier_root,
            protocol,
            output_root / "reports" / f"OSR0_{tier}_seed{seed}",
        )
        reports[tier] = report
        print("PRIMARY BALANCED", flush=True)
        for method, metrics in report["primary_balanced"].items():
            print(
                f"{method:10s} OSCR={metrics['oscr']:.2%} "
                f"macro-OSCR*={metrics['macro_oscr_research']:.2%} "
                f"AUROC={metrics['auroc']:.2%} FPR95={metrics['fpr95']:.2%}",
                flush=True,
            )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "seed": seed,
        "claim": "baseline problem validation; HBP not evaluated",
        "splits": reports,
    }
    summary_path = output_root / "reports" / f"osr_v1_seed{seed}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSAVED: {summary_path}", flush=True)
    return summary


def aggregate_osr_summaries(
    summaries: list[dict],
    output_root: Path,
) -> dict:
    """Aggregate balanced OSR results without selecting on unknown test data."""

    if not summaries:
        raise ValueError("Minimal satu summary OSR diperlukan untuk agregasi.")
    seeds = [int(summary["seed"]) for summary in summaries]
    tiers = ("near", "medium", "far")
    methods = sorted(
        summaries[0]["splits"]["near"]["primary_balanced"]
    )
    aggregate: dict[str, dict] = {}
    csv_rows: list[dict] = []
    for tier in tiers:
        aggregate[tier] = {}
        for method in methods:
            metric_summary = {}
            for metric_name in AGGREGATE_METRICS:
                values = np.asarray(
                    [
                        summary["splits"][tier]["primary_balanced"][method][
                            metric_name
                        ]
                        for summary in summaries
                    ],
                    dtype=np.float64,
                )
                metric_summary[metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "values": {
                        str(seed): float(value)
                        for seed, value in zip(seeds, values, strict=True)
                    },
                }
            aggregate[tier][method] = metric_summary
            csv_rows.append(
                {
                    "tier": tier,
                    "score": method.upper(),
                    **{
                        f"{name}_{stat}": metric_summary[name][stat]
                        for name in AGGREGATE_METRICS
                        for stat in ("mean", "std")
                    },
                }
            )

    result = {
        "protocol_id": summaries[0]["protocol_id"],
        "seeds": seeds,
        "claim": "multi-seed baseline problem validation; HBP not evaluated",
        "splits": aggregate,
    }
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "osr_v1_aggregate.json"
    csv_path = report_root / "osr_v1_aggregate.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n=== AGREGAT BASELINE OSR ===", flush=True)
    for tier in tiers:
        print(f"\n{tier.upper()}", flush=True)
        for method in methods:
            row = aggregate[tier][method]
            print(
                f"{method.upper():10s} "
                f"OSCR={row['oscr']['mean']:.2%}±{row['oscr']['std']:.2%} "
                f"AUROC={row['auroc']['mean']:.2%}±{row['auroc']['std']:.2%} "
                f"FPR95={row['fpr95']['mean']:.2%}±{row['fpr95']['std']:.2%} "
                f"Reject={row['unknown_rejection']['mean']:.2%}"
            )
    print(f"\nSAVED: {json_path}", flush=True)
    print(f"SAVED: {csv_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Coffee17 semantic OSR v1 GAP baselines (no HBP)"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help=(
            "Lokasi dataset OSR hasil preparasi. Gunakan disk lokal Colab agar "
            "ribuan gambar tidak disalin ke Google Drive."
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/osr/coffee17_osr_v1.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/osr/OSR0_efficientnetv2_gap_ce.yaml"),
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        help="Satu seed (kompatibilitas runner lama).",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Satu atau lebih seed; contoh: --seeds 42 123 2026.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--artifact-repo")
    parser.add_argument("--artifact-sync-every", type=int, default=5)
    args = parser.parse_args()
    seeds = args.seeds or [args.seed if args.seed is not None else 123]
    summaries = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"\n{'#' * 72}\nSEED {seed} ({index}/{len(seeds)})\n{'#' * 72}",
            flush=True,
        )
        summaries.append(
            run_osr_baselines(
                args.data_root,
                args.output_root,
                args.prepared_root,
                args.protocol,
                args.config,
                seed,
                args.resume,
                args.skip_training,
                args.artifact_repo,
                args.artifact_sync_every,
            )
        )
    aggregate_osr_summaries(summaries, args.output_root)


if __name__ == "__main__":
    main()
