from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from tqdm.auto import tqdm

from bilinear_lmmd.data.multisource import MultiSourceLoaders, build_multisource_loaders
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.modeling.losses import (
    OntologyMarginalLoss,
    TaxonomyCompatibleContrastiveLoss,
)
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.modeling.ontology import DatasetOntology, load_ontology


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config OMSL harus berupa YAML dictionary.")
    return payload


def _observed_logits(logits: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    mapping = mapping.to(device=logits.device, dtype=torch.bool)
    expanded = logits[:, None, :].masked_fill(~mapping[None, :, :], -torch.inf)
    return torch.logsumexp(expanded, dim=2)


@torch.no_grad()
def _evaluate_dataset(
    model: nn.Module,
    loader,
    dataset_ontology: DatasetOntology,
    device: torch.device,
) -> dict:
    model.eval()
    observed_true: list[int] = []
    observed_pred: list[int] = []
    leaf_true: list[int] = []
    leaf_pred: list[int] = []
    mapping = dataset_ontology.compatibility
    singleton = mapping.sum(dim=1) == 1
    for images, compatibility, _, observed_target in loader:
        output = model(images.to(device))
        canonical_logits = output.logits
        predictions = _observed_logits(canonical_logits, mapping).argmax(dim=1)
        observed_true.extend(observed_target.tolist())
        observed_pred.extend(predictions.cpu().tolist())
        exact = singleton[observed_target]
        if torch.any(exact):
            exact_targets = compatibility[exact].to(dtype=torch.int64).argmax(dim=1)
            leaf_true.extend(exact_targets.tolist())
            leaf_pred.extend(canonical_logits[exact.to(device)].argmax(dim=1).cpu().tolist())
    labels = list(range(len(dataset_ontology.observed_labels)))
    metrics = {
        "samples": len(observed_true),
        "observed_accuracy": float(accuracy_score(observed_true, observed_pred)),
        "observed_balanced_accuracy": float(
            balanced_accuracy_score(observed_true, observed_pred)
        ),
        "observed_macro_f1": float(
            f1_score(observed_true, observed_pred, labels=labels, average="macro", zero_division=0)
        ),
        "observed_labels": list(dataset_ontology.observed_labels),
        "exact_leaf_samples": len(leaf_true),
    }
    if leaf_true:
        metrics.update(
            {
                "exact_leaf_accuracy": float(accuracy_score(leaf_true, leaf_pred)),
                "exact_leaf_macro_f1": float(
                    f1_score(leaf_true, leaf_pred, average="macro", zero_division=0)
                ),
            }
        )
    else:
        metrics.update({"exact_leaf_accuracy": None, "exact_leaf_macro_f1": None})
    return metrics


@torch.no_grad()
def evaluate_omsl(
    model: nn.Module,
    loaders: MultiSourceLoaders,
    device: torch.device,
) -> dict:
    datasets = {
        name: _evaluate_dataset(
            model,
            loaders.validation[name],
            loaders.ontology.datasets[name],
            device,
        )
        for name in loaders.dataset_names
    }
    macro_scores = [row["observed_macro_f1"] for row in datasets.values()]
    balanced_scores = [row["observed_balanced_accuracy"] for row in datasets.values()]
    return {
        "dataset_macro_f1": float(np.mean(macro_scores)),
        "dataset_balanced_accuracy": float(np.mean(balanced_scores)),
        "datasets": datasets,
    }


def train_omsl(
    config_path: str | Path,
    *,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
    resume: bool = False,
) -> dict:
    config_path = Path(config_path)
    cfg = _load_yaml(config_path)
    seed = int(seed_override if seed_override is not None else cfg.get("seed", 42))
    _seed_everything(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    ontology_cfg = cfg.get("ontology", {})
    ontology_path = Path(ontology_cfg["path"])
    if not ontology_path.is_absolute():
        ontology_path = (config_path.parent / ontology_path).resolve()
    allow_provisional = bool(ontology_cfg.get("allow_provisional", False))
    ontology = load_ontology(ontology_path, allow_provisional=allow_provisional)
    if allow_provisional:
        provisional = [
            f"{name}/{label}"
            for name, dataset in ontology.datasets.items()
            for label, status in zip(dataset.observed_labels, dataset.statuses)
            if status == "provisional"
        ]
        if provisional:
            print(
                "WARNING: training memakai mapping provisional: " + ", ".join(provisional),
                flush=True,
            )

    data_cfg = dict(cfg["data"])
    for source in data_cfg.get("sources", []):
        root = Path(source["root"])
        if not root.is_absolute():
            source["root"] = str((config_path.parent / root).resolve())
    loaders = build_multisource_loaders(data_cfg, ontology, seed=seed)

    model_cfg = dict(cfg["model"])
    if int(model_cfg.get("num_classes", len(ontology.canonical_classes))) != len(
        ontology.canonical_classes
    ):
        raise ValueError("model.num_classes harus sama dengan canonical_classes ontology.")
    model_cfg["num_classes"] = len(ontology.canonical_classes)
    if str(model_cfg.get("classifier", "linear")) != "linear":
        raise ValueError("OMSL membutuhkan classifier linear tanpa fine-label ArcFace.")
    model = build_model(model_cfg).to(device)

    training = cfg.get("training", {})
    epochs = int(training.get("epochs", 50))
    output_dir = Path(
        output_dir_override
        if output_dir_override is not None
        else training.get("output_dir", "outputs/omsl")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    marginal_loss = OntologyMarginalLoss()
    contrastive_cfg = cfg.get("taxonomy_contrastive", {})
    contrastive_weight = float(contrastive_cfg.get("weight", 0.0))
    contrastive_loss = TaxonomyCompatibleContrastiveLoss(
        temperature=float(contrastive_cfg.get("temperature", 0.1)),
        nested_positives=bool(contrastive_cfg.get("nested_positives", False)),
    )

    start_epoch = 0
    best_score = -float("inf")
    history: list[dict] = []
    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        history = list(checkpoint.get("history", []))
        print(f"RESUME: epoch {start_epoch + 1}/{epochs}", flush=True)

    print(
        f"OMSL: {len(loaders.dataset_names)} dataset | "
        f"{len(ontology.canonical_classes)} canonical leaf | device={device}",
        flush=True,
    )
    print(f"Train counts: {loaders.train_counts}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_marginal = 0.0
        total_contrastive = 0.0
        sample_count = 0
        progress = tqdm(
            loaders.train,
            desc=f"epoch {epoch + 1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for images, compatibility, _, _ in progress:
            images = images.to(device, non_blocking=True)
            compatibility = compatibility.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            marginal = marginal_loss(output.logits, compatibility)
            contrastive = contrastive_loss(output.embedding, compatibility)
            loss = marginal + contrastive_weight * contrastive
            loss.backward()
            optimizer.step()
            batch = images.shape[0]
            sample_count += batch
            total_loss += float(loss.detach()) * batch
            total_marginal += float(marginal.detach()) * batch
            total_contrastive += float(contrastive.detach()) * batch
            progress.set_postfix(loss=f"{total_loss / sample_count:.4f}")
        scheduler.step()
        metrics = evaluate_omsl(model, loaders, device)
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(sample_count, 1),
            "marginal_loss": total_marginal / max(sample_count, 1),
            "contrastive_loss": total_contrastive / max(sample_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "validation": metrics,
        }
        history.append(row)
        score = metrics["dataset_macro_f1"]
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": max(best_score, score),
            "history": history,
            "canonical_classes": list(ontology.canonical_classes),
            "config": cfg,
        }
        atomic_torch_save(payload, last_path)
        if score > best_score:
            best_score = score
            atomic_torch_save(payload, output_dir / "best.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch {epoch + 1}: loss={row['loss']:.4f} "
            f"dataset-macro={score:.4f} best={best_score:.4f}",
            flush=True,
        )

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final_metrics = evaluate_omsl(model, loaders, device)
    report = {
        "method": "OMSL-TC" if contrastive_weight > 0 else "OMSL",
        "seed": seed,
        "ontology": str(ontology_path),
        "canonical_classes": list(ontology.canonical_classes),
        "contrastive_weight": contrastive_weight,
        "best_epoch": int(best["epoch"]) + 1,
        **final_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ontology-marginalized multi-source classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_omsl(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
