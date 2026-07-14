from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .config import load_config
from .data import build_loaders
from .losses import LMMDLoss, MMDLoss
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def adaptation_schedule(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def repeat_loader(loader):
    while True:
        yield from loader


def classification_metrics(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
    hard_groups: dict[str, list[str]],
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    group_scores = {}
    for group_name, members in hard_groups.items():
        unknown = [member for member in members if member not in per_class]
        if unknown:
            raise ValueError(f"Kelas hard-group tidak ditemukan: {unknown}")
        group_scores[group_name] = sum(per_class[name]["f1"] for name in members) / len(
            members
        )
    hard_members = list(dict.fromkeys(name for group in hard_groups.values() for name in group))
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": float(f1.mean()),
        "worst_class_f1": float(f1.min()),
        "hard_class_f1": (
            sum(per_class[name]["f1"] for name in hard_members) / len(hard_members)
            if hard_members
            else None
        ),
        "hard_groups": group_scores,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    hard_groups: dict[str, list[str]],
) -> dict:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    for images, targets in loader:
        output = model(images.to(device))
        predictions.extend(output.logits.argmax(1).cpu().tolist())
        labels.extend(targets.tolist())
    return classification_metrics(labels, predictions, class_names, hard_groups)


def train(
    config_path: str,
    seed_override: int | None = None,
    output_dir_override: str | None = None,
) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = seed_override
    if output_dir_override is not None:
        cfg["training"]["output_dir"] = output_dir_override
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    adaptation_cfg = cfg["adaptation"]
    method = adaptation_cfg["method"].lower()
    loaders = build_loaders(cfg["data"], require_target=method != "source_only")
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError(
            f"Dataset memiliki {len(loaders.classes)} kelas, tetapi model.num_classes="
            f"{cfg['model']['num_classes']}."
        )

    model = build_model(cfg["model"]).to(device)
    training_cfg = cfg["training"]
    if method not in {"source_only", "mmd", "lmmd", "dann"}:
        raise ValueError("adaptation.method harus source_only, mmd, lmmd, atau dann.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training_cfg["epochs"])
    )
    classification_loss = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    mmd_loss = MMDLoss(
        kernel_mul=float(adaptation_cfg["kernel_mul"]),
        kernel_num=int(adaptation_cfg["kernel_num"]),
    )
    lmmd_loss = LMMDLoss(
        num_classes=int(cfg["model"]["num_classes"]),
        kernel_mul=float(adaptation_cfg["kernel_mul"]),
        kernel_num=int(adaptation_cfg["kernel_num"]),
        confidence_threshold=float(adaptation_cfg.get("confidence_threshold", 0.0)),
    )

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    epochs = int(training_cfg["epochs"])
    best_f1 = -1.0
    history = []
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    for epoch in range(epochs):
        model.train()
        factor = adaptation_schedule(
            epoch, epochs, int(adaptation_cfg.get("warmup_epochs", 0))
        )
        adapt_weight = float(adaptation_cfg["weight"]) * factor
        target_batches = (
            repeat_loader(loaders.target_train) if loaders.target_train is not None else None
        )
        running_loss = 0.0

        progress = tqdm(loaders.source_train, desc=f"epoch {epoch + 1}/{epochs}")
        for source_batch in progress:
            source_images, source_labels = (x.to(device) for x in source_batch)
            optimizer.zero_grad(set_to_none=True)

            if method == "source_only":
                source_output = model(source_images)
                loss = classification_loss(source_output.logits, source_labels)
            else:
                target_images, _ = next(target_batches)
                target_images = target_images.to(device)
                domain_strength = factor if method == "dann" else None
                source_output = model(source_images, domain_strength=domain_strength)
                target_output = model(target_images, domain_strength=domain_strength)
                cls_loss = classification_loss(source_output.logits, source_labels)

                if method == "mmd":
                    adapt_loss = mmd_loss(source_output.embedding, target_output.embedding)
                elif method == "lmmd":
                    adapt_loss = lmmd_loss(
                        source_output.embedding,
                        target_output.embedding,
                        source_labels,
                        target_output.logits,
                    )
                else:
                    source_domain = torch.zeros(
                        source_images.shape[0], dtype=torch.long, device=device
                    )
                    target_domain = torch.ones(
                        target_images.shape[0], dtype=torch.long, device=device
                    )
                    adapt_loss = F.cross_entropy(
                        source_output.domain_logits, source_domain
                    ) + F.cross_entropy(target_output.domain_logits, target_domain)
                loss = cls_loss + adapt_weight * adapt_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", adapt=f"{adapt_weight:.3f}")

        scheduler.step()
        source_metrics = evaluate(
            model, loaders.source_val, device, loaders.classes, hard_groups
        )
        target_metrics = (
            evaluate(model, loaders.target_val, device, loaders.classes, hard_groups)
            if loaders.target_val is not None
            else None
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(len(loaders.source_train), 1),
            "source": source_metrics,
            "target": target_metrics,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": record["epoch"],
                    "loss": record["loss"],
                    "source": {
                        key: value
                        for key, value in source_metrics.items()
                        if key != "per_class"
                    },
                    "target": (
                        {
                            key: value
                            for key, value in target_metrics.items()
                            if key != "per_class"
                        }
                        if target_metrics is not None
                        else None
                    ),
                }
            )
        )

        checkpoint = {
            "model": model.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch + 1,
            "target_metrics": target_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        # Target labels are evaluation-only in unsupervised domain adaptation.
        # Checkpoint selection must not use target metrics.
        selection_metrics = source_metrics
        if selection_metrics["macro_f1"] > best_f1:
            best_f1 = selection_metrics["macro_f1"]
            torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train backbone + HBP + domain adaptation")
    parser.add_argument("--config", required=True, help="Path ke file YAML eksperimen")
    parser.add_argument("--seed", type=int, help="Override seed dari YAML")
    parser.add_argument("--output-dir", help="Override output_dir dari YAML")
    args = parser.parse_args()
    train(args.config, seed_override=args.seed, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
