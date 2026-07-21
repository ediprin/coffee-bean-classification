from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.loaders import SameClassPairDataset, build_loaders
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    current_git_commit,
    evaluate,
    load_resume_checkpoint,
    resolve_device,
    seed_everything,
)
from bilinear_lmmd.modeling.losses import (
    ConfusionAwareSupervisedContrastiveLoss,
)
from bilinear_lmmd.modeling.models import build_model


class ContrastiveProjectionHead(nn.Module):
    """Training-only projection head; absent from deployment checkpoints."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Dimensi projection head harus lebih besar dari nol.")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return F.normalize(self.layers(embeddings), p=2, dim=1)


def paired_train_loader(base_loader: DataLoader, workers: int) -> DataLoader:
    return DataLoader(
        SameClassPairDataset(base_loader.dataset),
        batch_size=base_loader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
    )


def normalized_soft_confusion(
    labels: Tensor,
    probabilities: Tensor,
    num_classes: int,
) -> Tensor:
    """Build symmetric class confusion from training probabilities only."""

    if labels.ndim != 1 or probabilities.ndim != 2:
        raise ValueError("labels/probabilities memiliki dimensi yang salah.")
    if labels.shape[0] != probabilities.shape[0] or probabilities.shape[1] != num_classes:
        raise ValueError("labels/probabilities tidak sejajar dengan num_classes.")
    matrix = probabilities.new_zeros((num_classes, num_classes))
    counts = probabilities.new_zeros(num_classes)
    matrix.index_add_(0, labels, probabilities)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=probabilities.dtype))
    matrix = matrix / counts.clamp_min(1.0)[:, None]
    matrix.fill_diagonal_(0.0)
    matrix = 0.5 * (matrix + matrix.t())
    maximum = matrix.max()
    if maximum > 0:
        matrix = matrix / maximum
    return matrix.clamp(0.0, 1.0)


def update_confusion_ema(previous: Tensor, current: Tensor, decay: float) -> Tensor:
    if previous.shape != current.shape:
        raise ValueError("Dimensi confusion EMA berbeda.")
    if not 0.0 <= decay < 1.0:
        raise ValueError("confusion_ema_decay harus berada pada [0, 1).")
    updated = current if not torch.any(previous) else decay * previous + (1.0 - decay) * current
    maximum = updated.max()
    return updated / maximum if maximum > 0 else updated


def _checkpoint(
    model: nn.Module,
    projection: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    classes: list[str],
    epoch: int,
    best_f1: float,
    history: list[dict],
    class_confusion: Tensor,
) -> dict:
    return {
        "model": model.state_dict(),
        "projection_head": projection.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "classes": classes,
        "epoch": epoch,
        "best_f1": best_f1,
        "history": history,
        "class_confusion": class_confusion.cpu(),
        "git_commit": current_git_commit(),
    }


def train_pairwise_contrastive(
    config_path: str | Path,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
    data_root_override: str | Path | None = None,
    resume: bool = False,
) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if output_dir_override is not None:
        cfg["training"]["output_dir"] = str(output_dir_override)
    if data_root_override is not None:
        cfg["data"]["root"] = str(data_root_override)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))

    if str(cfg["adaptation"]["method"]).lower() != "source_only":
        raise ValueError("Pairwise contrastive trainer hanya mendukung source_only.")
    if str(cfg["model"]["head"]).lower() != "gap":
        raise ValueError("Ablation CP1/CP2 dikunci pada head GAP.")
    if str(cfg["model"]["classifier"]).lower() != "linear":
        raise ValueError("Ablation CP1/CP2 dikunci pada classifier linear.")

    training_cfg = cfg["training"]
    mode = str(training_cfg.get("pairwise_mode", "standard")).lower()
    if mode not in {"standard", "confusion_aware"}:
        raise ValueError("pairwise_mode harus standard atau confusion_aware.")
    confusion_strength = float(training_cfg.get("confusion_strength", 0.0))
    if mode == "standard" and confusion_strength != 0.0:
        raise ValueError("CP1 standard harus memakai confusion_strength=0.")
    if mode == "confusion_aware" and confusion_strength <= 0.0:
        raise ValueError("CP2 confusion-aware membutuhkan confusion_strength > 0.")

    loaders = build_loaders(cfg["data"], require_target=False)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset berbeda dari model.num_classes.")
    train_loader = paired_train_loader(
        loaders.source_train, workers=int(cfg["data"].get("workers", 4))
    )
    model = build_model(cfg["model"]).to(device)
    embedding_dim = int(model.pool.output_dim)
    projection = ContrastiveProjectionHead(
        embedding_dim,
        int(training_cfg.get("contrastive_hidden_dim", embedding_dim)),
        int(training_cfg.get("contrastive_projection_dim", 128)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [*model.parameters(), *projection.parameters()],
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    classification_loss = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    temperature = float(training_cfg.get("contrastive_temperature", 0.1))
    contrastive_weight = float(training_cfg.get("contrastive_weight", 0.2))
    warmup_epochs = int(training_cfg.get("confusion_warmup_epochs", 5))
    confusion_decay = float(training_cfg.get("confusion_ema_decay", 0.8))
    if contrastive_weight <= 0.0:
        raise ValueError("contrastive_weight harus lebih besar dari nol.")
    if warmup_epochs < 0:
        raise ValueError("confusion_warmup_epochs tidak boleh negatif.")

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0
    num_classes = len(loaders.classes)
    class_confusion = torch.zeros(num_classes, num_classes, device=device)
    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_resume_checkpoint(last_path, device)
        required = {
            "model", "projection_head", "optimizer", "scheduler", "history",
            "best_f1", "epoch", "class_confusion",
        }
        if checkpoint is not None and required.issubset(checkpoint):
            if checkpoint.get("classes") != loaders.classes:
                raise ValueError("Urutan kelas checkpoint berbeda dari dataset.")
            model.load_state_dict(checkpoint["model"])
            projection.load_state_dict(checkpoint["projection_head"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            history = checkpoint["history"]
            best_f1 = float(checkpoint["best_f1"])
            start_epoch = int(checkpoint["epoch"])
            class_confusion = checkpoint["class_confusion"].to(device)
            print(f"RESUME: epoch {start_epoch + 1}/{epochs}", flush=True)

    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})
    print(
        "PAIRWISE CONTRASTIVE TRAINING: "
        f"mode={mode} weight={contrastive_weight} temperature={temperature} "
        f"confusion_strength={confusion_strength} device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        projection.train()
        active_strength = (
            confusion_strength
            if mode == "confusion_aware" and epoch >= warmup_epochs
            else 0.0
        )
        contrastive_loss = ConfusionAwareSupervisedContrastiveLoss(
            temperature=temperature,
            confusion_strength=active_strength,
        )
        running_total = 0.0
        running_ce = 0.0
        running_contrastive = 0.0
        epoch_labels: list[Tensor] = []
        epoch_probabilities: list[Tensor] = []
        progress = tqdm(train_loader, desc=f"PAIR {mode} {epoch + 1}/{epochs}")
        for images, positive_images, labels in progress:
            images = images.to(device)
            positive_images = positive_images.to(device)
            labels = labels.to(device)
            paired_images = torch.cat((images, positive_images), dim=0)
            paired_labels = torch.cat((labels, labels), dim=0)

            optimizer.zero_grad(set_to_none=True)
            output = model(paired_images, labels=paired_labels)
            ce = classification_loss(output.logits, paired_labels)
            projected = projection(output.embedding)
            pair_loss = contrastive_loss(
                projected,
                paired_labels,
                class_confusion if active_strength > 0.0 else None,
            )
            total = ce + contrastive_weight * pair_loss
            total.backward()
            optimizer.step()

            running_total += float(total.detach())
            running_ce += float(ce.detach())
            running_contrastive += float(pair_loss.detach())
            epoch_labels.append(paired_labels.detach().cpu())
            epoch_probabilities.append(output.logits.detach().softmax(1).cpu())
            progress.set_postfix(
                loss=f"{float(total.detach()):.3f}",
                ce=f"{float(ce.detach()):.3f}",
                pair=f"{float(pair_loss.detach()):.3f}",
                hard=f"{active_strength:.2f}",
            )

        current_confusion = normalized_soft_confusion(
            torch.cat(epoch_labels), torch.cat(epoch_probabilities), num_classes
        ).to(device)
        class_confusion = update_confusion_ema(
            class_confusion, current_confusion, confusion_decay
        )
        scheduler.step()
        metrics = evaluate(
            model, loaders.source_val, device, loaders.classes, hard_groups
        )
        batch_count = max(len(train_loader), 1)
        record = {
            "epoch": epoch + 1,
            "loss": running_total / batch_count,
            "ce_loss": running_ce / batch_count,
            "contrastive_loss": running_contrastive / batch_count,
            "active_confusion_strength": active_strength,
            "source": metrics,
            "class_confusion": class_confusion.detach().cpu().tolist(),
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": record["loss"],
                    "ce_loss": record["ce_loss"],
                    "contrastive_loss": record["contrastive_loss"],
                    "active_confusion_strength": active_strength,
                    "source": {key: value for key, value in metrics.items() if key != "per_class"},
                }
            ),
            flush=True,
        )

        is_best = float(metrics["macro_f1"]) > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])
        checkpoint = _checkpoint(
            model,
            projection,
            optimizer,
            scheduler,
            cfg,
            loaders.classes,
            epoch + 1,
            best_f1,
            history,
            class_confusion,
        )
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            # Projection head is training-only. This deploy/evaluation
            # checkpoint remains exactly compatible with the ordinary GAP model.
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "weights": "raw",
                    "training_only_projection_removed": True,
                },
                output_dir / "best.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-GAP with pairwise contrastive objective"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_pairwise_contrastive(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        data_root_override=args.data_root,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
