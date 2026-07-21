from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
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
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.modeling.progressive_multigranularity import jigsaw_generator


def _paired_train_loader(base_loader: DataLoader, workers: int) -> DataLoader:
    return DataLoader(
        SameClassPairDataset(base_loader.dataset),
        batch_size=base_loader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
    )


def progressive_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    descriptor: torch.Tensor,
    pair_batch_size: int | None,
    classification_loss: nn.Module,
    consistency_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    classification = classification_loss(logits, targets)
    if pair_batch_size is None:
        return classification, classification, None
    if descriptor.shape[0] != pair_batch_size * 2:
        raise ValueError("Descriptor pair harus berisi dua batch berukuran sama.")
    consistency = F.mse_loss(
        descriptor[:pair_batch_size], descriptor[pair_batch_size:]
    )
    # PMG-V2 dynamically balances CE and MSE before applying the stage weight.
    balance = classification.detach() / consistency.detach().clamp_min(1.0e-8)
    total = classification + consistency_weight * balance * consistency
    return total, classification, consistency


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    classes: list[str],
    epoch: int,
    best_f1: float,
    history: list[dict],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "classes": classes,
        "epoch": epoch,
        "best_f1": best_f1,
        "history": history,
        "git_commit": current_git_commit(),
    }


def train_progressive(
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

    head = str(cfg["model"]["head"])
    supported = {
        "progressive_multigranularity",
        "progressive_multigranularity_consistency",
    }
    if head not in supported:
        raise ValueError(f"Trainer progressive membutuhkan head {sorted(supported)}.")
    if str(cfg["adaptation"]["method"]).lower() != "source_only":
        raise ValueError("Trainer progressive hanya mendukung source_only.")

    loaders = build_loaders(cfg["data"], require_target=False)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset berbeda dari model.num_classes.")
    category_consistency = head.endswith("_consistency")
    train_loader = (
        _paired_train_loader(
            loaders.source_train,
            workers=int(cfg["data"].get("workers", 4)),
        )
        if category_consistency
        else loaders.source_train
    )

    model = build_model(cfg["model"]).to(device)
    training_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    classification_loss = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    grids = tuple(int(value) for value in training_cfg.get("pmg_grids", (8, 4, 2)))
    weights = tuple(
        float(value)
        for value in training_cfg.get("pmg_consistency_weights", (0.01, 0.05, 0.1))
    )
    if len(grids) != 3 or len(weights) != 3:
        raise ValueError("pmg_grids dan pmg_consistency_weights harus berisi 3 nilai.")
    image_size = int(cfg["data"]["image_size"])
    if any(image_size % grid for grid in grids):
        raise ValueError("Semua PMG grid harus membagi image_size tanpa sisa.")

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0
    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_resume_checkpoint(last_path, device)
        required = {"model", "optimizer", "scheduler", "history", "best_f1", "epoch"}
        if checkpoint is not None and required.issubset(checkpoint):
            if checkpoint.get("classes") != loaders.classes:
                raise ValueError("Urutan kelas checkpoint berbeda dari dataset.")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            history = checkpoint["history"]
            best_f1 = float(checkpoint["best_f1"])
            start_epoch = int(checkpoint["epoch"])
            print(f"RESUME: epoch {start_epoch + 1}/{epochs}", flush=True)

    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})
    print(
        "PROGRESSIVE TRAINING: "
        f"head={head} grids={list(grids)} consistency={category_consistency} "
        f"device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = {"fine": 0.0, "medium": 0.0, "coarse": 0.0, "concat": 0.0}
        consistency_sums = [0.0, 0.0, 0.0]
        progress = tqdm(train_loader, desc=f"PMG epoch {epoch + 1}/{epochs}")
        for batch in progress:
            if category_consistency:
                images, positive_images, labels = batch
                images = images.to(device)
                positive_images = positive_images.to(device)
                labels = labels.to(device)
                pair_batch_size = images.shape[0]
                progressive_images = torch.cat((images, positive_images), dim=0)
                progressive_targets = torch.cat((labels, labels), dim=0)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                pair_batch_size = None
                progressive_images = images
                progressive_targets = labels

            branch_losses = []
            for branch_index, (grid, consistency_weight) in enumerate(
                zip(grids, weights)
            ):
                optimizer.zero_grad(set_to_none=True)
                shuffled = jigsaw_generator(progressive_images, grid)
                branch = model.forward_branch(shuffled, branch_index)
                loss, ce, consistency = progressive_objective(
                    branch.logits,
                    progressive_targets,
                    branch.descriptor,
                    pair_batch_size,
                    classification_loss,
                    consistency_weight,
                )
                loss.backward()
                optimizer.step()
                branch_losses.append(float(loss.detach()))
                sums[model.BRANCH_NAMES[branch_index]] += float(ce.detach())
                if consistency is not None:
                    consistency_sums[branch_index] += float(consistency.detach())

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            concat_loss = 2.0 * classification_loss(
                output.expert_logits["concat"], labels
            )
            concat_loss.backward()
            optimizer.step()
            sums["concat"] += float(concat_loss.detach())
            progress.set_postfix(
                fine=f"{branch_losses[0]:.3f}",
                medium=f"{branch_losses[1]:.3f}",
                coarse=f"{branch_losses[2]:.3f}",
                concat=f"{float(concat_loss.detach()):.3f}",
            )

        scheduler.step()
        metrics = evaluate(
            model,
            loaders.source_val,
            device,
            loaders.classes,
            hard_groups,
        )
        batches = max(len(train_loader), 1)
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_ce": {name: value / batches for name, value in sums.items()},
            "train_consistency": [value / batches for value in consistency_sums],
            "source": metrics,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "macro_f1": metrics["macro_f1"],
                    "hard_f1": metrics["hard_class_f1"],
                    "worst_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )
        current_f1 = float(metrics["macro_f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            atomic_torch_save(
                _checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    loaders.classes,
                    epoch + 1,
                    best_f1,
                    history,
                ),
                output_dir / "best.pt",
            )
        atomic_torch_save(
            _checkpoint(
                model,
                optimizer,
                scheduler,
                cfg,
                loaders.classes,
                epoch + 1,
                best_f1,
                history,
            ),
            last_path,
        )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end EfficientNet progressive multi-granularity training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_progressive(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        data_root_override=args.data_root,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
