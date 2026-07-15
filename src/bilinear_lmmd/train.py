from __future__ import annotations

import argparse
import copy
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
from .hierarchy import build_parent_hierarchy
from .losses import LMMDLoss, MMDLoss, NonTargetExpertDiversityLoss
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


def atomic_torch_save(payload: dict, destination: Path) -> None:
    """Write a checkpoint completely before replacing the visible file."""

    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_resume_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict | None:
    """Return None for an interrupted/corrupt checkpoint instead of crashing."""

    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:  # torch raises several backend-specific errors
        print(
            f"WARNING: checkpoint resume rusak/tidak lengkap ({checkpoint_path}): "
            f"{type(exc).__name__}: {exc}. Training dimulai ulang.",
            flush=True,
        )
        return None


class ExponentialMovingAverage:
    """EMA of trainable parameters with current model buffers.

    BatchNorm running statistics are copied rather than averaged. This avoids
    applying floating-point EMA arithmetic to integer tracking buffers and
    keeps evaluation statistics aligned with the current training trajectory.
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay harus berada di antara 0 dan 1.")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        self.model.load_state_dict(model.state_dict())
        self.model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.mul_(self.decay).add_(current.detach(), alpha=1.0 - self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current.detach())
        self.model.eval()


def adaptation_schedule(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def repeat_loader(loader):
    while True:
        yield from loader


def supervised_objective(
    output,
    labels: torch.Tensor,
    classification_loss: nn.Module,
    diversity_loss: nn.Module,
    auxiliary_weight: float,
    diversity_weight: float,
    parent_mapping: torch.Tensor | None = None,
    hierarchy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fused_ce = classification_loss(output.logits, labels)
    components = {"fused_ce": fused_ce}
    total = fused_ce
    if output.parent_logits is not None:
        if parent_mapping is None or hierarchy_weight <= 0.0:
            raise ValueError(
                "Model menghasilkan parent logits tetapi hierarchy training belum valid."
            )
        parent_labels = parent_mapping[labels]
        parent_ce = classification_loss(output.parent_logits, parent_labels)
        total = total + hierarchy_weight * parent_ce
        components["parent_ce"] = parent_ce
    if output.expert_logits is not None:
        global_ce = classification_loss(output.expert_logits["hbp_global"], labels)
        local_ce = classification_loss(output.expert_logits["local_gmp"], labels)
        diversity = diversity_loss(
            output.expert_logits["hbp_global"],
            output.expert_logits["local_gmp"],
        )
        total = (
            total
            + auxiliary_weight * (global_ce + local_ce)
            + diversity_weight * diversity
        )
        components.update(
            {
                "hbp_global_ce": global_ce,
                "local_gmp_ce": local_ce,
                "expert_diversity": diversity,
            }
        )
    return total, components


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
    data_root_override: str | None = None,
    resume: bool = False,
) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = seed_override
    if output_dir_override is not None:
        cfg["training"]["output_dir"] = output_dir_override
    if data_root_override is not None:
        cfg["data"]["root"] = data_root_override
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

    hierarchy_cfg = cfg.get("hierarchy", {})
    hierarchy_enabled = bool(hierarchy_cfg.get("enabled", False))
    parent_mapping: torch.Tensor | None = None
    hierarchy_weight = 0.0
    if hierarchy_enabled:
        hierarchy = build_parent_hierarchy(
            loaders.classes, hierarchy_cfg.get("groups", {})
        )
        configured_parents = int(cfg["model"].get("hierarchy_num_parents", 0))
        if configured_parents != len(hierarchy.parent_names):
            raise ValueError(
                "model.hierarchy_num_parents tidak cocok dengan hierarchy.groups: "
                f"{configured_parents} != {len(hierarchy.parent_names)}"
            )
        hierarchy_weight = float(hierarchy_cfg.get("weight", 0.0))
        if hierarchy_weight <= 0.0:
            raise ValueError("hierarchy.weight harus lebih besar dari nol.")
        parent_mapping = torch.tensor(
            hierarchy.fine_to_parent, dtype=torch.long, device=device
        )
        print(
            f"HIERARCHY: {len(loaders.classes)} fine -> "
            f"{len(hierarchy.parent_names)} parent | weight={hierarchy_weight:.3f}",
            flush=True,
        )
    elif int(cfg["model"].get("hierarchy_num_parents", 0)):
        raise ValueError(
            "Model memiliki parent classifier tetapi hierarchy.enabled=false."
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
    ema_decay = float(training_cfg.get("ema_decay", 0.0))
    ema_start_epoch = int(training_cfg.get("ema_start_epoch", 0))
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError("training.ema_decay harus berada pada [0, 1).")
    if ema_start_epoch < 0:
        raise ValueError("training.ema_start_epoch tidak boleh negatif.")
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay > 0.0 else None
    ema_started = False
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
    expert_diversity_loss = NonTargetExpertDiversityLoss()
    expert_aux_weight = float(training_cfg.get("expert_aux_weight", 0.3))
    expert_diversity_weight = float(
        training_cfg.get("expert_diversity_weight", 0.05)
    )

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    epochs = int(training_cfg["epochs"])
    if ema is not None and ema_start_epoch >= epochs:
        raise ValueError("training.ema_start_epoch harus lebih kecil dari epochs.")
    best_f1 = -1.0
    best_raw_f1 = -1.0
    history = []
    start_epoch = 0
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    last_checkpoint = output_dir / "last.pt"
    if resume and last_checkpoint.is_file():
        checkpoint = load_resume_checkpoint(last_checkpoint, device)
        if checkpoint is not None:
            required_state = {"optimizer", "scheduler", "history", "best_f1"}
            if ema is not None:
                required_state.update(
                    {"ema_model", "ema_started", "best_raw_f1"}
                )
            missing_state = sorted(required_state.difference(checkpoint))
            if missing_state:
                print(
                    "Checkpoint lama tidak memiliki state resume lengkap "
                    f"({', '.join(missing_state)}); training dimulai ulang.",
                    flush=True,
                )
            else:
                if checkpoint.get("classes") != loaders.classes:
                    raise ValueError("Urutan kelas checkpoint resume berbeda dari dataset.")
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                if ema is not None:
                    ema_started = bool(checkpoint["ema_started"])
                    if ema_started:
                        if checkpoint["ema_model"] is None:
                            raise ValueError("Checkpoint EMA aktif tanpa ema_model.")
                        ema.model.load_state_dict(checkpoint["ema_model"])
                history = checkpoint["history"]
                best_f1 = float(checkpoint["best_f1"])
                if ema is not None:
                    best_raw_f1 = float(checkpoint["best_raw_f1"])
                start_epoch = int(checkpoint["epoch"])
                print(
                    f"RESUME: melanjutkan dari epoch {start_epoch + 1}/{epochs}",
                    flush=True,
                )

    for epoch in range(start_epoch, epochs):
        model.train()
        factor = adaptation_schedule(
            epoch, epochs, int(adaptation_cfg.get("warmup_epochs", 0))
        )
        adapt_weight = float(adaptation_cfg["weight"]) * factor
        target_batches = (
            repeat_loader(loaders.target_train) if loaders.target_train is not None else None
        )
        running_loss = 0.0
        running_components: dict[str, float] = {}
        gate_sum = torch.zeros(2)
        gate_count = 0

        progress = tqdm(loaders.source_train, desc=f"epoch {epoch + 1}/{epochs}")
        for source_batch in progress:
            source_images, source_labels = (x.to(device) for x in source_batch)
            optimizer.zero_grad(set_to_none=True)

            if method == "source_only":
                source_output = model(source_images, labels=source_labels)
                loss, supervised_components = supervised_objective(
                    source_output,
                    source_labels,
                    classification_loss,
                    expert_diversity_loss,
                    expert_aux_weight,
                    expert_diversity_weight,
                    parent_mapping,
                    hierarchy_weight,
                )
            else:
                target_images, _ = next(target_batches)
                target_images = target_images.to(device)
                domain_strength = factor if method == "dann" else None
                source_output = model(
                    source_images,
                    labels=source_labels,
                    domain_strength=domain_strength,
                )
                target_output = model(target_images, domain_strength=domain_strength)
                cls_loss, supervised_components = supervised_objective(
                    source_output,
                    source_labels,
                    classification_loss,
                    expert_diversity_loss,
                    expert_aux_weight,
                    expert_diversity_weight,
                    parent_mapping,
                    hierarchy_weight,
                )

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
            if ema is not None and epoch >= ema_start_epoch:
                if ema_started:
                    ema.update(model)
                else:
                    ema.copy_from(model)
                    ema_started = True
            running_loss += loss.item()
            for name, value in supervised_components.items():
                running_components[name] = running_components.get(name, 0.0) + value.item()
            postfix = {"loss": f"{loss.item():.4f}", "adapt": f"{adapt_weight:.3f}"}
            if source_output.gate_weights is not None:
                gate_sum += source_output.gate_weights.detach().sum(dim=0).cpu()
                gate_count += source_output.gate_weights.shape[0]
                postfix["gate_hbp"] = f"{(gate_sum[0] / gate_count).item():.3f}"
            progress.set_postfix(**postfix)

        scheduler.step()
        source_metrics_raw = evaluate(
            model, loaders.source_val, device, loaders.classes, hard_groups
        )
        target_metrics_raw = (
            evaluate(model, loaders.target_val, device, loaders.classes, hard_groups)
            if loaders.target_val is not None
            else None
        )
        source_metrics_ema = (
            evaluate(ema.model, loaders.source_val, device, loaders.classes, hard_groups)
            if ema is not None and ema_started
            else None
        )
        target_metrics_ema = (
            evaluate(ema.model, loaders.target_val, device, loaders.classes, hard_groups)
            if ema is not None and ema_started and loaders.target_val is not None
            else None
        )
        source_metrics = source_metrics_ema or source_metrics_raw
        target_metrics = (
            target_metrics_ema
            if source_metrics_ema is not None
            else target_metrics_raw
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(len(loaders.source_train), 1),
            "source": source_metrics,
            "target": target_metrics,
        }
        if ema is not None:
            record["source_raw"] = source_metrics_raw
            record["source_ema"] = source_metrics_ema
            record["target_raw"] = target_metrics_raw
            record["target_ema"] = target_metrics_ema
            record["selection_weights"] = "ema" if source_metrics_ema is not None else "raw"
        if running_components:
            record["loss_components"] = {
                name: value / max(len(loaders.source_train), 1)
                for name, value in running_components.items()
            }
        if gate_count:
            record["gate_mean"] = (gate_sum / gate_count).tolist()
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": record["epoch"],
                    "loss": record["loss"],
                    "loss_components": record.get("loss_components"),
                    "gate_mean": record.get("gate_mean"),
                    "source": {
                        key: value
                        for key, value in source_metrics.items()
                        if key != "per_class"
                    },
                    "source_raw": (
                        {
                            key: value
                            for key, value in source_metrics_raw.items()
                            if key != "per_class"
                        }
                        if ema is not None
                        else None
                    ),
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

        selection_metrics = source_metrics
        selection_ready = ema is None or ema_started
        is_best = selection_ready and selection_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = selection_metrics["macro_f1"]
        raw_is_best = (
            ema is not None and source_metrics_raw["macro_f1"] > best_raw_f1
        )
        if raw_is_best:
            best_raw_f1 = source_metrics_raw["macro_f1"]

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch + 1,
            "target_metrics": target_metrics,
            "history": history,
            "best_f1": best_f1,
        }
        if ema is not None:
            checkpoint["ema_model"] = ema.model.state_dict() if ema_started else None
            checkpoint["ema_started"] = ema_started
            checkpoint["best_raw_f1"] = best_raw_f1
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        # Target labels are evaluation-only in unsupervised domain adaptation.
        # Checkpoint selection must not use target metrics.
        if is_best:
            best_checkpoint = {
                key: checkpoint[key]
                for key in (
                    "model",
                    "classes",
                    "config",
                    "epoch",
                    "target_metrics",
                    "best_f1",
                )
            }
            if ema is not None and ema_started:
                best_checkpoint["model"] = ema.model.state_dict()
                best_checkpoint["weights"] = "ema"
                best_checkpoint["ema_decay"] = ema_decay
                best_checkpoint["ema_start_epoch"] = ema_start_epoch
            else:
                best_checkpoint["weights"] = "raw"
            atomic_torch_save(best_checkpoint, output_dir / "best.pt")
        if raw_is_best:
            raw_best_checkpoint = {
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "epoch": epoch + 1,
                "target_metrics": target_metrics_raw,
                "best_f1": best_raw_f1,
                "weights": "raw",
            }
            atomic_torch_save(raw_best_checkpoint, output_dir / "best_raw.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train backbone + HBP + domain adaptation")
    parser.add_argument("--config", required=True, help="Path ke file YAML eksperimen")
    parser.add_argument("--seed", type=int, help="Override seed dari YAML")
    parser.add_argument("--output-dir", help="Override output_dir dari YAML")
    parser.add_argument("--data-root", help="Override data.root dari YAML")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Lanjutkan dari last.pt jika state optimizer lengkap tersedia.",
    )
    args = parser.parse_args()
    train(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        data_root_override=args.data_root,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
