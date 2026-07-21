from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
from datetime import datetime, timezone
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

from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    normalize_remote_path,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.loaders import build_loaders
from bilinear_lmmd.modeling.hierarchy import build_parent_hierarchy
from bilinear_lmmd.modeling.losses import (
    BalancedSoftmaxLoss,
    LMMDLoss,
    MMDLoss,
    NonTargetExpertDiversityLoss,
)
from bilinear_lmmd.modeling.models import build_model


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


def current_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
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


def branch_gradient_cosine(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: list[nn.Parameter],
) -> float | None:
    """Measure gradient agreement without modifying accumulated gradients."""

    first = torch.autograd.grad(
        first_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second = torch.autograd.grad(
        second_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    dot = torch.zeros((), device=first_loss.device)
    first_norm = torch.zeros((), device=first_loss.device)
    second_norm = torch.zeros((), device=first_loss.device)
    for first_gradient, second_gradient in zip(first, second):
        if first_gradient is None or second_gradient is None:
            continue
        dot = dot + (first_gradient * second_gradient).sum()
        first_norm = first_norm + first_gradient.square().sum()
        second_norm = second_norm + second_gradient.square().sum()
    denominator = first_norm.sqrt() * second_norm.sqrt()
    if denominator.item() == 0.0:
        return None
    return float((dot / denominator).detach().cpu())


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
    open_set_loss = getattr(output, "open_set_loss", None)
    if open_set_loss is not None:
        total = total + open_set_loss
        components["open_set_regularization"] = open_set_loss
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
        expert_names = tuple(output.expert_logits)
        if expert_names == ("hbp_global", "local_gmp"):
            first_name, second_name = expert_names
        elif expert_names == ("gap", "hbp"):
            first_name, second_name = expert_names
        else:
            raise ValueError(f"Pasangan expert tidak dikenali: {expert_names}")
        first_ce = classification_loss(output.expert_logits[first_name], labels)
        second_ce = classification_loss(output.expert_logits[second_name], labels)
        diversity = diversity_loss(
            output.expert_logits[first_name],
            output.expert_logits[second_name],
        )
        total = (
            total
            + auxiliary_weight * (first_ce + second_ce)
            + diversity_weight * diversity
        )
        components.update(
            {
                f"{first_name}_ce": first_ce,
                f"{second_name}_ce": second_ce,
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
    prediction_head: str = "fused",
) -> dict:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    for images, targets in loader:
        output = model(images.to(device))
        if prediction_head == "fused":
            logits = output.logits
        else:
            if output.expert_logits is None or prediction_head not in output.expert_logits:
                available = tuple(output.expert_logits or {})
                raise ValueError(
                    f"prediction_head={prediction_head!r} tidak tersedia; "
                    f"expert={available}."
                )
            logits = output.expert_logits[prediction_head]
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(targets.tolist())
    return classification_metrics(labels, predictions, class_names, hard_groups)


def train(
    config_path: str,
    seed_override: int | None = None,
    output_dir_override: str | None = None,
    data_root_override: str | None = None,
    resume: bool = False,
    artifact_repo: str | None = None,
    artifact_path: str | None = None,
    artifact_sync_every: int = 5,
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
    freeze_backbone = bool(training_cfg.get("freeze_backbone", False))
    if freeze_backbone:
        model.encoder.requires_grad_(False)
        print("TRANSFER LEARNING: backbone frozen; head saja dilatih.", flush=True)
    if method not in {"source_only", "mmd", "lmmd", "dann"}:
        raise ValueError("adaptation.method harus source_only, mmd, lmmd, atau dann.")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    slow_start_epochs = int(training_cfg.get("slow_start_lr_epochs", 0))
    slow_start_factor = float(training_cfg.get("slow_start_lr_factor", 0.1))
    if slow_start_epochs < 0:
        raise ValueError("training.slow_start_lr_epochs tidak boleh negatif.")
    if not 0.0 < slow_start_factor <= 1.0:
        raise ValueError("training.slow_start_lr_factor harus berada di (0, 1].")
    slow_start_parameters = tuple(
        parameter
        for parameter in getattr(model, "slow_start_parameters", lambda: ())()
        if parameter.requires_grad
    )
    if slow_start_epochs and slow_start_parameters:
        slow_ids = {id(parameter) for parameter in slow_start_parameters}
        regular_parameters = [
            parameter
            for parameter in trainable_parameters
            if id(parameter) not in slow_ids
        ]
        optimizer_parameters = [
            {"params": regular_parameters, "group_name": "regular"},
            {"params": slow_start_parameters, "group_name": "slow_start"},
        ]
        print(
            "SLOW START: FB layer lr "
            f"{slow_start_factor:.2f} -> 1.00 selama "
            f"{slow_start_epochs} epoch.",
            flush=True,
        )
    else:
        optimizer_parameters = trainable_parameters
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler_name = str(training_cfg.get("scheduler", "cosine")).lower()
    if slow_start_epochs and slow_start_parameters:
        epochs = int(training_cfg["epochs"])

        def regular_schedule(epoch: int) -> float:
            if scheduler_name == "constant":
                return 1.0
            return 0.5 * (1.0 + math.cos(math.pi * epoch / max(epochs, 1)))

        def slow_schedule(epoch: int) -> float:
            progress = min(max(epoch, 0) / slow_start_epochs, 1.0)
            warmup = slow_start_factor + (1.0 - slow_start_factor) * progress
            return warmup * regular_schedule(epoch)

        if scheduler_name not in {"cosine", "constant"}:
            raise ValueError("training.scheduler harus 'cosine' atau 'constant'.")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[regular_schedule, slow_schedule],
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training_cfg["epochs"])
        )
    elif scheduler_name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
    else:
        raise ValueError("training.scheduler harus 'cosine' atau 'constant'.")
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    classification_loss_name = str(
        training_cfg.get("classification_loss", "cross_entropy")
    ).lower()
    if classification_loss_name == "cross_entropy":
        classification_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
    elif classification_loss_name == "balanced_softmax":
        train_targets = torch.as_tensor(
            loaders.source_train.dataset.targets, dtype=torch.long
        )
        class_counts = torch.bincount(
            train_targets, minlength=len(loaders.classes)
        )
        classification_loss = BalancedSoftmaxLoss(
            class_counts,
            label_smoothing=label_smoothing,
        ).to(device)
        training_cfg["resolved_class_counts"] = {
            name: int(class_counts[index])
            for index, name in enumerate(loaders.classes)
        }
        print(
            "CLASSIFICATION LOSS: Balanced Softmax | "
            f"counts={training_cfg['resolved_class_counts']}",
            flush=True,
        )
    else:
        raise ValueError(
            "training.classification_loss harus 'cross_entropy' atau "
            "'balanced_softmax'."
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
    resolved_artifact_path: str | None = None
    if artifact_repo:
        if artifact_sync_every <= 0:
            raise ValueError("artifact_sync_every harus lebih besar dari nol.")
        resolved_artifact_path = normalize_remote_path(
            artifact_path or f"outputs/{output_dir.name}"
        )
        try:
            ensure_artifact_repo(artifact_repo, private=True)
            restored = (
                restore_artifacts(
                    artifact_repo,
                    resolved_artifact_path,
                    output_dir,
                    overwrite=False,
                )
                if resume
                else []
            )
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face artifact store tidak dapat diakses. "
                "Pastikan HF_TOKEN memiliki izin write dan repo_id benar."
            ) from exc
        print(
            f"HF ARTIFACT: {artifact_repo}/{resolved_artifact_path} | "
            f"restored={len(restored)} | sync_every={artifact_sync_every} epoch",
            flush=True,
        )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    manifest_path = output_dir / "artifact_manifest.json"
    previous_manifest = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_manifest = {}
    manifest = {
        "schema_version": 1,
        "created_at_utc": previous_manifest.get(
            "created_at_utc", datetime.now(timezone.utc).isoformat()
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "config_path": str(config_path),
        "seed": int(cfg["seed"]),
        "data_root": str(cfg["data"]["root"]),
        "artifact_repo": artifact_repo,
        "artifact_path": resolved_artifact_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    epochs = int(training_cfg["epochs"])
    if ema is not None and ema_start_epoch >= epochs:
        raise ValueError("training.ema_start_epoch harus lebih kecil dari epochs.")
    best_f1 = -1.0
    best_raw_f1 = -1.0
    dual_expert_names = (
        ("gap", "hbp")
        if str(cfg["model"].get("head", "")).startswith("decoupled_gap_hbp_")
        else ()
    )
    best_expert_f1 = {name: -1.0 for name in dual_expert_names}
    history = []
    start_epoch = 0
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    last_checkpoint = output_dir / "last.pt"
    if resume and last_checkpoint.is_file():
        checkpoint = load_resume_checkpoint(last_checkpoint, device)
        if checkpoint is not None:
            required_state = {"optimizer", "scheduler", "history", "best_f1"}
            if dual_expert_names:
                required_state.add("best_expert_f1")
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
                if dual_expert_names:
                    best_expert_f1 = {
                        name: float(checkpoint["best_expert_f1"][name])
                        for name in dual_expert_names
                    }
                if ema is not None:
                    best_raw_f1 = float(checkpoint["best_raw_f1"])
                start_epoch = int(checkpoint["epoch"])
                print(
                    f"RESUME: melanjutkan dari epoch {start_epoch + 1}/{epochs}",
                    flush=True,
                )

    for epoch in range(start_epoch, epochs):
        model.train()
        if freeze_backbone:
            model.encoder.eval()
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
        gradient_cosine: float | None = None

        progress = tqdm(loaders.source_train, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_index, source_batch in enumerate(progress):
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

            if (
                batch_index == 0
                and source_output.expert_logits is not None
                and tuple(source_output.expert_logits) == ("gap", "hbp")
                and hasattr(model, "shared_parameters_for_audit")
            ):
                gradient_cosine = branch_gradient_cosine(
                    supervised_components["gap_ce"],
                    supervised_components["hbp_ce"],
                    model.shared_parameters_for_audit(),
                )

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
                gate_label = (
                    "gate_gap"
                    if tuple(source_output.expert_logits or ()) == ("gap", "hbp")
                    else "gate_hbp"
                )
                postfix[gate_label] = f"{(gate_sum[0] / gate_count).item():.3f}"
            if gradient_cosine is not None:
                postfix["grad_cos"] = f"{gradient_cosine:+.3f}"
            progress.set_postfix(**postfix)

        scheduler.step()
        source_metrics_raw = evaluate(
            model, loaders.source_val, device, loaders.classes, hard_groups
        )
        source_expert_metrics_raw = {
            name: evaluate(
                model,
                loaders.source_val,
                device,
                loaders.classes,
                hard_groups,
                prediction_head=name,
            )
            for name in dual_expert_names
        }
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
        if source_expert_metrics_raw:
            record["source_experts"] = source_expert_metrics_raw
        if running_components:
            record["loss_components"] = {
                name: value / max(len(loaders.source_train), 1)
                for name, value in running_components.items()
            }
        if gate_count:
            record["gate_mean"] = (gate_sum / gate_count).tolist()
        if gradient_cosine is not None:
            record["branch_gradient_cosine"] = gradient_cosine
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": record["epoch"],
                    "loss": record["loss"],
                    "loss_components": record.get("loss_components"),
                    "gate_mean": record.get("gate_mean"),
                    "branch_gradient_cosine": record.get("branch_gradient_cosine"),
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
                    "source_experts": {
                        name: {
                            key: value
                            for key, value in metrics.items()
                            if key != "per_class"
                        }
                        for name, metrics in source_expert_metrics_raw.items()
                    }
                    or None,
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
        expert_is_best = {
            name: metrics["macro_f1"] > best_expert_f1[name]
            for name, metrics in source_expert_metrics_raw.items()
        }
        for name, improved in expert_is_best.items():
            if improved:
                best_expert_f1[name] = source_expert_metrics_raw[name]["macro_f1"]

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
            "best_expert_f1": best_expert_f1,
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
        for name, improved in expert_is_best.items():
            if not improved:
                continue
            expert_checkpoint = {
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "epoch": epoch + 1,
                "target_metrics": target_metrics_raw,
                "best_f1": best_expert_f1[name],
                "weights": "raw",
                "selection_head": name,
            }
            atomic_torch_save(expert_checkpoint, output_dir / f"best_{name}.pt")
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
        if artifact_repo and resolved_artifact_path and (
            (epoch + 1) % artifact_sync_every == 0 or epoch + 1 == epochs
        ):
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["completed_epoch"] = epoch + 1
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"HF SYNC: epoch {epoch + 1}/{epochs} -> "
                f"{artifact_repo}/{resolved_artifact_path}",
                flush=True,
            )
            try:
                sync_artifacts(
                    artifact_repo,
                    resolved_artifact_path,
                    output_dir,
                    commit_message=(
                        f"Checkpoint {output_dir.name} epoch {epoch + 1}/{epochs}"
                    ),
                )
            except Exception as exc:
                print(
                    "WARNING: upload checkpoint ke Hugging Face gagal; "
                    f"training tetap dilanjutkan. {type(exc).__name__}: {exc}",
                    flush=True,
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
    parser.add_argument(
        "--artifact-repo",
        help="Repo model Hugging Face private, contoh user/coffee-checkpoints.",
    )
    parser.add_argument(
        "--artifact-path",
        help="Folder run di dalam repo Hugging Face.",
    )
    parser.add_argument(
        "--artifact-sync-every",
        type=int,
        default=5,
        help="Upload last.pt dan metadata setiap N epoch (default: 5).",
    )
    args = parser.parse_args()
    train(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        data_root_override=args.data_root,
        resume=args.resume,
        artifact_repo=args.artifact_repo,
        artifact_path=args.artifact_path,
        artifact_sync_every=args.artifact_sync_every,
    )


if __name__ == "__main__":
    main()
