from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm import tqdm

from bilinear_lmmd.core.artifact_store import (
    ensure_artifact_repo,
    normalize_remote_path,
    restore_artifacts,
    sync_artifacts,
)
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.loaders import build_loaders
from bilinear_lmmd.data.region_confusion import region_confusion
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    current_git_commit,
    evaluate,
    load_resume_checkpoint,
    resolve_device,
    seed_everything,
)
from bilinear_lmmd.engine.train_pairwise_contrastive import (
    ContrastiveProjectionHead,
    normalized_soft_confusion,
    update_confusion_ema,
)
from bilinear_lmmd.modeling.dcl_finegrained import (
    DCLFineGrainedModel,
    DCLTrainingOutput,
)
from bilinear_lmmd.modeling.losses import (
    ConfusionAwareSupervisedContrastiveLoss,
)
from bilinear_lmmd.modeling.models import build_model


def dcl_objective(
    output: DCLTrainingOutput,
    labels: Tensor,
    swap_targets: Tensor,
    layout_targets: Tensor,
    *,
    classification_loss: nn.Module,
    swap_weight: float,
    layout_weight: float,
    projection: nn.Module | None = None,
    contrastive_loss: nn.Module | None = None,
    contrastive_weight: float = 0.0,
    class_confusion: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if min(swap_weight, layout_weight, contrastive_weight) < 0.0:
        raise ValueError("Bobot objective DCL tidak boleh negatif.")
    if output.classification.logits.shape[0] != labels.shape[0]:
        raise ValueError("Logit klasifikasi dan labels tidak sejajar.")
    if output.swap_logits.shape[0] != swap_targets.shape[0]:
        raise ValueError("Logit swap dan swap_targets tidak sejajar.")
    if output.layout.shape != layout_targets.shape:
        raise ValueError(
            "Layout prediction/target berbeda: "
            f"{tuple(output.layout.shape)} != {tuple(layout_targets.shape)}."
        )

    ce = classification_loss(output.classification.logits, labels)
    swap = nn.functional.cross_entropy(output.swap_logits, swap_targets)
    layout = nn.functional.l1_loss(output.layout, layout_targets)
    components = {
        "classification_ce": ce,
        "swap_ce": swap,
        "layout_l1": layout,
    }
    total = ce + swap_weight * swap + layout_weight * layout

    if contrastive_weight > 0.0:
        if projection is None or contrastive_loss is None:
            raise ValueError(
                "Projection dan contrastive loss wajib ketika bobotnya positif."
            )
        projected = projection(output.classification.embedding)
        contrastive = contrastive_loss(projected, labels, class_confusion)
        total = total + contrastive_weight * contrastive
        components["contrastive"] = contrastive
    elif projection is not None or contrastive_loss is not None:
        raise ValueError(
            "Projection/contrastive loss tidak boleh aktif pada kontrol DCL murni."
        )
    return total, components


def _checkpoint(
    *,
    model: DCLFineGrainedModel,
    projection: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    classes: list[str],
    epoch: int,
    best_f1: float,
    history: list[dict],
    class_confusion: Tensor,
    rcm_generator: torch.Generator,
) -> dict:
    return {
        "model": model.state_dict(),
        "projection_head": (
            projection.state_dict() if projection is not None else None
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "classes": classes,
        "epoch": epoch,
        "best_f1": best_f1,
        "history": history,
        "class_confusion": class_confusion.cpu(),
        "rcm_rng_state": rcm_generator.get_state(),
        "git_commit": current_git_commit(),
    }


def _sync_or_raise(
    repo_id: str | None,
    remote_path: str | None,
    output_dir: Path,
    *,
    commit_message: str,
    required: bool,
) -> None:
    if not repo_id or not remote_path:
        return
    try:
        sync_artifacts(
            repo_id,
            remote_path,
            output_dir,
            commit_message=commit_message,
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                "Sinkronisasi checkpoint DCL wajib tetapi gagal."
            ) from exc
        print(
            "WARNING: sinkronisasi checkpoint DCL gagal: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def train_dcl_finegrained(
    config_path: str | Path,
    *,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
    data_root_override: str | Path | None = None,
    resume: bool = False,
    artifact_repo: str | None = None,
    artifact_path: str | None = None,
    artifact_sync_every: int | None = None,
    artifact_required: bool = False,
) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if output_dir_override is not None:
        cfg["training"]["output_dir"] = str(output_dir_override)
    if data_root_override is not None:
        cfg["data"]["root"] = str(data_root_override)

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    if str(cfg["adaptation"]["method"]).lower() != "source_only":
        raise ValueError("DCL Coffee17 hanya mendukung source_only.")
    if str(cfg["model"]["head"]).lower() != "dcl_gap":
        raise ValueError("Trainer DCL membutuhkan model.head=dcl_gap.")
    if str(cfg["model"]["classifier"]).lower() != "linear":
        raise ValueError("Trainer DCL membutuhkan classifier linear.")
    if str(cfg["training"].get("precision", "fp32")).lower() != "fp32":
        raise ValueError(
            "Screening DCL dikunci ke fp32 agar setara dengan baseline."
        )

    loaders = build_loaders(cfg["data"], require_target=False)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset berbeda dari model.num_classes.")
    model = build_model(cfg["model"]).to(device)
    if not isinstance(model, DCLFineGrainedModel):
        raise TypeError("Factory tidak menghasilkan DCLFineGrainedModel.")

    training_cfg = cfg["training"]
    mode = str(training_cfg.get("dcl_contrastive_mode", "none")).lower()
    if mode not in {"none", "standard", "confusion_aware"}:
        raise ValueError(
            "dcl_contrastive_mode harus none, standard, atau confusion_aware."
        )
    contrastive_weight = float(training_cfg.get("contrastive_weight", 0.0))
    confusion_strength = float(training_cfg.get("confusion_strength", 0.0))
    if mode == "none" and (
        contrastive_weight != 0.0 or confusion_strength != 0.0
    ):
        raise ValueError("Kontrol DCL murni harus mematikan contrastive.")
    if mode == "standard" and (
        contrastive_weight <= 0.0 or confusion_strength != 0.0
    ):
        raise ValueError(
            "DCL standard membutuhkan contrastive_weight>0 dan "
            "confusion_strength=0."
        )
    if mode == "confusion_aware" and (
        contrastive_weight <= 0.0 or confusion_strength <= 0.0
    ):
        raise ValueError(
            "DCL confusion-aware membutuhkan kedua bobot positif."
        )

    projection: ContrastiveProjectionHead | None = None
    if mode != "none":
        projection_rng_state = torch.random.get_rng_state()
        projection = ContrastiveProjectionHead(
            model.output_dim,
            int(training_cfg.get("contrastive_hidden_dim", model.output_dim)),
            int(training_cfg.get("contrastive_projection_dim", 128)),
        ).to(device)
        torch.random.set_rng_state(projection_rng_state)
    parameters = list(model.parameters())
    if projection is not None:
        parameters.extend(projection.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    classification_loss = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    swap_weight = float(training_cfg.get("dcl_swap_weight", 1.0))
    layout_weight = float(training_cfg.get("dcl_layout_weight", 1.0))
    grid_size = int(cfg["model"].get("dcl_grid_size", 7))
    temperature = float(training_cfg.get("contrastive_temperature", 0.1))
    warmup_epochs = int(training_cfg.get("confusion_warmup_epochs", 5))
    confusion_decay = float(training_cfg.get("confusion_ema_decay", 0.8))

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_repo = artifact_repo or os.environ.get(
        "BILINEAR_LMMD_ARTIFACT_REPO"
    )
    namespace = os.environ.get(
        "BILINEAR_LMMD_ARTIFACT_NAMESPACE", ""
    ).strip("/")
    sync_every = (
        int(artifact_sync_every)
        if artifact_sync_every is not None
        else int(os.environ.get("BILINEAR_LMMD_ARTIFACT_SYNC_EVERY", "1"))
    )
    if sync_every <= 0:
        raise ValueError("artifact_sync_every harus positif.")
    artifact_required = artifact_required or os.environ.get(
        "BILINEAR_LMMD_ARTIFACT_REQUIRED", ""
    ).lower() in {"1", "true", "yes", "on"}
    if artifact_required and not artifact_repo:
        raise RuntimeError("Artifact repo wajib untuk training DCL persisten.")
    remote_path: str | None = None
    if artifact_repo:
        remote_path = normalize_remote_path(
            artifact_path
            or "/".join(
                part
                for part in (namespace, "outputs", output_dir.name)
                if part
            )
        )
        ensure_artifact_repo(artifact_repo, private=True)
        restored = (
            restore_artifacts(
                artifact_repo,
                remote_path,
                output_dir,
                overwrite=False,
            )
            if resume
            else []
        )
        print(
            f"HF ARTIFACT: {artifact_repo}/{remote_path} | "
            f"restored={len(restored)} | sync_every={sync_every}",
            flush=True,
        )

    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "config_path": str(config_path),
        "seed": seed,
        "data_root": str(cfg["data"]["root"]),
        "artifact_repo": artifact_repo,
        "artifact_path": remote_path,
        "protocol": "coffee17_dcl_local_detail_v1",
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _sync_or_raise(
        artifact_repo,
        remote_path,
        output_dir,
        commit_message=f"Initialize DCL run {output_dir.name}",
        required=artifact_required,
    )

    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0
    num_classes = len(loaders.classes)
    class_confusion = torch.zeros(num_classes, num_classes, device=device)
    rcm_generator = torch.Generator().manual_seed(seed + 1729)
    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_resume_checkpoint(last_path, device)
        required = {
            "model",
            "optimizer",
            "scheduler",
            "history",
            "best_f1",
            "epoch",
            "class_confusion",
            "rcm_rng_state",
        }
        if checkpoint is not None and required.issubset(checkpoint):
            if checkpoint.get("classes") != loaders.classes:
                raise ValueError("Urutan kelas checkpoint berbeda dari dataset.")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if projection is not None:
                if checkpoint.get("projection_head") is None:
                    raise ValueError("Checkpoint contrastive tanpa projection head.")
                projection.load_state_dict(checkpoint["projection_head"])
            elif checkpoint.get("projection_head") is not None:
                raise ValueError("Checkpoint DCL murni memiliki projection head.")
            history = checkpoint["history"]
            best_f1 = float(checkpoint["best_f1"])
            start_epoch = int(checkpoint["epoch"])
            class_confusion = checkpoint["class_confusion"].to(device)
            # load_resume_checkpoint maps tensor payloads to the training
            # device. This generator intentionally remains on CPU so its
            # state is portable across CUDA/CPU sessions.
            rcm_generator.set_state(checkpoint["rcm_rng_state"].cpu())
            print(f"RESUME: epoch {start_epoch + 1}/{epochs}", flush=True)

    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})
    print(
        "DCL FINE-GRAINED TRAINING | "
        f"mode={mode} grid={grid_size} swap={swap_weight:.2f} "
        f"layout={layout_weight:.2f} contrastive={contrastive_weight:.2f} "
        f"device={device}",
        flush=True,
    )
    print(
        f"PARAMETER: inference={model.inference_parameter_count():,} "
        f"auxiliary={model.auxiliary_parameter_count():,}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        if projection is not None:
            projection.train()
        active_strength = (
            confusion_strength
            if mode == "confusion_aware" and epoch >= warmup_epochs
            else 0.0
        )
        contrastive_loss = (
            ConfusionAwareSupervisedContrastiveLoss(
                temperature=temperature,
                confusion_strength=active_strength,
            )
            if mode != "none"
            else None
        )
        running: dict[str, float] = {
            "loss": 0.0,
            "classification_ce": 0.0,
            "swap_ce": 0.0,
            "layout_l1": 0.0,
            "contrastive": 0.0,
        }
        epoch_labels: list[Tensor] = []
        epoch_probabilities: list[Tensor] = []
        progress = tqdm(
            loaders.source_train,
            desc=f"DCL {mode} {epoch + 1}/{epochs}",
        )
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            confused = region_confusion(
                images,
                grid_size,
                generator=rcm_generator,
            )
            paired_images = torch.cat((images, confused.images), dim=0)
            paired_labels = torch.cat((labels, labels), dim=0)
            swap_targets = torch.cat(
                (
                    torch.zeros_like(labels),
                    torch.ones_like(labels),
                )
            )
            layout_targets = torch.cat(
                (confused.original_layout, confused.confused_layout),
                dim=0,
            )

            optimizer.zero_grad(set_to_none=True)
            output = model.forward_dcl(paired_images)
            total, components = dcl_objective(
                output,
                paired_labels,
                swap_targets,
                layout_targets,
                classification_loss=classification_loss,
                swap_weight=swap_weight,
                layout_weight=layout_weight,
                projection=projection,
                contrastive_loss=contrastive_loss,
                contrastive_weight=contrastive_weight,
                class_confusion=(
                    class_confusion if active_strength > 0.0 else None
                ),
            )
            total.backward()
            optimizer.step()

            running["loss"] += float(total.detach())
            for name, value in components.items():
                running[name] += float(value.detach())
            original_logits = output.classification.logits[: labels.shape[0]]
            epoch_labels.append(labels.detach().cpu())
            epoch_probabilities.append(original_logits.detach().softmax(1).cpu())
            progress.set_postfix(
                loss=f"{float(total.detach()):.3f}",
                ce=f"{float(components['classification_ce'].detach()):.3f}",
                swap=f"{float(components['swap_ce'].detach()):.3f}",
                layout=f"{float(components['layout_l1'].detach()):.3f}",
            )

        current_confusion = normalized_soft_confusion(
            torch.cat(epoch_labels),
            torch.cat(epoch_probabilities),
            num_classes,
        ).to(device)
        class_confusion = update_confusion_ema(
            class_confusion,
            current_confusion,
            confusion_decay,
        )
        scheduler.step()
        metrics = evaluate(
            model,
            loaders.source_val,
            device,
            loaders.classes,
            hard_groups,
        )
        batch_count = max(len(loaders.source_train), 1)
        record = {
            "epoch": epoch + 1,
            **{
                name: value / batch_count
                for name, value in running.items()
            },
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
                    "classification_ce": record["classification_ce"],
                    "swap_ce": record["swap_ce"],
                    "layout_l1": record["layout_l1"],
                    "contrastive": record["contrastive"],
                    "source": {
                        key: value
                        for key, value in metrics.items()
                        if key != "per_class"
                    },
                }
            ),
            flush=True,
        )

        is_best = float(metrics["macro_f1"]) > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])
        checkpoint = _checkpoint(
            model=model,
            projection=projection,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            classes=loaders.classes,
            epoch=epoch + 1,
            best_f1=best_f1,
            history=history,
            class_confusion=class_confusion,
            rcm_generator=rcm_generator,
        )
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "weights": "raw",
                    "training_only_heads": [
                        "swap_classifier",
                        "layout_head",
                        *(
                            ["contrastive_projection"]
                            if projection is not None
                            else []
                        ),
                    ],
                    "inference_path": "encoder_gap_linear",
                },
                output_dir / "best.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_epochs"] = epoch + 1
        manifest["best_macro_f1"] = best_f1
        (output_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if (epoch + 1) % sync_every == 0 or epoch + 1 == epochs:
            _sync_or_raise(
                artifact_repo,
                remote_path,
                output_dir,
                commit_message=(
                    f"DCL {output_dir.name} epoch {epoch + 1}/{epochs}"
                ),
                required=artifact_required,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train paper-grounded DCL adaptation on Coffee17."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-repo")
    parser.add_argument("--artifact-path")
    parser.add_argument("--artifact-sync-every", type=int)
    parser.add_argument("--artifact-required", action="store_true")
    args = parser.parse_args()
    train_dcl_finegrained(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        data_root_override=args.data_root,
        resume=args.resume,
        artifact_repo=args.artifact_repo,
        artifact_path=args.artifact_path,
        artifact_sync_every=args.artifact_sync_every,
        artifact_required=args.artifact_required,
    )


if __name__ == "__main__":
    main()
