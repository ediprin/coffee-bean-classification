from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    classification_metrics,
    resolve_device,
)
from bilinear_lmmd.engine.wavelet_residual_hbp import (
    build_candidate,
    validate_config as validate_wr_config,
)
from bilinear_lmmd.modeling.self_assessment_residual import (
    TopKSelfAssessmentResidual,
)


PROTOCOL = "coffee17-wr-hbp-self-assessment-v1"


def validate_config(cfg: dict) -> None:
    validate_wr_config(cfg)
    sar = cfg.get("self_assessment", {})
    if sar.get("base_model") != "WR_HBP":
        raise ValueError("self_assessment.base_model harus WR_HBP.")
    if sar.get("base_frozen") is not True:
        raise ValueError("WR-HBP base harus frozen.")
    if int(sar.get("top_k", -1)) != 5:
        raise ValueError("V1 dikunci top_k=5.")
    if int(sar.get("feature_index", -1)) != 1:
        raise ValueError("V1 memakai mid-level feature_index=1.")
    if int(sar.get("embedding_dim", -1)) != 128:
        raise ValueError("V1 embedding_dim=128.")
    if int(sar.get("hidden_dim", -1)) != 128:
        raise ValueError("V1 hidden_dim=128.")
    if sar.get("fusion") != "topk_additive_residual":
        raise ValueError("Fusion harus topk_additive_residual.")
    if sar.get("residual_init") != "zero":
        raise ValueError("Residual harus zero-initialized.")
    if sar.get("validation_tuning") is not False:
        raise ValueError("Validation tuning tidak diizinkan.")
    if int(sar.get("epochs", -1)) != 20:
        raise ValueError("V1 reassessment epochs=20.")
    if abs(float(sar.get("lr", -1.0)) - 3.0e-4) > 1.0e-12:
        raise ValueError("V1 reassessment lr=3e-4.")
    if abs(float(sar.get("weight_decay", -1.0)) - 1.0e-4) > 1.0e-12:
        raise ValueError("V1 reassessment weight_decay=1e-4.")


def build_self_assessment(
    cfg: dict,
    *,
    base_checkpoint: Path,
) -> TopKSelfAssessmentResidual:
    validate_config(cfg)
    checkpoint = torch.load(
        Path(base_checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    base = build_candidate(cfg)
    base.load_state_dict(checkpoint["model"])
    sar = cfg["self_assessment"]
    return TopKSelfAssessmentResidual(
        base,
        num_classes=int(cfg["model"]["num_classes"]),
        top_k=int(sar["top_k"]),
        embedding_dim=int(sar["embedding_dim"]),
        hidden_dim=int(sar["hidden_dim"]),
        feature_index=int(sar["feature_index"]),
    )


@torch.no_grad()
def zero_residual_preflight(
    cfg: dict,
    *,
    base_checkpoint: Path,
) -> dict:
    validate_config(cfg)
    base_checkpoint = Path(base_checkpoint)
    checkpoint = torch.load(
        base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    base = build_candidate(cfg).eval()
    base.load_state_dict(checkpoint["model"])
    candidate = build_self_assessment(
        cfg,
        base_checkpoint=base_checkpoint,
    ).eval()

    raw = torch.linspace(
        0.0,
        1.0,
        steps=2 * 3 * 224 * 224,
        dtype=torch.float32,
    ).reshape(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)

    base_logits = base(normalized, raw_rgb=raw).logits
    output = candidate(normalized, raw_rgb=raw)
    max_abs = float((base_logits - output.logits).abs().max().item())
    residual_max_abs = float(
        output.expert_logits["residual"].abs().max().item()
    )
    if max_abs > 1.0e-7 or residual_max_abs != 0.0:
        raise RuntimeError(
            "Zero-residual self-assessment tidak identik dengan WR-HBP base: "
            f"logit={max_abs}, residual={residual_max_abs}"
        )

    return {
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "zero_residual_initial_logit_max_abs_difference": max_abs,
        "zero_residual_max_abs": residual_max_abs,
        "trainable_parameter_count": candidate.trainable_parameter_count(),
        "base_frozen": all(
            not parameter.requires_grad
            for parameter in candidate.base.parameters()
        ),
    }


def _forward(model, images, device):
    raw = images.to(device, non_blocking=True)
    normalized = imagenet_normalize(raw)
    return model(normalized, raw_rgb=raw)


@torch.no_grad()
def _evaluate(
    model,
    loader,
    device,
    classes: list[str],
    hard_groups: dict,
) -> tuple[dict, list[int], list[int], list[list[float]], list[str]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    for images, targets in loader:
        output = _forward(model, images, device)
        probs = output.logits.softmax(1).cpu()
        labels.extend(targets.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())

    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah path validation berbeda dari prediksi.")

    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


def train_self_assessment(
    cfg: dict,
    *,
    base_checkpoint: Path,
    run_dir: Path,
    run_contract_sha256: str,
    resume: bool,
) -> dict:
    validate_config(cfg)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    model = build_self_assessment(
        cfg,
        base_checkpoint=base_checkpoint,
    ).to(device)
    model.train()

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Self-assessment tidak memiliki parameter trainable.")

    sar = cfg["self_assessment"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(sar["lr"]),
        weight_decay=float(sar["weight_decay"]),
    )
    epochs = int(sar["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
    )

    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_head.pt"
    last_path = run_dir / "last_head.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError("Checkpoint SAR berasal dari kontrak berbeda.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        model.train()
        running_loss = 0.0
        batches = 0
        progress = tqdm(
            loaders.train,
            desc=f"WR_HBP_SAR epoch {epoch + 1}/{epochs}",
        )

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, images, device)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics, _, _, _, _ = _evaluate(
            model,
            loaders.val,
            device,
            loaders.classes,
            hard_groups,
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(batches, 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        is_best = float(metrics["macro_f1"]) > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "base_checkpoint_sha256": sha256_file(base_checkpoint),
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "run_contract_sha256": run_contract_sha256,
                    "base_checkpoint_sha256": sha256_file(base_checkpoint),
                },
                best_path,
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    eval_model = build_self_assessment(
        cfg,
        base_checkpoint=base_checkpoint,
    )
    eval_model.load_state_dict(best["model"])
    eval_model = eval_model.to(device)

    metrics, labels, predictions, probabilities, paths = _evaluate(
        eval_model,
        loaders.val,
        device,
        loaders.classes,
        hard_groups,
    )
    _write_evaluation(
        run_dir / "validation",
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        paths=paths,
        classes=loaders.classes,
        checkpoint=best_path,
        split="val",
    )

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "best_epoch": int(best["epoch"]),
        "trainable_parameter_count": eval_model.trainable_parameter_count(),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
    }
