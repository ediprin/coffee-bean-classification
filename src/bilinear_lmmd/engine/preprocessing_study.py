from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    model_state_fingerprint,
    restore_rng_state,
    seed_everything,
)
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    build_preprocessing_evaluation_loader,
    set_study_epoch,
)
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    classification_metrics,
    resolve_device,
)
from bilinear_lmmd.modeling.models import build_model


def _evaluate_model(
    model: torch.nn.Module,
    loader,
    runtime: PreprocessingRuntime,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
) -> tuple[dict, list[int], list[int], list[list[float]], list[str]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, targets in loader:
            model_input = runtime(images)
            output = model(model_input)
            probs = output.logits.softmax(1).cpu()
            labels.extend(targets.tolist())
            predictions.extend(probs.argmax(1).tolist())
            probabilities.extend(probs.tolist())
    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah path validation berbeda dari jumlah prediksi.")
    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


def _write_evaluation(
    output_dir: Path,
    *,
    metrics: dict,
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    paths: list[str],
    classes: list[str],
    checkpoint: Path,
    split: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metrics,
        "checkpoint": str(checkpoint),
        "split": split,
        "classes": classes,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(classes)))
    )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *classes])
        for name, row in zip(classes, matrix.tolist()):
            writer.writerow([name, *row])
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "actual", "predicted", "correct"]
            + [f"prob::{name}" for name in classes]
        )
        for path, actual, predicted, probs in zip(
            paths, labels, predictions, probabilities
        ):
            writer.writerow(
                [
                    path,
                    classes[actual],
                    classes[predicted],
                    int(actual == predicted),
                    *probs,
                ]
            )


def train_preprocessing_study(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_model_sha256: str,
    resume: bool,
) -> dict:
    """Source-only M0 recipe isolated for the frozen preprocessing study."""

    if cfg["adaptation"]["method"] != "source_only":
        raise ValueError("Preprocessing study dikunci source_only.")
    if cfg["model"]["head"] != "gap":
        raise ValueError("Preprocessing study dikunci pada GAP.")
    if cfg["training"]["classification_loss"] != "cross_entropy":
        raise ValueError("Preprocessing study dikunci CrossEntropy.")
    if float(cfg["training"].get("ema_decay", 0.0)) != 0.0:
        raise ValueError("Primary preprocessing study tidak menggunakan EMA.")

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError(
            f"Dataset {len(loaders.classes)} kelas, model {cfg['model']['num_classes']}."
        )

    model = build_model(cfg["model"]).to(device)
    initial_sha = model_state_fingerprint(model)
    if initial_sha != expected_initial_model_sha256:
        raise RuntimeError(
            "Initial model berbeda dari static preflight: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )

    runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0))
    )
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError("Checkpoint resume berasal dari run-contract berbeda.")
        if checkpoint.get("classes") != loaders.classes:
            raise RuntimeError("Urutan kelas checkpoint berbeda.")
        required = {"model", "optimizer", "scheduler", "history", "best_f1", "rng_state"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint resume tidak lengkap: {missing}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        model.train()
        running_loss = 0.0
        progress = tqdm(loaders.train, desc=f"epoch {epoch + 1}/{epochs}")
        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            model_input = runtime(images)
            optimizer.zero_grad(set_to_none=True)
            output = model(model_input, labels=labels)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics, _, _, _, _ = _evaluate_model(
            model,
            loaders.val,
            runtime,
            device,
            loaders.classes,
            hard_groups,
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(len(loaders.train), 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": record["loss"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "worst_class_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )
        is_best = metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = metrics["macro_f1"]

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch + 1,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "initial_model_state_sha256": expected_initial_model_sha256,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            best_checkpoint = {
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "weights": "raw",
                "run_contract_sha256": run_contract_sha256,
                "initial_model_state_sha256": expected_initial_model_sha256,
            }
            atomic_torch_save(best_checkpoint, best_path)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Training selesai tanpa best/last checkpoint.")
    final_last = torch.load(last_path, map_location="cpu", weights_only=False)
    if int(final_last.get("epoch", 0)) < epochs:
        raise RuntimeError("Training belum menyelesaikan seluruh epoch kontrak.")

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    eval_cfg = best_checkpoint["config"]
    eval_device = resolve_device(str(eval_cfg["device"]))
    eval_loaders = build_preprocessing_study_loaders(eval_cfg["data"], seed=seed)
    eval_model_cfg = dict(eval_cfg["model"])
    eval_model_cfg["pretrained"] = False
    eval_model = build_model(eval_model_cfg).to(eval_device)
    eval_model.load_state_dict(best_checkpoint["model"])
    eval_runtime = PreprocessingRuntime.from_config(
        eval_cfg["preprocessing"], eval_device
    )
    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        eval_model,
        eval_loaders.val,
        eval_runtime,
        eval_device,
        eval_loaders.classes,
        hard_groups,
    )
    _write_evaluation(
        run_dir / "validation",
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        paths=paths,
        classes=eval_loaders.classes,
        checkpoint=best_path,
        split="val",
    )
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "initial_model_state_sha256": expected_initial_model_sha256,
    }


def evaluate_preprocessing_checkpoint(
    checkpoint_path: Path,
    *,
    data_root: Path,
    split: str,
    output_dir: Path,
) -> dict:
    """Inference-only evaluator used later by validation/OOF stages."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["root"] = str(data_root)
    cfg["data"]["val_split"] = split
    cfg["model"]["pretrained"] = False
    seed = int(cfg["seed"])
    device = resolve_device(str(cfg["device"]))
    loader, classes = build_preprocessing_evaluation_loader(
        cfg["data"], split=split, seed=seed
    )
    if checkpoint.get("classes") != classes:
        raise RuntimeError("Urutan kelas evaluation split berbeda dari checkpoint.")
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        model,
        loader,
        runtime,
        device,
        classes,
        cfg.get("evaluation", {}).get("hard_groups", {}),
    )
    _write_evaluation(
        output_dir,
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        paths=paths,
        classes=classes,
        checkpoint=checkpoint_path,
        split=split,
    )
    return metrics
