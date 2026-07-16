from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from .config import load_config
from .data import build_loaders
from .losses import KnowledgeDistillationLoss
from .models import build_model
from .run_cbd_stacking_confirmation import (
    _validate_pair,
    fit_meta_model,
    load_prediction_table,
)
from .train import (
    atomic_torch_save,
    evaluate,
    load_resume_checkpoint,
    resolve_device,
    seed_everything,
)


TEACHER_CHOICES = ("gap_cal", "stacking")


def _load_frozen_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    return model.to(device), checkpoint["classes"]


class FrozenStackingTeacher(nn.Module):
    """Torch reproduction of the fixed sklearn calibration/stacking teacher."""

    def __init__(
        self,
        gap_model: nn.Module,
        hbp_model: nn.Module | None,
        mean: np.ndarray,
        scale: np.ndarray,
        coefficients: np.ndarray,
        intercept: np.ndarray,
    ):
        super().__init__()
        self.gap_model = gap_model
        self.hbp_model = hbp_model
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer(
            "meta_weight", torch.as_tensor(coefficients, dtype=torch.float32)
        )
        self.register_buffer("meta_bias", torch.as_tensor(intercept, dtype=torch.float32))
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        # A teacher must never enable dropout or update BatchNorm statistics.
        return super().train(False)

    def forward(self, images: Tensor) -> Tensor:
        features = [F.log_softmax(self.gap_model(images).logits, dim=1)]
        if self.hbp_model is not None:
            features.append(F.log_softmax(self.hbp_model(images).logits, dim=1))
        stacked = torch.cat(features, dim=1)
        normalized = (stacked - self.feature_mean) / self.feature_scale
        return F.linear(normalized, self.meta_weight, self.meta_bias)


def build_teacher(
    teacher_kind: str,
    gap_checkpoint: Path,
    gap_val_predictions: Path,
    device: torch.device,
    hbp_checkpoint: Path | None = None,
    hbp_val_predictions: Path | None = None,
) -> tuple[FrozenStackingTeacher, list[str]]:
    if teacher_kind not in TEACHER_CHOICES:
        raise ValueError(f"teacher_kind harus salah satu {TEACHER_CHOICES}.")
    gap_table = load_prediction_table(gap_val_predictions)
    features = [np.log(gap_table.probabilities)]

    gap_model, gap_classes = _load_frozen_model(gap_checkpoint, device)
    hbp_model = None
    if teacher_kind == "stacking":
        if hbp_checkpoint is None or hbp_val_predictions is None:
            raise ValueError("Stacking teacher membutuhkan checkpoint/prediksi HBP.")
        hbp_table = load_prediction_table(hbp_val_predictions)
        _validate_pair(gap_table, hbp_table)
        features.append(np.log(hbp_table.probabilities))
        hbp_model, hbp_classes = _load_frozen_model(hbp_checkpoint, device)
        if hbp_classes != gap_classes:
            raise ValueError("Urutan kelas checkpoint GAP dan HBP berbeda.")

    if gap_table.classes != gap_classes:
        raise ValueError("Urutan kelas prediction CSV dan checkpoint berbeda.")
    pipeline = fit_meta_model(np.concatenate(features, axis=1), gap_table.labels)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    expected_classes = np.arange(len(gap_classes))
    if not np.array_equal(classifier.classes_, expected_classes):
        raise ValueError("Meta-model tidak melihat seluruh kelas validation.")
    teacher = FrozenStackingTeacher(
        gap_model,
        hbp_model,
        scaler.mean_,
        scaler.scale_,
        classifier.coef_,
        classifier.intercept_,
    ).to(device)
    return teacher, gap_classes


def train_distillation(
    config_path: Path,
    teacher_kind: str,
    gap_checkpoint: Path,
    gap_val_predictions: Path,
    seed_override: int | None = None,
    data_root_override: Path | None = None,
    output_dir_override: Path | None = None,
    hbp_checkpoint: Path | None = None,
    hbp_val_predictions: Path | None = None,
    resume: bool = False,
) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = seed_override
    if data_root_override is not None:
        cfg["data"]["root"] = str(data_root_override)
    if output_dir_override is not None:
        cfg["training"]["output_dir"] = str(output_dir_override)
    if cfg["adaptation"]["method"] != "source_only":
        raise ValueError("KD CBD hanya mendukung adaptation.method=source_only.")

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(cfg["device"])
    loaders = build_loaders(cfg["data"], require_target=False)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan student berbeda.")

    # Build the student at exactly the same seeded point as CBD0. Teacher model
    # construction is isolated from RNG so KDG and KDS have matched students.
    student = build_model(cfg["model"]).to(device)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    teacher, teacher_classes = build_teacher(
        teacher_kind,
        gap_checkpoint,
        gap_val_predictions,
        device,
        hbp_checkpoint,
        hbp_val_predictions,
    )
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if loaders.classes != teacher_classes:
        raise ValueError("Urutan kelas teacher dan student berbeda.")

    training_cfg = cfg["training"]
    distillation_cfg = cfg.get("distillation", {})
    temperature = float(distillation_cfg.get("temperature", 2.0))
    hard_weight = float(distillation_cfg.get("hard_weight", 0.5))
    criterion = KnowledgeDistillationLoss(
        temperature=temperature,
        hard_weight=hard_weight,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.1)),
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["distillation"] = {
        "teacher": teacher_kind,
        "temperature": temperature,
        "hard_weight": hard_weight,
        "gap_checkpoint": str(gap_checkpoint),
        "hbp_checkpoint": str(hbp_checkpoint) if hbp_checkpoint else None,
        "meta_fit_split": "validation",
        "teacher_input": "same online-augmented image as student",
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0
    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_resume_checkpoint(last_path, device)
        required = {"optimizer", "scheduler", "history", "best_f1"}
        if checkpoint is not None and required.issubset(checkpoint):
            if checkpoint.get("classes") != loaders.classes:
                raise ValueError("Urutan kelas checkpoint resume berbeda.")
            student.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            history = checkpoint["history"]
            best_f1 = float(checkpoint["best_f1"])
            start_epoch = int(checkpoint["epoch"])
            print(f"RESUME KD: epoch {start_epoch + 1}/{epochs}", flush=True)

    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})
    print(
        f"KD teacher={teacher_kind} | T={temperature:g} | "
        f"hard_weight={hard_weight:g} | device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        student.train()
        totals = {"loss": 0.0, "hard_ce": 0.0, "soft_kl": 0.0}
        progress = tqdm(loaders.source_train, desc=f"KD epoch {epoch + 1}/{epochs}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images).logits
            loss, components = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
            totals["loss"] += loss.item()
            totals["hard_ce"] += components["hard_ce"].item()
            totals["soft_kl"] += components["soft_kl"].item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{components['hard_ce'].item():.4f}",
                soft=f"{components['soft_kl'].item():.4f}",
            )
        scheduler.step()
        metrics = evaluate(
            student, loaders.source_val, device, loaders.classes, hard_groups
        )
        batches = max(len(loaders.source_train), 1)
        record = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / batches,
            "hard_ce": totals["hard_ce"] / batches,
            "soft_kl": totals["soft_kl"] / batches,
            "source": metrics,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": record["loss"],
                    "hard_ce": record["hard_ce"],
                    "soft_kl": record["soft_kl"],
                    "macro_f1": metrics["macro_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )
        is_best = metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = metrics["macro_f1"]
        checkpoint = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch + 1,
            "history": history,
            "best_f1": best_f1,
            "weights": "raw",
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(
                {
                    "model": student.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "weights": "raw",
                },
                output_dir / "best.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distil CBD ensemble into one GAP CNN")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", choices=TEACHER_CHOICES, required=True)
    parser.add_argument("--gap-checkpoint", type=Path, required=True)
    parser.add_argument("--gap-val-predictions", type=Path, required=True)
    parser.add_argument("--hbp-checkpoint", type=Path)
    parser.add_argument("--hbp-val-predictions", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_distillation(
        args.config,
        args.teacher,
        args.gap_checkpoint,
        args.gap_val_predictions,
        seed_override=args.seed,
        data_root_override=args.data_root,
        output_dir_override=args.output_dir,
        hbp_checkpoint=args.hbp_checkpoint,
        hbp_val_predictions=args.hbp_val_predictions,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
