from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    model_state_fingerprint,
    restore_rng_state,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.preprocessing_study import _evaluate_model, _write_evaluation
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.modeling.models import build_model


ARMS = ("R0", "C0", "F0", "W0")
TEACHER_MODES = ("R0", "ALL4")
PROTOCOL = "coffee17-preprocessing-kd-exploratory-v1"


@dataclass(frozen=True)
class FrozenTeacher:
    arm: str
    model: nn.Module
    runtime: PreprocessingRuntime
    checkpoint_path: Path
    checkpoint_sha256: str


def _json(path: Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _authority_run_map(authority: dict) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for row in authority.get("primary_runs", []):
        key = (str(row["arm"]).upper(), int(row["fold"]))
        if key in rows:
            raise RuntimeError(f"Primary authority memiliki run duplikat: {key}")
        rows[key] = row
    return rows


def validate_kd_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("Preprocessing-KD v1 dikunci pada seed 42.")
    if cfg.get("model", {}).get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Student KD v1 harus memakai MobileNetV3-Large.")
    if cfg.get("model", {}).get("head") != "gap":
        raise ValueError("Student KD v1 harus memakai GAP head.")
    preprocessing = cfg.get("preprocessing", {})
    if str(preprocessing.get("code", "")).upper() != "R0":
        raise ValueError("Student KD harus menerima RGB/R0 pada inference.")
    if preprocessing.get("method") != "raw":
        raise ValueError("Student KD harus memakai frontend raw/R0.")
    mode = str(cfg.get("distillation", {}).get("teacher_mode", "")).upper()
    if mode not in TEACHER_MODES:
        raise ValueError(f"teacher_mode harus salah satu {TEACHER_MODES}.")
    temperature = float(cfg.get("distillation", {}).get("temperature", 0.0))
    hard_weight = float(cfg.get("distillation", {}).get("hard_weight", -1.0))
    if temperature <= 0.0:
        raise ValueError("temperature harus > 0.")
    if not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight harus berada pada [0,1].")


def load_frozen_teachers(
    *,
    authority_path: Path,
    experiments_root: Path,
    fold: int,
    device: torch.device,
) -> tuple[dict[str, FrozenTeacher], list[str], dict]:
    authority = _json(authority_path, "Primary confirmation")
    if authority.get("decision") != "AUTHORIZE_OOF_TEST_EVALUATION":
        raise RuntimeError("Primary confirmation tidak valid.")
    if int(authority.get("seed", -1)) != 42:
        raise RuntimeError("Primary confirmation bukan seed-42.")
    if authority.get("all_validation_only") is not True:
        raise RuntimeError("Teacher authority tidak berasal dari validation-only primary runs.")

    run_map = _authority_run_map(authority)
    experiments_root = Path(experiments_root).expanduser().resolve()
    teachers: dict[str, FrozenTeacher] = {}
    classes: list[str] | None = None
    metadata: dict[str, dict] = {}

    for arm in ARMS:
        key = (arm, int(fold))
        if key not in run_map:
            raise RuntimeError(f"Teacher primary tidak ada dalam authority: {key}")
        row = run_map[key]
        checkpoint_path = experiments_root / row["checkpoint_relative_path"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint teacher tidak ditemukan: {checkpoint_path}")
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != row["checkpoint_sha256"]:
            raise RuntimeError(f"SHA teacher berubah: {arm}/fold{fold}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_classes = list(checkpoint["classes"])
        if classes is None:
            classes = checkpoint_classes
        elif classes != checkpoint_classes:
            raise RuntimeError("Urutan kelas berbeda antar-teacher.")

        teacher_cfg = copy.deepcopy(checkpoint["config"])
        if int(teacher_cfg.get("seed", -1)) != 42:
            raise RuntimeError(f"Seed checkpoint teacher salah: {arm}")
        if str(teacher_cfg.get("preprocessing", {}).get("code", "")).upper() != arm:
            raise RuntimeError(f"Frontend checkpoint tidak cocok dengan arm {arm}.")
        teacher_cfg["model"]["pretrained"] = False

        model = build_model(teacher_cfg["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.requires_grad_(False)
        model.eval()
        runtime = PreprocessingRuntime.from_config(teacher_cfg["preprocessing"], device)

        teachers[arm] = FrozenTeacher(
            arm=arm,
            model=model,
            runtime=runtime,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
        )
        metadata[arm] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "authority_validation_macro_f1": float(row["validation_macro_f1"]),
        }

    assert classes is not None
    return teachers, classes, metadata


def mean_temperature_teacher_probabilities(
    logits: list[Tensor],
    temperature: float,
) -> Tensor:
    if not logits:
        raise ValueError("Minimal satu teacher logits diperlukan.")
    shape = logits[0].shape
    if any(value.shape != shape for value in logits):
        raise ValueError("Semua teacher logits harus memiliki shape identik.")
    probabilities = [
        F.softmax(value / float(temperature), dim=1)
        for value in logits
    ]
    return torch.stack(probabilities, dim=0).mean(dim=0)


def kd_loss_from_teacher_probabilities(
    student_logits: Tensor,
    teacher_probabilities: Tensor,
    labels: Tensor,
    *,
    temperature: float,
    hard_weight: float,
    label_smoothing: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if student_logits.shape != teacher_probabilities.shape:
        raise ValueError(
            "Dimensi student logits dan teacher probabilities harus identik "
            f"({student_logits.shape} vs {teacher_probabilities.shape})."
        )
    hard = F.cross_entropy(
        student_logits,
        labels,
        label_smoothing=float(label_smoothing),
    )
    soft = F.kl_div(
        F.log_softmax(student_logits / float(temperature), dim=1),
        teacher_probabilities,
        reduction="batchmean",
    ) * (float(temperature) ** 2)
    total = float(hard_weight) * hard + (1.0 - float(hard_weight)) * soft
    return total, {"hard_ce": hard.detach(), "soft_kl": soft.detach()}


@torch.no_grad()
def teacher_probabilities_for_batch(
    teachers: dict[str, FrozenTeacher],
    images: Tensor,
    *,
    mode: str,
    temperature: float,
) -> Tensor:
    mode = str(mode).upper()
    if mode == "R0":
        selected = ("R0",)
    elif mode == "ALL4":
        selected = ARMS
    else:
        raise ValueError(f"Mode teacher tidak dikenal: {mode}")

    logits: list[Tensor] = []
    for arm in selected:
        teacher = teachers[arm]
        teacher.model.eval()
        model_input = teacher.runtime(images)
        logits.append(teacher.model(model_input).logits)
    return mean_temperature_teacher_probabilities(logits, temperature)


def train_preprocessing_kd(
    cfg: dict,
    *,
    fold: int,
    data_root: Path,
    authority_path: Path,
    experiments_root: Path,
    run_dir: Path,
    run_contract_sha256: str,
    resume: bool = False,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_kd_config(cfg)
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan student berbeda.")

    # Build the student at the same seeded point as the original R0 primary arm.
    student = build_model(cfg["model"]).to(device)
    initial_sha = model_state_fingerprint(student)

    authority = _json(authority_path, "Primary confirmation")
    expected_initial_sha = authority.get("invariants", {}).get(
        "common_initialized_model_state_sha256"
    )
    if not expected_initial_sha:
        raise RuntimeError("Primary authority tidak memuat initial-model fingerprint.")
    if initial_sha != expected_initial_sha:
        raise RuntimeError(
            "Initial student berbeda dari primary R0 initialization: "
            f"{initial_sha} != {expected_initial_sha}"
        )

    # Teacher construction initializes temporary model parameters before loading
    # checkpoints. Preserve RNG so it cannot alter the matched student trajectory.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    teachers, teacher_classes, teacher_metadata = load_frozen_teachers(
        authority_path=authority_path,
        experiments_root=experiments_root,
        fold=fold,
        device=device,
    )
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    if loaders.classes != teacher_classes:
        raise RuntimeError("Urutan kelas teacher dan student berbeda.")

    student_runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    training_cfg = cfg["training"]
    distillation_cfg = cfg["distillation"]
    teacher_mode = str(distillation_cfg["teacher_mode"]).upper()
    temperature = float(distillation_cfg["temperature"])
    hard_weight = float(distillation_cfg["hard_weight"])
    label_smoothing = float(training_cfg.get("label_smoothing", 0.1))

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir = Path(run_dir).expanduser().resolve()
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
            raise RuntimeError("Urutan kelas checkpoint resume berbeda.")
        required = {"model", "optimizer", "scheduler", "history", "best_f1", "rng_state"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint resume tidak lengkap: {missing}")
        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME KD: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        student.train()
        totals = {"loss": 0.0, "hard_ce": 0.0, "soft_kl": 0.0}
        progress = tqdm(loaders.train, desc=f"KD {teacher_mode} epoch {epoch + 1}/{epochs}")

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            teacher_probs = teacher_probabilities_for_batch(
                teachers,
                images,
                mode=teacher_mode,
                temperature=temperature,
            )
            student_input = student_runtime(images)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(student_input, labels=labels).logits
            loss, parts = kd_loss_from_teacher_probabilities(
                student_logits,
                teacher_probs,
                labels,
                temperature=temperature,
                hard_weight=hard_weight,
                label_smoothing=label_smoothing,
            )
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["hard_ce"] += float(parts["hard_ce"].item())
            totals["soft_kl"] += float(parts["soft_kl"].item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{parts['hard_ce'].item():.4f}",
                soft=f"{parts['soft_kl'].item():.4f}",
            )

        scheduler.step()
        metrics, _, _, _, _ = _evaluate_model(
            student,
            loaders.val,
            student_runtime,
            device,
            loaders.classes,
            hard_groups,
        )
        batches = max(len(loaders.train), 1)
        record = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / batches,
            "hard_ce": totals["hard_ce"] / batches,
            "soft_kl": totals["soft_kl"] / batches,
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "teacher_mode": teacher_mode,
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
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch + 1,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "initial_model_state_sha256": initial_sha,
            "teacher_metadata": teacher_metadata,
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
                    "run_contract_sha256": run_contract_sha256,
                    "initial_model_state_sha256": initial_sha,
                    "teacher_metadata": teacher_metadata,
                },
                best_path,
            )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not best_path.is_file():
        raise RuntimeError("Training KD selesai tanpa best checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    eval_model_cfg = copy.deepcopy(best["config"]["model"])
    eval_model_cfg["pretrained"] = False
    eval_model = build_model(eval_model_cfg).to(device)
    eval_model.load_state_dict(best["model"])
    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        eval_model,
        loaders.val,
        student_runtime,
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
        "protocol": PROTOCOL,
        "fold": int(fold),
        "seed": seed,
        "teacher_mode": teacher_mode,
        "best_checkpoint": str(best_path),
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "initial_model_state_sha256": initial_sha,
        "teacher_metadata": teacher_metadata,
    }
