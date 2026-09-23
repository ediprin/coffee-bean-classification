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
)
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.preprocessing_study import _evaluate_model, _write_evaluation
from bilinear_lmmd.engine.shared_multiview_sbn import (
    auxiliary_batchnorm_batch_stats,
)
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.modeling.models import build_model


VIEWS = ("R0", "C0", "F0", "W0")
AUX_VIEWS = ("C0", "F0", "W0")
PROTOCOL = "coffee17-at-sbn-v1"


def validate_at_sbn_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("AT-SBN v1 dikunci seed 42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("AT-SBN v1 harus source_only.")
    if cfg.get("model", {}).get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if cfg.get("model", {}).get("head") != "gap":
        raise ValueError("Head harus GAP.")
    if str(cfg.get("model", {}).get("classifier", "linear")) != "linear":
        raise ValueError("AT-SBN v1 membutuhkan classifier linear.")
    if cfg.get("model", {}).get("out_indices") != [4]:
        raise ValueError("out_indices harus [4].")
    if cfg.get("data", {}).get("augmentation_mode") != "preprocessing_study":
        raise ValueError("Augmentasi harus preprocessing_study.")
    if bool(cfg.get("data", {}).get("object_crop", False)):
        raise ValueError("object_crop harus false.")
    if int(cfg.get("training", {}).get("epochs", -1)) != 50:
        raise ValueError("Eksperimen utama harus 50 epoch.")
    if cfg.get("training", {}).get("classification_loss") != "cross_entropy":
        raise ValueError("Loss klasifikasi harus cross_entropy.")
    if float(cfg.get("training", {}).get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")

    at = cfg.get("auxiliary_training", {})
    if at.get("method") != "at_sbn":
        raise ValueError("auxiliary_training.method harus at_sbn.")
    if tuple(at.get("views", ())) != VIEWS:
        raise ValueError(f"views harus tepat {VIEWS}.")
    if str(at.get("primary_view", "")).upper() != "R0":
        raise ValueError("primary_view harus R0.")
    if bool(at.get("selective_batch_norm")) is not True:
        raise ValueError("selective_batch_norm harus true.")
    if str(at.get("self_distill_teacher", "")).upper() != "R0":
        raise ValueError("Self-distillation teacher harus R0.")
    if bool(at.get("self_distill_stop_gradient")) is not True:
        raise ValueError("Primary teacher harus stop-gradient.")
    if bool(at.get("auxiliary_dropout")) is not False:
        raise ValueError("AT-SBN v1 mengunci auxiliary_dropout=false.")

    if abs(float(at.get("aux_ce_weight_each", -1.0)) - 0.05) > 1.0e-12:
        raise ValueError("aux_ce_weight_each v1 dikunci 0.05.")
    if abs(float(at.get("self_distill_weight_each", -1.0)) - 0.05) > 1.0e-12:
        raise ValueError("self_distill_weight_each v1 dikunci 0.05.")
    if abs(float(at.get("self_distill_temperature", -1.0)) - 1.0) > 1.0e-12:
        raise ValueError("self_distill_temperature v1 dikunci 1.0.")
    if abs(float(at.get("merge_weight", -1.0)) - 1.0) > 1.0e-12:
        raise ValueError("merge_weight v1 dikunci 1.0.")
    if int(at.get("merge_start_epoch", -1)) != 44:
        raise ValueError("merge_start_epoch v1 dikunci epoch 44.")

    frontends = at.get("frontends", {})
    if tuple(frontends) != VIEWS:
        raise ValueError("Frontend harus tepat R0/C0/F0/W0.")
    for arm in VIEWS:
        if str(frontends[arm].get("code", "")).upper() != arm:
            raise ValueError(f"Frontend {arm} memiliki code salah.")
    if str(cfg.get("preprocessing", {}).get("code", "")).upper() != "R0":
        raise ValueError("Deployment/evaluation preprocessing harus R0.")


def build_at_sbn_runtimes(
    cfg: dict,
    device: torch.device,
) -> dict[str, PreprocessingRuntime]:
    validate_at_sbn_config(cfg)
    return {
        arm: PreprocessingRuntime.from_config(
            cfg["auxiliary_training"]["frontends"][arm],
            device,
        )
        for arm in VIEWS
    }


@dataclass
class ViewForward:
    logits: Tensor
    embedding: Tensor


class AuxiliaryTrainingModel(nn.Module):
    """One deployable R0 model plus three training-only linear classifiers.

    The base model is built first and fingerprinted before any auxiliary
    parameter is created. Auxiliary-head initialization is wrapped by RNG
    save/restore so adding these heads does not alter the subsequent global
    RNG stream used by the primary model's training.
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.base = build_model(model_cfg)
        if not hasattr(self.base, "encoder") or not hasattr(self.base, "pool"):
            raise TypeError("AT-SBN membutuhkan AdaptationModel standar.")
        if not isinstance(self.base.classifier, nn.Linear):
            raise TypeError("Primary classifier AT-SBN harus nn.Linear.")

        self.primary_initial_sha256 = model_state_fingerprint(self.base)
        dim = int(self.base.classifier.in_features)
        classes = int(self.base.classifier.out_features)

        rng_state = torch.random.get_rng_state()
        self.auxiliary_classifiers = nn.ModuleDict(
            {
                arm: nn.Linear(dim, classes)
                for arm in AUX_VIEWS
            }
        )
        torch.random.set_rng_state(rng_state)

    def forward_primary(
        self,
        images: Tensor,
        labels: Tensor | None = None,
    ) -> ViewForward:
        output = self.base(images, labels=labels)
        return ViewForward(logits=output.logits, embedding=output.embedding)

    def forward_auxiliary(
        self,
        arm: str,
        images: Tensor,
    ) -> ViewForward:
        if arm not in self.auxiliary_classifiers:
            raise KeyError(f"Auxiliary arm tidak dikenal: {arm}")
        features = self.base.encoder(images)
        embedding = self.base.pool(features)
        # Deliberately no extra dropout in the training-only auxiliary heads.
        logits = self.auxiliary_classifiers[arm](embedding)
        return ViewForward(logits=logits, embedding=embedding)


def soft_target_cross_entropy(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Zhang-style self-distillation with a detached primary target.

    At temperature=1 this has the same gradient as KL(teacher || student),
    differing only by the teacher entropy constant.
    """
    if temperature <= 0.0:
        raise ValueError("temperature harus > 0.")
    target = F.softmax(teacher_logits.detach() / temperature, dim=1)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    return -(target * log_student).sum(dim=1).mean() * (temperature ** 2)


def classifier_merge_loss(model: AuxiliaryTrainingModel) -> tuple[Tensor, dict[str, float]]:
    primary = model.base.classifier
    total = primary.weight.new_zeros(())
    distances: dict[str, float] = {}
    for arm, head in model.auxiliary_classifiers.items():
        value = (head.weight - primary.weight).square().sum()
        if head.bias is not None and primary.bias is not None:
            value = value + (head.bias - primary.bias).square().sum()
        total = total + value
        distances[arm] = float(value.detach().item())
    return total, distances


def _js_divergence(left_logits: Tensor, right_logits: Tensor) -> Tensor:
    left = F.softmax(left_logits.detach(), dim=1).clamp_min(1.0e-8)
    right = F.softmax(right_logits.detach(), dim=1).clamp_min(1.0e-8)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=1)
        + (right * (right.log() - middle.log())).sum(dim=1)
    ).mean()


@torch.no_grad()
def evaluate_auxiliary_view(
    model: AuxiliaryTrainingModel,
    loader,
    runtime: PreprocessingRuntime,
    arm: str,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
) -> dict:
    from bilinear_lmmd.engine.train import classification_metrics

    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    for images, targets in loader:
        images = runtime(images)
        # Auxiliary transformed inputs use their own current batch statistics,
        # exactly as in training, without changing persistent R0 BN buffers.
        with auxiliary_batchnorm_batch_stats(model.base):
            output = model.forward_auxiliary(arm, images)
        predictions.extend(output.logits.argmax(1).cpu().tolist())
        labels.extend(targets.tolist())
    return classification_metrics(labels, predictions, classes, hard_groups)


def train_at_sbn(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_primary_initial_sha256: str,
    resume: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    validate_at_sbn_config(cfg)

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")

    model = AuxiliaryTrainingModel(cfg["model"]).to(device)
    if model.primary_initial_sha256 != expected_primary_initial_sha256:
        raise RuntimeError(
            "Primary initial model berbeda dari matched R0 reference: "
            f"{model.primary_initial_sha256} != {expected_primary_initial_sha256}"
        )

    runtimes = build_at_sbn_runtimes(cfg, device)
    eval_runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)

    training_cfg = cfg["training"]
    at_cfg = cfg["auxiliary_training"]
    aux_ce_weight = float(at_cfg["aux_ce_weight_each"])
    kd_weight = float(at_cfg["self_distill_weight_each"])
    temperature = float(at_cfg["self_distill_temperature"])
    merge_weight = float(at_cfg["merge_weight"])
    merge_start_epoch = int(at_cfg["merge_start_epoch"])

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.1))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
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
            raise RuntimeError("Checkpoint resume berasal dari kontrak berbeda.")
        if checkpoint.get("classes") != loaders.classes:
            raise RuntimeError("Urutan kelas checkpoint berbeda.")
        required = {
            "model", "optimizer", "scheduler", "history",
            "best_f1", "rng_state",
        }
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
        print(f"RESUME AT-SBN: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        merge_active = epoch_number >= merge_start_epoch
        set_study_epoch(loaders.train, epoch)
        model.train()

        totals = {
            "loss": 0.0,
            "primary_ce": 0.0,
            "c0_ce": 0.0,
            "f0_ce": 0.0,
            "w0_ce": 0.0,
            "c0_kd": 0.0,
            "f0_kd": 0.0,
            "w0_kd": 0.0,
            "merge_loss": 0.0,
            "c0_disagreement": 0.0,
            "f0_disagreement": 0.0,
            "w0_disagreement": 0.0,
            "c0_js": 0.0,
            "f0_js": 0.0,
            "w0_js": 0.0,
        }
        samples = 0
        progress = tqdm(loaders.train, desc=f"AT-SBN epoch {epoch_number}/{epochs}")

        for images, labels in progress:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            raw_input = runtimes["R0"](images)
            raw = model.forward_primary(raw_input, labels=labels)
            primary_ce = loss_fn(raw.logits, labels)

            auxiliary_ce: dict[str, Tensor] = {}
            auxiliary_kd: dict[str, Tensor] = {}
            auxiliary_logits: dict[str, Tensor] = {}

            with auxiliary_batchnorm_batch_stats(model.base):
                for arm in AUX_VIEWS:
                    view_input = runtimes[arm](images)
                    view = model.forward_auxiliary(arm, view_input)
                    auxiliary_logits[arm] = view.logits
                    auxiliary_ce[arm] = loss_fn(view.logits, labels)
                    auxiliary_kd[arm] = soft_target_cross_entropy(
                        view.logits,
                        raw.logits,
                        temperature=temperature,
                    )

            merge, distances = classifier_merge_loss(model)
            merge_term = merge_weight * merge if merge_active else merge.detach() * 0.0

            loss = (
                primary_ce
                + aux_ce_weight * sum(auxiliary_ce.values())
                + kd_weight * sum(auxiliary_kd.values())
                + merge_term
            )
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            samples += batch_size
            totals["loss"] += float(loss.item())
            totals["primary_ce"] += float(primary_ce.item())
            totals["merge_loss"] += float(merge.item())

            raw_pred = raw.logits.detach().argmax(1)
            for arm in AUX_VIEWS:
                totals[f"{arm.lower()}_ce"] += float(auxiliary_ce[arm].item())
                totals[f"{arm.lower()}_kd"] += float(auxiliary_kd[arm].item())
                aux_pred = auxiliary_logits[arm].detach().argmax(1)
                totals[f"{arm.lower()}_disagreement"] += float(
                    (aux_pred != raw_pred).sum().item()
                )
                totals[f"{arm.lower()}_js"] += float(
                    _js_divergence(raw.logits, auxiliary_logits[arm]).item()
                    * batch_size
                )

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                raw=f"{primary_ce.item():.4f}",
                f0=f"{auxiliary_ce['F0'].item():.4f}",
                merge=("on" if merge_active else "off"),
            )

        scheduler.step()
        metrics, _, _, _, _ = _evaluate_model(
            model.base,
            loaders.val,
            eval_runtime,
            device,
            loaders.classes,
            hard_groups,
        )
        batches = max(len(loaders.train), 1)
        _, end_distances = classifier_merge_loss(model)
        record = {
            "epoch": epoch_number,
            "loss": totals["loss"] / batches,
            "primary_ce": totals["primary_ce"] / batches,
            "c0_ce": totals["c0_ce"] / batches,
            "f0_ce": totals["f0_ce"] / batches,
            "w0_ce": totals["w0_ce"] / batches,
            "c0_kd": totals["c0_kd"] / batches,
            "f0_kd": totals["f0_kd"] / batches,
            "w0_kd": totals["w0_kd"] / batches,
            "merge_active": merge_active,
            "merge_loss": totals["merge_loss"] / batches,
            "classifier_distance": end_distances,
            "c0_disagreement": totals["c0_disagreement"] / max(samples, 1),
            "f0_disagreement": totals["f0_disagreement"] / max(samples, 1),
            "w0_disagreement": totals["w0_disagreement"] / max(samples, 1),
            "c0_js": totals["c0_js"] / max(samples, 1),
            "f0_js": totals["f0_js"] / max(samples, 1),
            "w0_js": totals["w0_js"] / max(samples, 1),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch_number,
                    "loss": record["loss"],
                    "primary_ce": record["primary_ce"],
                    "f0_ce": record["f0_ce"],
                    "merge_active": merge_active,
                    "macro_f1": metrics["macro_f1"],
                    "hard_class_f1": metrics["hard_class_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                    "f0_disagreement": record["f0_disagreement"],
                    "f0_js": record["f0_js"],
                }
            ),
            flush=True,
        )

        is_best = metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch_number,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "primary_initial_model_state_sha256": model.primary_initial_sha256,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "epoch": epoch_number,
                    "best_f1": best_f1,
                    "weights": "raw_primary",
                    "run_contract_sha256": run_contract_sha256,
                    "primary_initial_model_state_sha256": model.primary_initial_sha256,
                },
                best_path,
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Training selesai tanpa best/last checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    eval_cfg = copy.deepcopy(best["config"])
    eval_cfg["model"]["pretrained"] = False
    eval_model = AuxiliaryTrainingModel(eval_cfg["model"]).to(device)
    eval_model.load_state_dict(best["model"])

    metrics, labels, predictions, probabilities, paths = _evaluate_model(
        eval_model.base,
        loaders.val,
        eval_runtime,
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

    auxiliary_validation = {}
    for arm in AUX_VIEWS:
        auxiliary_validation[arm] = evaluate_auxiliary_view(
            eval_model,
            loaders.val,
            runtimes[arm],
            arm,
            device,
            loaders.classes,
            hard_groups,
        )
    (run_dir / "validation" / "auxiliary_metrics.json").write_text(
        json.dumps(auxiliary_validation, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "primary_initial_model_state_sha256": model.primary_initial_sha256,
    }
