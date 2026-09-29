from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
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
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-dcl-local-learning-v1"
ARMS = ("HBP_CE", "HBP_DCL")


def validate_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("DCL V1 dikunci seed=42.")
    if cfg.get("adaptation", {}).get("method") != "source_only":
        raise ValueError("DCL V1 harus source_only.")

    data = cfg.get("data", {})
    if data.get("augmentation_mode") != "preprocessing_study":
        raise ValueError("augmentation_mode harus preprocessing_study.")
    if int(data.get("image_size", -1)) != 224:
        raise ValueError("image_size harus 224.")
    if int(data.get("batch_size", -1)) != 32:
        raise ValueError("batch_size harus 32.")
    if int(data.get("workers", -1)) != 4:
        raise ValueError("workers harus 4.")
    if data.get("source", "source") != "source":
        raise ValueError("data.source harus source.")
    if data.get("train_split", "train") != "train":
        raise ValueError("train_split harus train.")
    if data.get("val_split", "val") != "val":
        raise ValueError("val_split harus val.")
    if list(data.get("rotation_angles", [])) != [0, 45, 90, 135, 180, 225, 270]:
        raise ValueError("rotation schedule harus canonical preprocessing-study.")
    if bool(data.get("object_crop", False)):
        raise ValueError("object_crop harus false.")

    model = cfg.get("model", {})
    if model.get("backbone") != "mobilenetv3_large_100":
        raise ValueError("Backbone harus MobileNetV3-Large.")
    if model.get("pretrained") is not True:
        raise ValueError("Backbone harus memakai ImageNet pretrained weights.")
    if model.get("head") != "hbp":
        raise ValueError("Head harus HBP.")
    if list(model.get("out_indices", [])) != [1, 3, 4]:
        raise ValueError("HBP out_indices harus [1,3,4].")
    if int(model.get("projection_dim", -1)) != 512:
        raise ValueError("projection_dim harus 512.")
    if model.get("classifier", "linear") != "linear":
        raise ValueError("Classifier harus linear.")
    if int(model.get("num_classes", -1)) != 17:
        raise ValueError("Coffee17 harus 17 kelas.")
    if abs(float(model.get("dropout", -1.0)) - 0.2) > 1e-12:
        raise ValueError("dropout harus 0.2.")

    training = cfg.get("training", {})
    if int(training.get("epochs", -1)) != 50:
        raise ValueError("Training harus 50 epoch.")
    if abs(float(training.get("lr", -1.0)) - 3e-4) > 1e-12:
        raise ValueError("lr harus 3e-4.")
    if abs(float(training.get("weight_decay", -1.0)) - 1e-4) > 1e-12:
        raise ValueError("weight_decay harus 1e-4.")
    if training.get("classification_loss") != "cross_entropy":
        raise ValueError("classification_loss harus cross_entropy.")
    if bool(training.get("freeze_backbone", True)):
        raise ValueError("Backbone harus end-to-end trainable.")
    if abs(float(training.get("label_smoothing", -1.0)) - 0.1) > 1e-12:
        raise ValueError("label_smoothing harus 0.1.")
    if training.get("scheduler") != "cosine":
        raise ValueError("scheduler harus cosine.")
    if float(training.get("ema_decay", 0.0)) != 0.0:
        raise ValueError("EMA tidak dipakai.")
    if int(training.get("ema_start_epoch", -1)) != 0:
        raise ValueError("ema_start_epoch harus 0.")

    dcl = cfg.get("dcl", {})
    if dcl.get("implementation") != "local_rcm_dcl_style":
        raise ValueError("implementation harus local_rcm_dcl_style.")
    if int(dcl.get("grid_size", -1)) != 4:
        raise ValueError("DCL V1 dikunci grid 4x4 untuk input 224.")
    if int(dcl.get("neighbor_span", -1)) != 2:
        raise ValueError("neighbor_span harus 2, mengikuti local RCM DCL.")
    for key in ("classification_weight", "swap_weight", "location_weight"):
        if abs(float(dcl.get(key, -1.0)) - 1.0) > 1e-12:
            raise ValueError(f"{key} harus 1.0 pada V1.")
    if dcl.get("swap_task") != "binary_original_vs_shuffled":
        raise ValueError("swap_task harus binary_original_vs_shuffled.")
    if dcl.get("location_loss") != "l1":
        raise ValueError("location_loss harus l1.")
    if dcl.get("validation_tuning") is not False:
        raise ValueError("DCL V1 tidak mengizinkan validation tuning.")

    expected_pairs = [
        ["Withered", "Immature"],
        ["Severe Insect Damage", "Slight Insect Damage"],
        ["Cut", "Slight Insect Damage"],
        ["Partial Sour", "Full Sour"],
        ["Slight Insect Damage", "Fade"],
        ["Full Black", "Partial Black"],
    ]
    pairs = cfg.get("evaluation", {}).get("targeted_confusion_pairs", [])
    if pairs != expected_pairs:
        raise ValueError("DCL V1 targeted hard pairs berubah dari preregistration.")

    expected_hard_groups = {
        "withered_immature": ["Withered", "Immature"],
        "insect_severity": ["Severe Insect Damage", "Slight Insect Damage"],
        "cut_insect": ["Cut", "Slight Insect Damage"],
        "sour_extent": ["Partial Sour", "Full Sour"],
        "slight_fade": ["Slight Insect Damage", "Fade"],
        "black_extent": ["Full Black", "Partial Black"],
    }
    if cfg.get("evaluation", {}).get("hard_groups", {}) != expected_hard_groups:
        raise ValueError("DCL V1 hard_groups berubah dari preregistration.")


def build_core(cfg: dict) -> nn.Module:
    return build_model(copy.deepcopy(cfg["model"]))


def _tensor_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def core_fingerprint(model: nn.Module) -> str:
    return _tensor_fingerprint(model.state_dict())


def deterministic_adaptive_avg_pool2d(
    x: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Adaptive-average pooling using explicit slice means.

    PyTorch's CUDA backward for adaptive_avg_pool2d is nondeterministic on
    current Kaggle builds. This implements the same adaptive bin boundaries
    with deterministic mean reductions, so strict deterministic training can
    remain enabled.
    """

    if x.ndim != 4:
        raise ValueError("Expected BCHW tensor.")
    out_h, out_w = (int(output_size[0]), int(output_size[1]))
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_size harus positif.")

    height, width = int(x.shape[-2]), int(x.shape[-1])
    rows = []
    for i in range(out_h):
        h0 = (i * height) // out_h
        h1 = ((i + 1) * height + out_h - 1) // out_h
        cols = []
        for j in range(out_w):
            w0 = (j * width) // out_w
            w1 = ((j + 1) * width + out_w - 1) // out_w
            cols.append(x[:, :, h0:h1, w0:w1].mean(dim=(-2, -1)))
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


class DCLTrainingWrapper(nn.Module):
    """Training-only DCL heads around the unchanged HBP inference core."""

    def __init__(self, core: nn.Module, grid_size: int):
        super().__init__()
        self.core = core
        self.grid_size = int(grid_size)
        channels = list(self.core.encoder.feature_info.channels())
        if len(channels) != 3:
            raise RuntimeError("DCL V1 mengharapkan tiga HBP feature stages.")
        embedding_dim = int(self.core.pool.output_dim)

        # Auxiliary initialization must not perturb the RNG stream seen by the
        # shared HBP core relative to the matched control.
        rng_state = capture_rng_state()
        self.swap_classifier = nn.Linear(embedding_dim, 2, bias=False)
        self.location_head = nn.Conv2d(channels[-1], 1, kernel_size=1, bias=True)
        restore_rng_state(rng_state)

    def inference_core(self) -> nn.Module:
        return self.core

    def forward_training(self, normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.core.encoder(normalized)
        embedding = self.core.pool(features)
        class_logits = self.core.classifier(self.core.dropout(embedding))
        swap_logits = self.swap_classifier(embedding)
        location = self.location_head(features[-1])
        location = deterministic_adaptive_avg_pool2d(
            location,
            output_size=(self.grid_size, self.grid_size),
        )
        location = torch.tanh(location).flatten(1)
        return {
            "class_logits": class_logits,
            "swap_logits": swap_logits,
            "location": location,
        }


def build_candidate(cfg: dict) -> DCLTrainingWrapper:
    core = build_core(cfg)
    return DCLTrainingWrapper(core, grid_size=int(cfg["dcl"]["grid_size"]))


@torch.no_grad()
def preflight_matched_initialization(cfg: dict) -> dict:
    validate_config(cfg)
    seed_everything(42)
    control = build_core(cfg)
    control_sha = core_fingerprint(control)

    seed_everything(42)
    candidate = build_candidate(cfg)
    candidate_sha = core_fingerprint(candidate.core)

    if control_sha != candidate_sha:
        raise RuntimeError("Initial HBP core control/DCL berbeda.")

    left = control.state_dict()
    right = candidate.core.state_dict()
    mismatched = [
        key for key in left
        if key not in right or not torch.equal(left[key].cpu(), right[key].cpu())
    ]
    if mismatched:
        raise RuntimeError(
            "Initial core tensors berbeda: " + ", ".join(mismatched[:10])
        )

    control.eval()
    candidate.eval()
    raw = torch.linspace(
        0.0, 1.0, steps=2 * 3 * 224 * 224, dtype=torch.float32
    ).reshape(2, 3, 224, 224)
    normalized = imagenet_normalize(raw)
    control_logits = control(normalized).logits
    candidate_core_logits = candidate.core(normalized).logits
    candidate_training_outputs = candidate.forward_training(normalized)
    candidate_training_logits = candidate_training_outputs["class_logits"]

    core_max_abs = float(
        (control_logits - candidate_core_logits).abs().max().item()
    )
    training_path_max_abs = float(
        (control_logits - candidate_training_logits).abs().max().item()
    )
    if core_max_abs > 1e-7:
        raise RuntimeError(
            f"Initial control/DCL inference logits berbeda: {core_max_abs}"
        )
    if training_path_max_abs > 1e-7:
        raise RuntimeError(
            "Initial control/DCL classification training path berbeda: "
            f"{training_path_max_abs}"
        )
    if tuple(candidate_training_outputs["swap_logits"].shape) != (2, 2):
        raise RuntimeError("DCL swap head preflight shape bukan [2,2].")
    expected_locations = int(cfg["dcl"]["grid_size"]) ** 2
    if tuple(candidate_training_outputs["location"].shape) != (
        2,
        expected_locations,
    ):
        raise RuntimeError(
            "DCL location head preflight shape salah: "
            f"{tuple(candidate_training_outputs['location'].shape)}"
        )

    return {
        "initial_core_state_sha256": control_sha,
        "matched_core_tensor_equality": True,
        "initial_logit_max_abs_difference": core_max_abs,
        "training_path_logit_max_abs_difference": training_path_max_abs,
        "control_inference_parameter_count": sum(p.numel() for p in control.parameters()),
        "candidate_inference_parameter_count": sum(
            p.numel() for p in candidate.core.parameters()
        ),
        "candidate_training_only_parameter_count": sum(
            p.numel() for p in candidate.parameters()
        ) - sum(p.numel() for p in candidate.core.parameters()),
    }


def _local_permutation(
    grid_size: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Local RCM: adjacent shuffles horizontally then vertically (span=2)."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    grid = torch.arange(grid_size * grid_size, dtype=torch.long).reshape(
        grid_size, grid_size
    )

    for row in range(grid_size):
        for col in range(1, grid_size):
            if bool(torch.randint(0, 2, (1,), generator=generator).item()):
                tmp = grid[row, col - 1].clone()
                grid[row, col - 1] = grid[row, col]
                grid[row, col] = tmp

    for row in range(1, grid_size):
        if bool(torch.randint(0, 2, (1,), generator=generator).item()):
            tmp = grid[row - 1].clone()
            grid[row - 1] = grid[row]
            grid[row] = tmp

    permutation = grid.flatten()
    identity = torch.arange(grid_size * grid_size, dtype=torch.long)
    if torch.equal(permutation, identity):
        # A "shuffled" sample must actually be destructed; otherwise the
        # binary auxiliary target would be contradictory.
        permutation = permutation.clone()
        permutation[0], permutation[1] = permutation[1].clone(), permutation[0].clone()
    return permutation


def region_confusion_batch(
    images: torch.Tensor,
    *,
    grid_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError("RCM input harus BCHW.")
    batch, channels, height, width = images.shape
    if height % grid_size or width % grid_size:
        raise ValueError("Image size harus habis dibagi grid_size.")
    ph, pw = height // grid_size, width // grid_size
    patches = (
        images.unfold(2, ph, ph)
        .unfold(3, pw, pw)
        .contiguous()
        .reshape(batch, channels, grid_size * grid_size, ph, pw)
    )

    permutations = torch.stack(
        [
            _local_permutation(
                grid_size,
                seed=int(seed) + sample_index * 1009,
            )
            for sample_index in range(batch)
        ],
        dim=0,
    )
    shuffled = []
    for sample_index in range(batch):
        perm = permutations[sample_index].to(images.device)
        shuffled.append(patches[sample_index].index_select(1, perm))
    shuffled_tensor = torch.stack(shuffled, dim=0)
    shuffled_tensor = (
        shuffled_tensor.reshape(batch, channels, grid_size, grid_size, ph, pw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(batch, channels, height, width)
    )
    return shuffled_tensor, permutations


def location_targets(
    permutations: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = grid_size * grid_size
    identity = torch.arange(count, dtype=torch.float32).unsqueeze(0)
    identity = identity.repeat(permutations.shape[0], 1)
    shuffled = permutations.to(dtype=torch.float32)
    center = count // 2
    identity = (identity - center) / count
    shuffled = (shuffled - center) / count
    return identity.to(device), shuffled.to(device)


def validate_evaluation_classes(classes: list[str], cfg: dict) -> None:
    configured = set()
    for pair in cfg["evaluation"]["targeted_confusion_pairs"]:
        configured.update(pair)
    for members in cfg["evaluation"]["hard_groups"].values():
        configured.update(members)
    unknown = sorted(configured.difference(classes))
    if unknown:
        raise ValueError(
            "Nama kelas evaluation tidak ditemukan pada Coffee17 runtime: "
            + ", ".join(unknown)
        )


@torch.no_grad()
def _evaluate(
    core: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
) -> tuple[dict, list[int], list[int], list[list[float]], list[str]]:
    core.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for images, targets in loader:
        raw = images.to(device, non_blocking=True)
        output = core(imagenet_normalize(raw))
        probs = output.logits.softmax(1).cpu()
        labels.extend(targets.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())

    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah path validation berbeda dari prediksi.")
    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


def _build_arm(cfg: dict, arm: str) -> nn.Module:
    if arm == "HBP_CE":
        return build_core(cfg)
    if arm == "HBP_DCL":
        return build_candidate(cfg)
    raise ValueError(f"Arm tidak dikenal: {arm}")


def _core_from_arm(model: nn.Module, arm: str) -> nn.Module:
    return model if arm == "HBP_CE" else model.core


def gpu_training_smoke_test(
    cfg: dict,
    *,
    device: str = "cuda:0",
) -> dict:
    """Run one strict-deterministic forward/backward for both arms on CUDA.

    This is intentionally executed before any 50-epoch arm so unsupported
    deterministic CUDA operators fail in seconds rather than after the matched
    control has already finished.
    """

    validate_config(cfg)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("GPU smoke test membutuhkan strict determinism aktif.")

    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise RuntimeError("GPU smoke test harus dijalankan pada CUDA.")

    seed = int(cfg["seed"])
    batch = 2
    raw_cpu = torch.linspace(
        0.0,
        1.0,
        steps=batch * 3 * 224 * 224,
        dtype=torch.float32,
    ).reshape(batch, 3, 224, 224)
    labels_cpu = torch.tensor([0, 1], dtype=torch.long)
    reports: dict[str, dict] = {}

    for arm in ARMS:
        seed_everything(seed)
        model = _build_arm(cfg, arm).to(resolved)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        raw = raw_cpu.to(resolved)
        labels = labels_cpu.to(resolved)
        optimizer.zero_grad(set_to_none=True)

        if arm == "HBP_CE":
            logits = model(imagenet_normalize(raw)).logits
            loss = nn.CrossEntropyLoss(
                label_smoothing=float(cfg["training"]["label_smoothing"])
            )(logits, labels)
            components = {"classification": float(loss.detach().cpu().item())}
        else:
            grid_size = int(cfg["dcl"]["grid_size"])
            shuffled, permutations = region_confusion_batch(
                raw,
                grid_size=grid_size,
                seed=seed * 1_000_003,
            )
            combined = torch.cat((raw, shuffled), dim=0)
            combined_labels = torch.cat((labels, labels), dim=0)
            outputs = model.forward_training(imagenet_normalize(combined))

            class_loss = nn.CrossEntropyLoss(
                label_smoothing=float(cfg["training"]["label_smoothing"])
            )(outputs["class_logits"], combined_labels)
            swap_targets = torch.cat(
                (
                    torch.ones(batch, dtype=torch.long, device=resolved),
                    torch.zeros(batch, dtype=torch.long, device=resolved),
                ),
                dim=0,
            )
            swap_loss = nn.CrossEntropyLoss()(
                outputs["swap_logits"], swap_targets
            )
            identity_loc, shuffled_loc = location_targets(
                permutations,
                grid_size=grid_size,
                device=resolved,
            )
            location_loss = nn.L1Loss()(
                outputs["location"],
                torch.cat((identity_loc, shuffled_loc), dim=0),
            )
            loss = (
                float(cfg["dcl"]["classification_weight"]) * class_loss
                + float(cfg["dcl"]["swap_weight"]) * swap_loss
                + float(cfg["dcl"]["location_weight"]) * location_loss
            )
            components = {
                "classification": float(class_loss.detach().cpu().item()),
                "swap": float(swap_loss.detach().cpu().item()),
                "location": float(location_loss.detach().cpu().item()),
            }

        loss.backward()

        if arm == "HBP_DCL":
            required_grads = {
                "swap_classifier": model.swap_classifier.weight.grad,
                "location_head": model.location_head.weight.grad,
                "core_classifier": model.core.classifier.weight.grad,
            }
            missing_required = [
                name
                for name, grad in required_grads.items()
                if grad is None or not torch.isfinite(grad).all()
            ]
            if missing_required:
                raise RuntimeError(
                    "GPU smoke DCL: auxiliary/core gradient tidak valid: "
                    + ", ".join(missing_required)
                )

        finite_gradients = 0
        trainable_with_grad = 0
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            trainable_with_grad += 1
            if torch.isfinite(parameter.grad).all():
                finite_gradients += 1
        if trainable_with_grad == 0 or finite_gradients != trainable_with_grad:
            raise RuntimeError(
                f"GPU smoke {arm}: gradient non-finite atau tidak terbentuk "
                f"({finite_gradients}/{trainable_with_grad})."
            )

        optimizer.step()
        torch.cuda.synchronize(resolved)

        reports[arm] = {
            "loss": float(loss.detach().cpu().item()),
            "components": components,
            "gradient_tensors": int(trainable_with_grad),
            "all_gradients_finite": True,
            "optimizer_step_passed": True,
        }
        del model, optimizer, raw, labels, loss
        torch.cuda.empty_cache()

    return {
        "device": str(resolved),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "arms": reports,
        "passed": True,
    }


def train_arm(
    cfg: dict,
    *,
    arm: str,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_core_sha256: str,
    resume: bool,
) -> dict:
    validate_config(cfg)
    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}.")

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas dataset dan model berbeda.")
    validate_evaluation_classes(loaders.classes, cfg)

    model = _build_arm(cfg, arm)
    core = _core_from_arm(model, arm)
    initial_sha = core_fingerprint(core)
    if initial_sha != expected_initial_core_sha256:
        raise RuntimeError(
            f"Initial core {arm} berbeda dari preflight: "
            f"{initial_sha} != {expected_initial_core_sha256}"
        )
    model = model.to(device)
    core = _core_from_arm(model, arm)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
    )
    swap_loss_fn = nn.CrossEntropyLoss()
    location_loss_fn = nn.L1Loss()
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError(f"Checkpoint {arm} berasal dari kontrak berbeda.")
        if checkpoint.get("classes") != loaders.classes:
            raise RuntimeError(f"Urutan kelas checkpoint {arm} berbeda.")
        required = {
            "model", "optimizer", "scheduler", "history",
            "best_f1", "rng_state",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint {arm} tidak lengkap: {missing}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME {arm}: epoch {start_epoch + 1}/{epochs}", flush=True)

    dcl_cfg = cfg["dcl"]
    grid_size = int(dcl_cfg["grid_size"])

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        set_study_epoch(loaders.train, epoch)
        model.train()
        running = {
            "total": 0.0,
            "classification": 0.0,
            "swap": 0.0,
            "location": 0.0,
        }
        batches = 0
        progress = tqdm(loaders.train, desc=f"{arm} epoch {epoch_number}/{epochs}")

        for batch_index, (images, labels) in enumerate(progress):
            raw = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if arm == "HBP_CE":
                output = model(imagenet_normalize(raw))
                class_loss = class_loss_fn(output.logits, labels)
                swap_loss = class_loss.new_zeros(())
                location_loss = class_loss.new_zeros(())
                loss = class_loss
            else:
                shuffle_seed = (
                    seed * 1_000_003
                    + epoch * 10_007
                    + batch_index * 101
                )
                shuffled, permutations = region_confusion_batch(
                    raw,
                    grid_size=grid_size,
                    seed=shuffle_seed,
                )
                combined = torch.cat((raw, shuffled), dim=0)
                combined_labels = torch.cat((labels, labels), dim=0)
                outputs = model.forward_training(imagenet_normalize(combined))

                class_loss = class_loss_fn(
                    outputs["class_logits"], combined_labels
                )
                swap_targets = torch.cat(
                    (
                        torch.ones(raw.shape[0], dtype=torch.long, device=device),
                        torch.zeros(raw.shape[0], dtype=torch.long, device=device),
                    ),
                    dim=0,
                )
                swap_loss = swap_loss_fn(outputs["swap_logits"], swap_targets)

                identity_loc, shuffled_loc = location_targets(
                    permutations,
                    grid_size=grid_size,
                    device=device,
                )
                loc_targets = torch.cat((identity_loc, shuffled_loc), dim=0)
                location_loss = location_loss_fn(
                    outputs["location"], loc_targets
                )

                loss = (
                    float(dcl_cfg["classification_weight"]) * class_loss
                    + float(dcl_cfg["swap_weight"]) * swap_loss
                    + float(dcl_cfg["location_weight"]) * location_loss
                )

            loss.backward()
            optimizer.step()

            running["total"] += float(loss.item())
            running["classification"] += float(class_loss.item())
            running["swap"] += float(swap_loss.item())
            running["location"] += float(location_loss.item())
            batches += 1
            progress.set_postfix(
                total=f"{loss.item():.4f}",
                cls=f"{class_loss.item():.4f}",
            )

        scheduler.step()
        metrics, _, _, _, _ = _evaluate(
            core,
            loaders.val,
            device,
            loaders.classes,
            hard_groups,
        )
        record = {
            "epoch": epoch_number,
            "loss": {
                key: value / max(batches, 1)
                for key, value in running.items()
            },
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "epoch": epoch_number,
                    "loss": record["loss"],
                    "macro_f1": metrics["macro_f1"],
                    "hard_class_f1": metrics["hard_class_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )

        is_best = float(metrics["macro_f1"]) > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "arm": arm,
            "epoch": epoch_number,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "initial_core_state_sha256": expected_initial_core_sha256,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "inference_core": core.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "arm": arm,
                    "epoch": epoch_number,
                    "best_f1": best_f1,
                    "run_contract_sha256": run_contract_sha256,
                    "initial_core_state_sha256": expected_initial_core_sha256,
                },
                best_path,
            )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError(f"Training {arm} selesai tanpa best/last checkpoint.")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    eval_core = build_core(cfg)
    eval_core.load_state_dict(best["inference_core"])
    eval_core = eval_core.to(device)

    metrics, labels, predictions, probabilities, paths = _evaluate(
        eval_core,
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
        "arm": arm,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": float(metrics["macro_f1"]),
        "initial_core_state_sha256": expected_initial_core_sha256,
        "inference_parameter_count": sum(p.numel() for p in eval_core.parameters()),
        "training_parameter_count": sum(p.numel() for p in model.parameters()),
    }
