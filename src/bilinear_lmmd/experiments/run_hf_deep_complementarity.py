from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    capture_rng_state,
    current_git_commit,
    restore_rng_state,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_evaluation_loader,
    build_preprocessing_study_loaders,
    identity_from_path,
    set_study_epoch,
)
from bilinear_lmmd.engine.physical_logit_residual_wr_hbp import (
    configure_strict_determinism,
)
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    classification_metrics,
    resolve_device,
)
from bilinear_lmmd.features.tulsi_handcrafted import (
    FEATURE_NAMES,
    extract_feature_matrix,
    mrmr_select,
)
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-hf-deep-complementarity-v1"
ARMS = ("HF20", "HBP_EMB", "HF20_HBP_EMB")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)
FEATURE_EXTRACTOR_VERSION = "tulsi-aligned-71-v1"


def validate_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("V1 dikunci seed=42.")

    data = cfg.get("data", {})
    expected_data = {
        "augmentation_mode": "preprocessing_study",
        "image_size": 224,
        "batch_size": 32,
        "workers": 4,
        "source": "source",
        "train_split": "train",
        "val_split": "val",
        "object_crop": False,
    }
    for key, expected in expected_data.items():
        value = data.get(key)
        if value != expected:
            raise ValueError(f"data.{key} harus {expected!r}, didapat {value!r}.")
    if list(data.get("rotation_angles", [])) != [0, 45, 90, 135, 180, 225, 270]:
        raise ValueError("rotation_angles berubah dari protokol.")

    model = cfg.get("model", {})
    expected_model = {
        "backbone": "mobilenetv3_large_100",
        "pretrained": True,
        "head": "hbp",
        "classifier": "linear",
        "num_classes": 17,
        "dropout": 0.2,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model.{key} harus {expected!r}.")
    if list(model.get("out_indices", [])) != [1, 3, 4]:
        raise ValueError("HBP out_indices harus [1,3,4].")
    if int(model.get("projection_dim", -1)) != 512:
        raise ValueError("HBP projection_dim harus 512.")

    training = cfg.get("training", {})
    expected_training = {
        "epochs": 50,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "classification_loss": "cross_entropy",
        "label_smoothing": 0.1,
        "scheduler": "cosine",
        "freeze_backbone": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} harus {expected!r}.")

    hf = cfg.get("handcrafted", {})
    expected_hf = {
        "descriptor": "tulsi_aligned_71",
        "selection": "mrmr_difference",
        "selected_features": 20,
        "selection_fit": "train_fold_only",
        "segmentation": "project_background_distance_otsu",
        "augmentation_for_probe": False,
    }
    for key, expected in expected_hf.items():
        if hf.get(key) != expected:
            raise ValueError(f"handcrafted.{key} harus {expected!r}.")

    probe = cfg.get("probe", {})
    expected_probe = {
        "classifier": "multinomial_logistic_regression",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 5000,
        "deep_normalization": "l2",
        "handcrafted_normalization": "train_standardize_then_l2",
        "fusion_block_weighting": "equal_l2_blocks",
        "validation_tuning": False,
    }
    for key, expected in expected_probe.items():
        if probe.get(key) != expected:
            raise ValueError(f"probe.{key} harus {expected!r}.")

    expected_pairs = [
        ["Withered", "Immature"],
        ["Severe Insect Damage", "Slight Insect Damage"],
        ["Cut", "Slight Insect Damage"],
        ["Partial Sour", "Full Sour"],
        ["Slight Insect Damage", "Fade"],
        ["Full Black", "Partial Black"],
    ]
    if cfg.get("evaluation", {}).get("targeted_confusion_pairs") != expected_pairs:
        raise ValueError("targeted_confusion_pairs berubah dari preregistration.")


def _tensor_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_hbp(cfg: dict) -> nn.Module:
    return build_model(dict(cfg["model"]))


def hbp_gpu_smoke_test(cfg: dict, *, device: str) -> dict:
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise RuntimeError("Smoke V1 harus dijalankan pada CUDA.")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Strict determinism belum aktif.")

    seed_everything(int(cfg["seed"]))
    model = build_hbp(cfg).to(resolved)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    batch = int(cfg["data"]["batch_size"])
    raw = torch.linspace(
        0.0, 1.0, steps=batch * 3 * 224 * 224, dtype=torch.float32,
        device=resolved,
    ).reshape(batch, 3, 224, 224)
    labels = torch.arange(batch, device=resolved) % int(cfg["model"]["num_classes"])
    optimizer.zero_grad(set_to_none=True)
    output = model(imagenet_normalize(raw))
    loss = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
    )(output.logits, labels)
    loss.backward()
    grad_tensors = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not grad_tensors or not all(torch.isfinite(g).all() for g in grad_tensors):
        raise RuntimeError("HBP smoke menghasilkan gradient invalid.")
    optimizer.step()
    torch.cuda.synchronize(resolved)
    return {
        "passed": True,
        "device": str(resolved),
        "batch_size": batch,
        "loss": float(loss.detach().cpu().item()),
        "gradient_tensors": len(grad_tensors),
        "optimizer_step_passed": True,
    }


@torch.no_grad()
def _evaluate_hbp(
    model: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for raw, labels in loader:
        raw = raw.to(device, non_blocking=True)
        logits = model(imagenet_normalize(raw)).logits
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(y_true, y_pred, classes, hard_groups)


def _hbp_run_complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return False
    state = torch.load(last, map_location="cpu", weights_only=False)
    return int(state.get("epoch", 0)) >= epochs


def train_hbp(
    cfg: dict,
    *,
    run_dir: Path,
    contract_sha256: str,
    resume: bool,
) -> Path:
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    if len(loaders.classes) != 17:
        raise RuntimeError("Coffee17 runtime harus 17 kelas.")

    model = build_hbp(cfg)
    initial_sha = _tensor_fingerprint(model.state_dict())
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
    )
    hard_groups = cfg["evaluation"]["hard_groups"]

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("contract_sha256") != contract_sha256:
            raise RuntimeError("HBP resume checkpoint berasal dari kontrak berbeda.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = list(state["history"])
        best_f1 = float(state["best_f1"])
        start_epoch = int(state["epoch"])
        restore_rng_state(state["rng_state"])
        print(f"RESUME HBP: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        model.train()
        running = 0.0
        batches = 0
        progress = tqdm(loaders.train, desc=f"HBP epoch {epoch + 1}/{epochs}")
        for raw, labels in progress:
            raw = raw.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(imagenet_normalize(raw))
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics = _evaluate_hbp(model, loaders.val, device, loaders.classes, hard_groups)
        macro = float(metrics["macro_f1"])
        history.append({
            "epoch": epoch + 1,
            "loss": running / max(batches, 1),
            "macro_f1": macro,
            "hard_class_f1": float(metrics["hard_class_f1"]),
            "worst_class_f1": float(metrics["worst_class_f1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        is_best = macro > best_f1
        if is_best:
            best_f1 = macro

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
            "contract_sha256": contract_sha256,
            "initial_state_sha256": initial_sha,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save({
                "inference_core": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "contract_sha256": contract_sha256,
                "initial_state_sha256": initial_sha,
            }, best_path)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    if not best_path.is_file():
        raise RuntimeError("Training HBP selesai tanpa best.pt.")
    return best_path


def _load_hbp_checkpoint(
    cfg: dict,
    checkpoint_path: Path,
    *,
    device: torch.device,
    classes: list[str],
) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("classes") != classes:
        raise RuntimeError("Urutan kelas checkpoint HBP berbeda dari fold runtime.")
    checkpoint_cfg = state.get("config")
    if isinstance(checkpoint_cfg, dict) and isinstance(checkpoint_cfg.get("model"), dict):
        expected = cfg["model"]
        observed = checkpoint_cfg["model"]
        for key in (
            "backbone", "pretrained", "head", "classifier", "num_classes",
            "out_indices", "projection_dim", "dropout",
        ):
            if observed.get(key) != expected.get(key):
                raise RuntimeError(
                    f"External HBP checkpoint model.{key} berbeda: "
                    f"{observed.get(key)!r} != {expected.get(key)!r}"
                )
    weights = state.get("inference_core", state.get("model"))
    if weights is None:
        raise RuntimeError("Checkpoint HBP tidak memiliki inference_core/model.")
    model = build_hbp(cfg)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    return model


def _dataset_identity_and_paths(loader) -> tuple[list[str], list[Path], np.ndarray]:
    identities: list[str] = []
    paths: list[Path] = []
    labels: list[int] = []
    for path, label in loader.dataset.samples:
        paths.append(Path(path))
        identities.append(identity_from_path(path))
        labels.append(int(label))
    return identities, paths, np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def _extract_embeddings(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    chunks = []
    model.eval()
    for raw, _ in loader:
        raw = raw.to(device, non_blocking=True)
        chunks.append(model(imagenet_normalize(raw)).embedding.cpu().numpy())
    result = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("Embedding HBP mengandung nilai non-finite.")
    return result


def _shared_hf_features(
    cache_path: Path,
    *,
    identities: list[str],
    paths: list[Path],
) -> np.ndarray:
    """Persistent identity-keyed HF cache shared across folds.

    Only identities explicitly requested by the current development fold are
    extracted. Missing identities are appended; outer-test identities are
    never discovered or traversed by this function.
    """

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stored_ids: list[str] = []
    stored_matrix = np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        version = str(data["version"].item())
        if version != FEATURE_EXTRACTOR_VERSION:
            raise RuntimeError(
                f"Shared HF cache version berbeda: {version} != "
                f"{FEATURE_EXTRACTOR_VERSION}"
            )
        stored_ids = [str(v) for v in data["identities"].tolist()]
        stored_matrix = np.asarray(data["matrix"], dtype=np.float64)
        if stored_matrix.shape != (len(stored_ids), len(FEATURE_NAMES)):
            raise RuntimeError("Shared HF cache shape invalid.")

    index = {identity: i for i, identity in enumerate(stored_ids)}
    missing = [
        (identity, path)
        for identity, path in zip(identities, paths)
        if identity not in index
    ]
    if missing:
        new_matrix = extract_feature_matrix([path for _, path in missing])
        start = len(stored_ids)
        stored_ids.extend(identity for identity, _ in missing)
        stored_matrix = np.concatenate((stored_matrix, new_matrix), axis=0)
        index.update(
            {identity: start + offset for offset, (identity, _) in enumerate(missing)}
        )
        np.savez_compressed(
            cache_path,
            version=np.asarray(FEATURE_EXTRACTOR_VERSION),
            identities=np.asarray(stored_ids),
            matrix=stored_matrix,
        )

    return np.stack([stored_matrix[index[identity]] for identity in identities], axis=0)


def _cache_matrix(
    path: Path,
    *,
    identities: list[str],
    labels: np.ndarray,
    build,
    version: str,
) -> np.ndarray:
    meta_path = path.with_suffix(".json")
    identity_sha = canonical_json_sha256({
        "identities": identities,
        "labels": labels.tolist(),
        "version": version,
    })
    if path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("identity_sha256") == identity_sha:
            return np.load(path)
    matrix = np.asarray(build(), dtype=np.float64)
    if matrix.shape[0] != len(identities):
        raise RuntimeError("Jumlah row cache tidak cocok dengan identity.")
    np.save(path, matrix)
    meta_path.write_text(json.dumps({
        "identity_sha256": identity_sha,
        "shape": list(matrix.shape),
        "version": version,
    }, indent=2) + "\n", encoding="utf-8")
    return matrix


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> LogisticRegression:
    probe = cfg["probe"]
    model = LogisticRegression(
        C=float(probe["C"]),
        solver=str(probe["solver"]),
        max_iter=int(probe["max_iter"]),
        random_state=int(cfg["seed"]),
    )
    model.fit(x_train, y_train)
    if int(model.n_iter_.max()) >= int(probe["max_iter"]):
        raise RuntimeError("Logistic probe mencapai max_iter; hasil tidak diterima.")
    return model


def _write_predictions(
    path: Path,
    *,
    identities: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "actual", "predicted", "correct"] + [
        f"p::{name}" for name in classes
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for identity, actual, predicted, prob in zip(
            identities, y_true, y_pred, probabilities
        ):
            row = {
                "path": identity,
                "actual": classes[int(actual)],
                "predicted": classes[int(predicted)],
                "correct": "1" if int(actual) == int(predicted) else "0",
            }
            row.update({f"p::{name}": float(prob[i]) for i, name in enumerate(classes)})
            writer.writerow(row)


def _targeted_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    pairs: list[list[str]],
) -> dict[str, int]:
    names_true = [classes[int(v)] for v in y_true]
    names_pred = [classes[int(v)] for v in y_pred]
    counts: dict[str, int] = {}
    for left, right in pairs:
        key = f"{left} <-> {right}"
        counts[key] = sum(
            1 for a, b in zip(names_true, names_pred)
            if (a == left and b == right) or (a == right and b == left)
        )
    counts["TOTAL"] = int(sum(counts.values()))
    return counts


def _paired_outcomes(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_fusion: np.ndarray,
) -> dict[str, int]:
    base_ok = pred_base == y_true
    fusion_ok = pred_fusion == y_true
    rescue = int(np.count_nonzero(~base_ok & fusion_ok))
    damage = int(np.count_nonzero(base_ok & ~fusion_ok))
    return {
        "count": int(y_true.size),
        "rescue": rescue,
        "damage": damage,
        "both_correct": int(np.count_nonzero(base_ok & fusion_ok)),
        "both_wrong": int(np.count_nonzero(~base_ok & ~fusion_ok)),
        "net_correct": rescue - damage,
    }


def run_fold(
    *,
    config_path: Path,
    fold: int,
    data_root: Path,
    output_root: Path,
    required_commit: str,
    device: str,
    resume: bool,
    authorize_training: bool,
    hbp_checkpoint: Path | None,
    shared_hf_cache: Path | None,
) -> dict:
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 wajib diset.")
    if not authorize_training and hbp_checkpoint is None:
        raise RuntimeError("Training HBP memerlukan --authorize-training.")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari scientific pin: {actual_commit} != {required_commit}"
        )

    cfg = load_config(config_path)
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).resolve())
    validate_config(cfg)
    determinism = configure_strict_determinism(int(cfg["seed"]))
    resolved_device = resolve_device(device)

    train_loader, train_classes = build_preprocessing_evaluation_loader(
        cfg["data"], split="train", seed=int(cfg["seed"])
    )
    val_loader, val_classes = build_preprocessing_evaluation_loader(
        cfg["data"], split="val", seed=int(cfg["seed"])
    )
    if train_classes != val_classes or len(train_classes) != 17:
        raise RuntimeError("Class mapping train/val Coffee17 tidak cocok.")
    classes = train_classes

    train_ids, train_paths, y_train = _dataset_identity_and_paths(train_loader)
    val_ids, val_paths, y_val = _dataset_identity_and_paths(val_loader)
    if set(train_ids).intersection(val_ids):
        raise RuntimeError("Identity leakage train/val terdeteksi.")

    run_root = Path(output_root).resolve() / f"fold_{fold}" / "seed42"
    run_root.mkdir(parents=True, exist_ok=True)
    hbp_dir = run_root / "hbp_core"
    feature_dir = run_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    smoke = hbp_gpu_smoke_test(cfg, device=device)
    print(json.dumps({"hbp_gpu_smoke": smoke}, indent=2), flush=True)

    if hbp_checkpoint is None:
        contract = {
            "format": "bilinear_lmmd.hf_deep.hbp_contract.v1",
            "protocol": PROTOCOL,
            "fold": fold,
            "seed": 42,
            "git_commit": actual_commit,
            "model": cfg["model"],
            "training": cfg["training"],
            "train_identities": train_ids,
            "val_identities": val_ids,
            "outer_test_accessed": False,
        }
        contract_sha = canonical_json_sha256(contract)
        contract["contract_sha256"] = contract_sha
        contract_path = hbp_dir / "run_contract.json"
        if contract_path.is_file():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != contract:
                raise RuntimeError("Existing HBP contract berbeda.")
        else:
            hbp_dir.mkdir(parents=True, exist_ok=True)
            contract_path.write_text(
                json.dumps(contract, indent=2) + "\n", encoding="utf-8"
            )
        if not _hbp_run_complete(hbp_dir, int(cfg["training"]["epochs"])):
            hbp_checkpoint = train_hbp(
                cfg,
                run_dir=hbp_dir,
                contract_sha256=contract_sha,
                resume=resume,
            )
        else:
            hbp_checkpoint = hbp_dir / "best.pt"
    else:
        hbp_checkpoint = Path(hbp_checkpoint).resolve()
        if not hbp_checkpoint.is_file():
            raise FileNotFoundError(f"External HBP checkpoint tidak ada: {hbp_checkpoint}")

    hbp_sha = sha256_file(hbp_checkpoint)
    model = _load_hbp_checkpoint(
        cfg, hbp_checkpoint, device=resolved_device, classes=classes
    )

    if shared_hf_cache is not None:
        hf_train = _shared_hf_features(
            shared_hf_cache,
            identities=train_ids,
            paths=train_paths,
        )
        hf_val = _shared_hf_features(
            shared_hf_cache,
            identities=val_ids,
            paths=val_paths,
        )
    else:
        hf_train = _cache_matrix(
            feature_dir / "hf71_train.npy",
            identities=train_ids,
            labels=y_train,
            version=FEATURE_EXTRACTOR_VERSION,
            build=lambda: extract_feature_matrix(train_paths),
        )
        hf_val = _cache_matrix(
            feature_dir / "hf71_val.npy",
            identities=val_ids,
            labels=y_val,
            version=FEATURE_EXTRACTOR_VERSION,
            build=lambda: extract_feature_matrix(val_paths),
        )

    deep_train = _cache_matrix(
        feature_dir / "hbp_train.npy",
        identities=train_ids,
        labels=y_train,
        version=f"hbp-embedding::{hbp_sha}",
        build=lambda: _extract_embeddings(model, train_loader, resolved_device),
    )
    deep_val = _cache_matrix(
        feature_dir / "hbp_val.npy",
        identities=val_ids,
        labels=y_val,
        version=f"hbp-embedding::{hbp_sha}",
        build=lambda: _extract_embeddings(model, val_loader, resolved_device),
    )

    selected = mrmr_select(
        hf_train,
        y_train,
        k=int(cfg["handcrafted"]["selected_features"]),
        random_state=int(cfg["seed"]),
    )
    selected_names = [FEATURE_NAMES[i] for i in selected]

    scaler = StandardScaler()
    hf_train_selected = scaler.fit_transform(hf_train[:, selected])
    hf_val_selected = scaler.transform(hf_val[:, selected])
    hf_train_norm = normalize(hf_train_selected, norm="l2")
    hf_val_norm = normalize(hf_val_selected, norm="l2")
    deep_train_norm = normalize(deep_train, norm="l2")
    deep_val_norm = normalize(deep_val, norm="l2")
    fusion_train = np.concatenate((deep_train_norm, hf_train_norm), axis=1) / math.sqrt(2.0)
    fusion_val = np.concatenate((deep_val_norm, hf_val_norm), axis=1) / math.sqrt(2.0)

    matrices = {
        "HF20": (hf_train_norm, hf_val_norm),
        "HBP_EMB": (deep_train_norm, deep_val_norm),
        "HF20_HBP_EMB": (fusion_train, fusion_val),
    }
    hard_groups = cfg["evaluation"]["hard_groups"]
    pairs = cfg["evaluation"]["targeted_confusion_pairs"]
    arm_metrics: dict[str, dict] = {}
    arm_pred: dict[str, np.ndarray] = {}
    targeted: dict[str, dict] = {}

    for arm in ARMS:
        x_train, x_val = matrices[arm]
        probe = _fit_probe(x_train, y_train, cfg)
        pred = probe.predict(x_val).astype(np.int64)
        prob = probe.predict_proba(x_val)
        metrics = classification_metrics(
            y_val.tolist(), pred.tolist(), classes, hard_groups
        )
        arm_metrics[arm] = {key: float(metrics[key]) for key in METRICS}
        arm_pred[arm] = pred
        targeted[arm] = _targeted_confusions(y_val, pred, classes, pairs)
        _write_predictions(
            run_root / arm / "predictions.csv",
            identities=val_ids,
            y_true=y_val,
            y_pred=pred,
            probabilities=prob,
            classes=classes,
        )
        (run_root / arm / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            run_root / arm / "probe.npz",
            coef=probe.coef_,
            intercept=probe.intercept_,
            classes=probe.classes_,
        )

    delta = {
        key: arm_metrics["HF20_HBP_EMB"][key] - arm_metrics["HBP_EMB"][key]
        for key in METRICS
    }
    targeted_delta = {
        key: targeted["HF20_HBP_EMB"][key] - targeted["HBP_EMB"][key]
        for key in targeted["HBP_EMB"]
    }
    outcomes = _paired_outcomes(
        y_val, arm_pred["HBP_EMB"], arm_pred["HF20_HBP_EMB"]
    )

    selection_record = {
        "selected_indices_zero_based": selected,
        "selected_feature_names": selected_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fit_scope": "train_fold_only",
    }
    (run_root / "feature_selection.json").write_text(
        json.dumps(selection_record, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "format": "bilinear_lmmd.hf_deep.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": fold,
        "seed": 42,
        "git_commit": actual_commit,
        "classes": classes,
        "train_count": int(y_train.size),
        "validation_count": int(y_val.size),
        "train_validation_identity_overlap": 0,
        "hbp_checkpoint": str(hbp_checkpoint),
        "hbp_checkpoint_sha256": hbp_sha,
        "hbp_gpu_smoke": smoke,
        "strict_determinism": determinism,
        "selected_feature_names": selected_names,
        "arms": arm_metrics,
        "delta_fusion_minus_deep": delta,
        "targeted_confusions": targeted,
        "targeted_confusion_delta_fusion_minus_deep": targeted_delta,
        "paired_outcomes_fusion_vs_deep": outcomes,
        "outer_test_accessed": False,
    }
    (run_root / "fold_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--hbp-checkpoint", type=Path)
    parser.add_argument(
        "--shared-hf-cache",
        type=Path,
        help="Optional identity-keyed Tulsi71 cache shared across folds.",
    )
    args = parser.parse_args()
    run_fold(
        config_path=args.config,
        fold=args.fold,
        data_root=args.data_root,
        output_root=args.output_root,
        required_commit=args.required_commit,
        device=args.device,
        resume=args.resume,
        authorize_training=args.authorize_training,
        hbp_checkpoint=args.hbp_checkpoint,
        shared_hf_cache=args.shared_hf_cache,
    )


if __name__ == "__main__":
    main()
