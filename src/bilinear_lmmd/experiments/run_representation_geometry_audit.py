from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from timm.data import resolve_model_data_config
from tqdm import tqdm

from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_evaluation_loader,
    identity_from_path,
)
from bilinear_lmmd.engine.train import classification_metrics, resolve_device


PROTOCOL = "coffee17-representation-geometry-audit-v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int) -> dict:
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_available": torch.cuda.is_available(),
    }


def _l2_normalize_numpy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _class_centroids(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    centroids = []
    for class_id in range(num_classes):
        selected = features[labels == class_id]
        if len(selected) == 0:
            raise RuntimeError(f"Class {class_id} kosong pada training split.")
        centroid = selected.mean(axis=0, keepdims=True)
        centroids.append(_l2_normalize_numpy(centroid)[0])
    return np.stack(centroids)


def _hard_groups(pairs: list[list[str]]) -> dict[str, list[str]]:
    return {
        f"pair_{index + 1}_{left}_vs_{right}": [left, right]
        for index, (left, right) in enumerate(pairs)
    }


def _validate_config(cfg: dict) -> None:
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol harus {PROTOCOL}.")
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("V1 dikunci seed=42.")
    data = cfg.get("data", {})
    if int(data.get("image_size", -1)) != 224:
        raise ValueError("V1 dikunci image_size=224.")
    if bool(data.get("object_crop", False)):
        raise ValueError("V1 memakai raw RGB tanpa object crop.")
    if int(cfg["evaluation"].get("knn_k", -1)) != 5:
        raise ValueError("V1 dikunci kNN k=5.")
    if float(cfg["evaluation"].get("linear_probe_c", -1)) != 1.0:
        raise ValueError("V1 dikunci LogisticRegression C=1.0.")
    if cfg["evaluation"].get("embedding_normalization") != "l2":
        raise ValueError("V1 memakai L2-normalized embeddings.")
    expected = {
        "mobilenetv3_imagenet": "mobilenetv3_large_100",
        "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
        "convnext_tiny_imagenet": "convnext_tiny.fb_in22k_ft_in1k",
    }
    observed = {
        item["name"]: item["model_name"]
        for item in cfg.get("encoders", [])
    }
    if observed != expected:
        raise ValueError(
            "Encoder V1 harus tepat MobileNetV3/ImageNet, DINOv2-S/14, "
            "dan ConvNeXt-Tiny/ImageNet."
        )


def _preflight_models(cfg: dict) -> dict[str, bool]:
    status = {}
    for spec in cfg["encoders"]:
        model_name = str(spec["model_name"])
        status[model_name] = bool(timm.is_model(model_name))
        if not status[model_name]:
            token = model_name.split(".")[0]
            nearby = timm.list_models(f"*{token}*")[:20]
            raise RuntimeError(
                f"Model timm tidak tersedia: {model_name}. Kandidat dekat: {nearby}"
            )
    return status


def _build_encoder(model_name: str, device: torch.device):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    )
    model.eval().to(device)
    model.requires_grad_(False)
    data_cfg = resolve_model_data_config(model)
    mean = torch.tensor(data_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
    return model, mean.to(device), std.to(device), data_cfg


@torch.inference_mode()
def _extract(
    *,
    model,
    loader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    for images, targets in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        normalized = (images - mean) / std
        embedding = model(normalized)
        if isinstance(embedding, (tuple, list)):
            if len(embedding) != 1:
                raise RuntimeError(
                    "Encoder menghasilkan tuple/list multi-output yang tidak didukung."
                )
            embedding = embedding[0]
        if embedding.ndim > 2:
            embedding = embedding.flatten(1)
        if embedding.ndim != 2:
            raise RuntimeError(
                f"Embedding harus matriks [B,D], observed={tuple(embedding.shape)}"
            )
        features.append(embedding.float().cpu().numpy())
        labels.append(targets.numpy())

    matrix = np.concatenate(features, axis=0)
    target = np.concatenate(labels, axis=0).astype(np.int64)
    paths = [identity_from_path(path) for path, _ in loader.dataset.samples]
    if len(paths) != matrix.shape[0] or len(paths) != target.shape[0]:
        raise RuntimeError("Path/feature/label count tidak cocok.")
    return matrix, target, paths


def _decoder_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    *,
    k: int,
    linear_c: float,
    max_iter: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        weights="uniform",
    )
    knn.fit(train_x, train_y)
    knn_pred = knn.predict(val_x).astype(np.int64)
    _, neighbor_indices = knn.kneighbors(val_x, n_neighbors=k)
    neighbor_labels = train_y[neighbor_indices]

    centroids = _class_centroids(
        train_x,
        train_y,
        int(train_y.max()) + 1,
    )
    centroid_similarity = val_x @ centroids.T
    centroid_pred = centroid_similarity.argmax(axis=1).astype(np.int64)

    linear = LogisticRegression(
        C=linear_c,
        max_iter=max_iter,
        solver="lbfgs",
    )
    linear.fit(train_x, train_y)
    linear_pred = linear.predict(val_x).astype(np.int64)

    return (
        {
            "knn5": knn_pred,
            "nearest_centroid": centroid_pred,
            "linear_probe": linear_pred,
        },
        neighbor_labels,
        centroid_similarity,
    )


def _pair_geometry(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    classes: list[str],
    pairs: list[list[str]],
) -> dict:
    class_to_id = {name: index for index, name in enumerate(classes)}
    centroids = _class_centroids(train_x, train_y, len(classes))
    payload = {}
    for left, right in pairs:
        if left not in class_to_id or right not in class_to_id:
            raise ValueError(f"Hard pair tidak ditemukan: {left}, {right}")
        a = class_to_id[left]
        b = class_to_id[right]
        xa = train_x[train_y == a]
        xb = train_x[train_y == b]
        ca = centroids[a]
        cb = centroids[b]
        between = float(1.0 - np.dot(ca, cb))
        within_a = float(np.mean(1.0 - xa @ ca))
        within_b = float(np.mean(1.0 - xb @ cb))
        pooled_within = (within_a + within_b) / 2.0
        ratio = between / max(pooled_within, 1e-12)
        payload[f"{left}__vs__{right}"] = {
            "left": left,
            "right": right,
            "centroid_cosine_distance": between,
            "left_mean_within_cosine_distance": within_a,
            "right_mean_within_cosine_distance": within_b,
            "inter_to_pooled_intra_ratio": ratio,
        }
    return payload


def _pair_confusions(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
    pairs: list[list[str]],
) -> dict:
    class_to_id = {name: index for index, name in enumerate(classes)}
    payload = {}
    for left, right in pairs:
        a, b = class_to_id[left], class_to_id[right]
        a_to_b = int(np.sum((labels == a) & (predictions == b)))
        b_to_a = int(np.sum((labels == b) & (predictions == a)))
        payload[f"{left}__vs__{right}"] = {
            f"{left}->{right}": a_to_b,
            f"{right}->{left}": b_to_a,
            "TOTAL": a_to_b + b_to_a,
        }
    return payload


def _write_sample_diagnostics(
    *,
    path: Path,
    paths: list[str],
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    neighbor_labels: np.ndarray,
    centroid_similarity: np.ndarray,
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    k = neighbor_labels.shape[1]
    for index, identity in enumerate(paths):
        target = int(labels[index])
        sims = centroid_similarity[index]
        own_sim = float(sims[target])
        rival_sims = sims.copy()
        rival_sims[target] = -np.inf
        rival = int(rival_sims.argmax())
        purity = float(np.mean(neighbor_labels[index] == target))
        row = {
            "identity": identity,
            "actual": classes[target],
            "knn5_predicted": classes[int(predictions["knn5"][index])],
            "nearest_centroid_predicted": classes[
                int(predictions["nearest_centroid"][index])
            ],
            "linear_probe_predicted": classes[int(predictions["linear_probe"][index])],
            "knn_same_class_fraction": purity,
            "own_centroid_cosine_similarity": own_sim,
            "nearest_rival_class": classes[rival],
            "nearest_rival_cosine_similarity": float(sims[rival]),
            "centroid_similarity_margin": own_sim - float(sims[rival]),
            "neighbor_labels": "|".join(
                classes[int(class_id)] for class_id in neighbor_labels[index]
            ),
            "all_three_correct": int(
                all(int(pred[index]) == target for pred in predictions.values())
            ),
            "all_three_wrong": int(
                all(int(pred[index]) != target for pred in predictions.values())
            ),
        }
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_audit(
    *,
    config_path: Path,
    data_root: Path,
    output_dir: Path,
    device_name: str,
) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _validate_config(cfg)
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())

    seed = int(cfg["seed"])
    determinism = configure_determinism(seed)
    device = resolve_device(device_name)
    model_preflight = _preflight_models(cfg)

    train_loader, train_classes = build_preprocessing_evaluation_loader(
        cfg["data"],
        split=str(cfg["data"].get("train_split", "train")),
        seed=seed,
    )
    val_loader, val_classes = build_preprocessing_evaluation_loader(
        cfg["data"],
        split=str(cfg["data"].get("val_split", "val")),
        seed=seed,
    )
    if train_classes != val_classes:
        raise RuntimeError("Urutan kelas train/val berbeda.")
    classes = train_classes
    pairs = cfg["evaluation"]["hard_pairs"]
    hard_groups = _hard_groups(pairs)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_payload = {
        "format": "bilinear_lmmd.representation_geometry.fold.v1",
        "protocol": PROTOCOL,
        "outer_test_accessed": False,
        "classes": classes,
        "determinism": determinism,
        "model_preflight": model_preflight,
        "encoders": {},
    }

    for spec in cfg["encoders"]:
        name = str(spec["name"])
        model_name = str(spec["model_name"])
        print(f"\n=== ENCODER {name}: {model_name} ===", flush=True)
        model, mean, std, model_data_cfg = _build_encoder(model_name, device)

        train_features, train_labels, _ = _extract(
            model=model,
            loader=train_loader,
            mean=mean,
            std=std,
            device=device,
            desc=f"{name} train",
        )
        val_features, val_labels, val_paths = _extract(
            model=model,
            loader=val_loader,
            mean=mean,
            std=std,
            device=device,
            desc=f"{name} val",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_features = _l2_normalize_numpy(train_features.astype(np.float64))
        val_features = _l2_normalize_numpy(val_features.astype(np.float64))

        predictions, neighbor_labels, centroid_similarity = _decoder_predictions(
            train_features,
            train_labels,
            val_features,
            k=int(cfg["evaluation"]["knn_k"]),
            linear_c=float(cfg["evaluation"]["linear_probe_c"]),
            max_iter=int(cfg["evaluation"]["linear_probe_max_iter"]),
        )

        metrics = {
            decoder: classification_metrics(
                val_labels.tolist(),
                pred.tolist(),
                classes,
                hard_groups,
            )
            for decoder, pred in predictions.items()
        }
        pair_geometry = _pair_geometry(
            train_x=train_features,
            train_y=train_labels,
            classes=classes,
            pairs=pairs,
        )
        pair_confusions = {
            decoder: _pair_confusions(
                val_labels,
                pred,
                classes,
                pairs,
            )
            for decoder, pred in predictions.items()
        }

        encoder_dir = output_dir / name
        _write_json(encoder_dir / "metrics.json", metrics)
        _write_json(encoder_dir / "pair_geometry.json", pair_geometry)
        _write_json(encoder_dir / "pair_confusions.json", pair_confusions)
        _write_sample_diagnostics(
            path=encoder_dir / "sample_diagnostics.csv",
            paths=val_paths,
            labels=val_labels,
            predictions=predictions,
            neighbor_labels=neighbor_labels,
            centroid_similarity=centroid_similarity,
            classes=classes,
        )

        fold_payload["encoders"][name] = {
            "model_name": model_name,
            "embedding_dim": int(train_features.shape[1]),
            "train_count": int(train_features.shape[0]),
            "val_count": int(val_features.shape[0]),
            "input_image_size_used": int(cfg["data"]["image_size"]),
            "model_default_input_size": list(model_data_cfg.get("input_size", [])),
            "metrics": metrics,
            "pair_geometry": pair_geometry,
            "pair_confusions": pair_confusions,
        }

    _write_json(output_dir / "audit_result.json", fold_payload)
    print(json.dumps(fold_payload, indent=2), flush=True)
    return fold_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_audit(
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
