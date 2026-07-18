from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from bilinear_lmmd.data.attribute_features import load_rgb, segment_bean
from bilinear_lmmd.modeling.models import AdaptationModel, build_model
from bilinear_lmmd.engine.train import resolve_device


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Explanation:
    probabilities: np.ndarray
    prediction: int
    target: int
    references: list[int]
    layercam: np.ndarray
    finer_layercam: np.ndarray


def normalize_cam(cam: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Normalize each CAM in a batch to [0, 1]."""
    flat = cam.flatten(1)
    minimum = flat.amin(1).view(-1, 1, 1, 1)
    maximum = flat.amax(1).view(-1, 1, 1, 1)
    return (cam - minimum) / (maximum - minimum).clamp_min(epsilon)


def _forward_with_features(
    model: AdaptationModel | nn.Module, images: Tensor
) -> tuple[Tensor, list[Tensor]]:
    """Run the ordinary classifier while keeping backbone feature tensors."""
    features = list(model.encoder(images))
    embedding = model.pool(features)
    if getattr(model, "classifier_type", "linear") == "arcface":
        logits = model.classifier(embedding)
    else:
        logits = model.classifier(model.dropout(embedding))
    return logits, features


def multilayer_layercam(
    score: Tensor,
    features: Sequence[Tensor],
    output_size: tuple[int, int],
    retain_graph: bool,
) -> Tensor:
    """LayerCAM on each selected backbone depth, followed by mean fusion.

    LayerCAM weights every activation location by its positive gradient. Each
    depth is independently normalized before upsampling so a high-amplitude
    deep layer cannot erase a spatially detailed shallow layer.
    """
    gradients = torch.autograd.grad(
        score,
        tuple(features),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    maps = []
    for feature, gradient in zip(features, gradients):
        cam = (gradient.relu() * feature).sum(dim=1, keepdim=True).relu()
        cam = normalize_cam(cam)
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        maps.append(normalize_cam(cam))
    return normalize_cam(torch.stack(maps).mean(0))


def explain_tensor(
    model: AdaptationModel | nn.Module,
    image: Tensor,
    target: int | None = None,
    top_k: int = 3,
    gamma: float = 0.6,
) -> Explanation:
    """Return ordinary LayerCAM and class-comparison Finer-LayerCAM.

    Finer-CAM changes *what* is explained: for every reference class ``d`` it
    explains ``y_target - gamma * y_d``. Pairwise maps for the top-k competing
    classes are then averaged, matching the aggregation in Zhang et al. (2025).
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("XAI saat ini menerima tepat satu gambar [1,C,H,W].")
    if top_k < 1:
        raise ValueError("top_k minimal 1.")
    if gamma < 0:
        raise ValueError("gamma tidak boleh negatif.")

    model.eval()
    logits, features = _forward_with_features(model, image)
    probabilities = logits.softmax(1)
    prediction = int(logits.argmax(1).item())
    target = prediction if target is None else int(target)
    if not 0 <= target < logits.shape[1]:
        raise ValueError(f"Indeks target di luar rentang: {target}")

    candidates = logits[0].detach().clone()
    candidates[target] = -torch.inf
    reference_count = min(top_k, logits.shape[1] - 1)
    references = candidates.topk(reference_count).indices.tolist()
    output_size = tuple(int(value) for value in image.shape[-2:])

    layercam = multilayer_layercam(
        logits[0, target], features, output_size, retain_graph=True
    )
    finer_maps = []
    for index, reference in enumerate(references):
        score = logits[0, target] - gamma * logits[0, reference]
        finer_maps.append(
            multilayer_layercam(
                score,
                features,
                output_size,
                retain_graph=index < len(references) - 1,
            )
        )
    finer_layercam = normalize_cam(torch.stack(finer_maps).mean(0))
    return Explanation(
        probabilities=probabilities[0].detach().cpu().numpy(),
        prediction=prediction,
        target=target,
        references=references,
        layercam=layercam[0, 0].detach().cpu().numpy(),
        finer_layercam=finer_layercam[0, 0].detach().cpu().numpy(),
    )


@torch.no_grad()
def deletion_metrics(
    model: AdaptationModel | nn.Module,
    image: Tensor,
    cam: np.ndarray,
    target: int,
    reference: int,
    fraction: float = 0.05,
) -> dict[str, float | bool]:
    """Mask top CAM pixels and compute target and relative confidence drops."""
    if not 0 < fraction < 1:
        raise ValueError("Deletion fraction harus berada di antara 0 dan 1.")
    flat = np.asarray(cam, dtype=np.float32).reshape(-1)
    if float(flat.max()) <= 1e-8:
        return {
            "target_confidence_drop": 0.0,
            "reference_confidence_drop": 0.0,
            "relative_confidence_drop": 0.0,
            "deletion_fraction": fraction,
            "cam_valid": False,
        }
    count = max(1, int(np.ceil(flat.size * fraction)))
    selected = np.argpartition(flat, flat.size - count)[-count:]
    mask = torch.zeros(flat.size, dtype=torch.bool, device=image.device)
    mask[torch.as_tensor(selected, device=image.device)] = True
    mask = mask.view(*image.shape[-2:])

    original = model(image).logits.softmax(1)[0]
    deleted = image.clone()
    # Zero is the ImageNet-normalized mean pixel, a neutral deletion value.
    deleted[0].masked_fill_(mask.unsqueeze(0), 0.0)
    changed = model(deleted).logits.softmax(1)[0]
    target_drop = float((original[target] - changed[target]).item())
    reference_drop = float((original[reference] - changed[reference]).item())
    return {
        "target_confidence_drop": target_drop,
        "reference_confidence_drop": reference_drop,
        "relative_confidence_drop": target_drop - reference_drop,
        "deletion_fraction": fraction,
        "cam_valid": True,
    }


def attention_overlap(
    cam: np.ndarray,
    foreground_mask: np.ndarray,
    top_fraction: float = 0.20,
) -> dict[str, float]:
    """Measure attention mass and top-region overlap inside the bean mask."""
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(foreground_mask, dtype=bool)
    if cam.shape != mask.shape:
        raise ValueError(f"CAM {cam.shape} dan mask {mask.shape} harus sama.")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction harus berada di antara 0 dan 1.")
    total = float(cam.clip(min=0).sum())
    mask_fraction = float(mask.mean())
    foreground_mass = float(cam.clip(min=0)[mask].sum() / total) if total > 0 else 0.0
    foreground_lift = foreground_mass / mask_fraction if mask_fraction > 0 else 0.0

    flat = cam.reshape(-1)
    count = max(1, int(np.ceil(flat.size * top_fraction)))
    indices = np.argpartition(flat, flat.size - count)[-count:]
    top = np.zeros(flat.size, dtype=bool)
    top[indices] = True
    top = top.reshape(cam.shape)
    union = np.logical_or(top, mask).sum()
    top_iou = float(np.logical_and(top, mask).sum() / union) if union else 0.0
    return {
        "foreground_mass": foreground_mass,
        "background_leakage": 1.0 - foreground_mass,
        "mask_area_fraction": mask_fraction,
        "foreground_lift": foreground_lift,
        "top20_iou": top_iou,
    }


def _eval_resize_crop(image: Image.Image, image_size: int, nearest: bool = False) -> Image.Image:
    interpolation = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = TF.resize(
        image,
        int(image_size * 256 / 224),
        interpolation=interpolation,
    )
    return TF.center_crop(resized, [image_size, image_size])


def load_eval_image(path: Path, image_size: int) -> tuple[Tensor, np.ndarray]:
    with Image.open(path) as opened:
        display = _eval_resize_crop(opened.convert("RGB"), image_size)
    tensor = TF.normalize(TF.to_tensor(display), IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0), np.asarray(display, dtype=np.uint8)


def load_eval_foreground_mask(path: Path, image_size: int) -> np.ndarray:
    rgb = load_rgb(path, max_size=512)
    mask = segment_bean(rgb)
    mask_image = Image.fromarray(np.uint8(mask) * 255)
    transformed = _eval_resize_crop(mask_image, image_size, nearest=True)
    return np.asarray(transformed, dtype=np.uint8) > 127


def load_checkpoint_model(
    checkpoint_path: Path, device_name: str = "auto"
) -> tuple[AdaptationModel, dict, torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["model"]["pretrained"] = False
    device = resolve_device(device_name)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, device


def analyze_explanation(
    model: AdaptationModel | nn.Module,
    image: Tensor,
    foreground_mask: np.ndarray,
    target: int,
    top_k: int,
    gamma: float,
    deletion_fraction: float,
) -> tuple[Explanation, dict[str, dict[str, float | bool]]]:
    explanation = explain_tensor(model, image, target=target, top_k=top_k, gamma=gamma)
    reference = explanation.references[0]
    metrics = {}
    for name, cam in (
        ("layercam", explanation.layercam),
        ("finer_layercam", explanation.finer_layercam),
    ):
        metrics[name] = {
            **attention_overlap(cam, foreground_mask),
            **deletion_metrics(
                model,
                image,
                cam,
                target,
                reference,
                fraction=deletion_fraction,
            ),
        }
    return explanation, metrics


def _overlay(display: np.ndarray, cam: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = display.astype(np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = 255.0
    heat[..., 1] = 210.0 * cam
    alpha = (0.58 * cam)[..., None]
    blended = base * (1.0 - alpha) + heat * alpha

    # Green bean boundary makes foreground leakage visually auditable.
    eroded = mask.copy()
    for axis in (0, 1):
        eroded &= np.roll(mask, 1, axis=axis)
        eroded &= np.roll(mask, -1, axis=axis)
    boundary = mask & ~eroded
    blended[boundary] = (40, 255, 70)
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


def colorize_cam(cam: np.ndarray) -> Image.Image:
    """Convert a normalized CAM to a dependency-free blue-to-red heatmap."""
    values = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray(
        [
            [0, 24, 120],
            [0, 190, 255],
            [255, 230, 0],
            [220, 0, 0],
        ],
        dtype=np.float32,
    )
    scaled = values * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    weight = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - weight) + stops[upper] * weight
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))


def render_cam_heatmap_panel(
    destination: Path,
    display: np.ndarray,
    foreground_mask: np.ndarray,
    classes: list[str],
    actual: int,
    outcome: str,
    context: str,
    explanations: dict[str, Explanation],
) -> None:
    """Render raw CAM heatmaps and overlays for an arbitrary model pair."""
    tile_size = display.shape[1], display.shape[0]
    tiles = [Image.fromarray(display)]
    labels = ["INPUT"]
    for model_name, explanation in explanations.items():
        predicted = classes[explanation.prediction]
        for method, cam in (
            ("LayerCAM", explanation.layercam),
            ("Finer-CAM", explanation.finer_layercam),
        ):
            tiles.extend(
                [
                    colorize_cam(cam),
                    _overlay(display, cam, foreground_mask),
                ]
            )
            labels.extend(
                [
                    f"{model_name} {method} heatmap",
                    f"{model_name} {method} overlay | pred={predicted}",
                ]
            )

    title_height = 58
    label_height = 44
    panel = Image.new(
        "RGB",
        (tile_size[0] * len(tiles), title_height + tile_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), f"{context} | {outcome}", fill="black")
    draw.text(
        (8, 29),
        f"actual={classes[actual]} | heatmap: blue=low, red=high | green=bean boundary",
        fill="black",
    )
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x = index * tile_size[0]
        panel.paste(tile, (x, title_height))
        draw.text((x + 5, title_height + tile_size[1] + 8), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination)


def render_comparison_panel(
    destination: Path,
    display: np.ndarray,
    foreground_mask: np.ndarray,
    classes: list[str],
    actual: int,
    outcome: str,
    context: str,
    explanations: dict[str, Explanation],
) -> None:
    tile_size = display.shape[1], display.shape[0]
    tiles = [_overlay(display, np.zeros(foreground_mask.shape), foreground_mask)]
    labels = ["INPUT (green=bean mask)"]
    for model_name in ("M1", "M5w01"):
        explanation = explanations[model_name]
        tiles.extend(
            [
                _overlay(display, explanation.layercam, foreground_mask),
                _overlay(display, explanation.finer_layercam, foreground_mask),
            ]
        )
        predicted = classes[explanation.prediction]
        labels.extend(
            [
                f"{model_name} LayerCAM | pred={predicted}",
                f"{model_name} Finer | pred={predicted}",
            ]
        )

    title_height = 54
    label_height = 42
    panel = Image.new(
        "RGB",
        (tile_size[0] * len(tiles), title_height + tile_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), f"{context} | {outcome}", fill="black")
    draw.text((8, 27), f"actual={classes[actual]} | Finer: gamma comparison, top references", fill="black")
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x = index * tile_size[0]
        panel.paste(tile, (x, title_height))
        draw.text((x + 5, title_height + tile_size[1] + 7), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination)
