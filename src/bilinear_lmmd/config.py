from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "root": "data/coffee",
        "source": "source",
        "target": "target",
        "train_split": "train",
        "val_split": "val",
        "image_size": 224,
        "batch_size": 32,
        "workers": 4,
        "rotation_angles": [0, 45, 90, 135, 180, 225, 270],
    },
    "model": {
        "backbone": "mobilenetv3_large_100",
        "pretrained": True,
        "head": "hbp",
        "num_classes": 17,
        "out_indices": [1, 3, 4],
        "projection_dim": 512,
        "hbp_spatial_size": 14,
        "hbp_mlp_dim": 672,
        "fusion_hbp_dim": 512,
        "fusion_gap_dim": 256,
        "residual_control_dim": 80,
        "residual_gap_dim": 128,
        "attention_reduction": 16,
        "moe_local_dim": 256,
        "moe_gate_hidden": 32,
        "moe_hbp_prior": 0.8,
        "dropout": 0.2,
        "classifier": "linear",
        "arcface_scale": 30.0,
        "arcface_margin": 0.3,
        "enable_domain_classifier": False,
        "hierarchy_num_parents": 0,
    },
    "adaptation": {
        "method": "lmmd",
        "weight": 1.0,
        "warmup_epochs": 5,
        "confidence_threshold": 0.0,
        "kernel_mul": 2.0,
        "kernel_num": 5,
    },
    "training": {
        "epochs": 50,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "label_smoothing": 0.1,
        "expert_aux_weight": 0.3,
        "expert_diversity_weight": 0.05,
        "output_dir": "outputs/default",
    },
    "hierarchy": {
        "enabled": False,
        "weight": 0.0,
        "groups": {},
    },
    "evaluation": {
        "hard_groups": {
            "sour_black": ["Partial Black", "Partial Sour", "Full Sour"],
            "shape_withered": ["Withered", "Immature", "Cut"],
            "insect_damage": ["Slight Insect Damage", "Severe Insect Damage"],
        }
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "hard_groups":
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    return _merge(DEFAULTS, override)
