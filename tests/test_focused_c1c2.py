from __future__ import annotations

from copy import deepcopy

from bilinear_lmmd.experiments.run_focused_c1c2_v1 import (
    _common_protocol_signature,
    _paired_outcomes_historical,
)


def _cfg() -> dict:
    return {
        "seed": 42,
        "device": "auto",
        "data": {
            "root": "data/a",
            "source": "source",
            "train_split": "train",
            "val_split": "val",
            "image_size": 224,
            "batch_size": 32,
            "workers": 4,
            "rotation_angles": [0, 45, 90, 135, 180, 225, 270],
            "augmentation_mode": "preprocessing_study",
            "object_crop": False,
        },
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": True,
            "classifier": "linear",
            "num_classes": 17,
            "projection_dim": 512,
            "dropout": 0.2,
        },
        "adaptation": {"method": "source_only"},
        "training": {
            "epochs": 50,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "classification_loss": "cross_entropy",
            "label_smoothing": 0.1,
            "freeze_backbone": False,
            "scheduler": "cosine",
            "ema_decay": 0.0,
        },
    }


def test_common_protocol_signature_ignores_paths_workers_and_new_sections() -> None:
    left = _cfg()
    right = deepcopy(left)
    right["data"]["root"] = "/tmp/other"
    right["data"]["workers"] = 99
    right["training"]["output_dir"] = "outputs/new"
    right["focused_efficiency"] = {"new": "candidate-only"}
    assert _common_protocol_signature(left) == _common_protocol_signature(right)


def test_common_protocol_signature_detects_training_change() -> None:
    left = _cfg()
    right = deepcopy(left)
    right["training"]["lr"] = 1e-3
    assert _common_protocol_signature(left) != _common_protocol_signature(right)


def test_paired_historical_matches_by_class_filename_not_absolute_root() -> None:
    anchor = [
        {"path": "/old/root/A/img1.jpg", "actual": "A", "correct": "0"},
        {"path": "/old/root/B/img2.jpg", "actual": "B", "correct": "1"},
        {"path": "/old/root/C/img3.jpg", "actual": "C", "correct": "1"},
    ]
    candidate = [
        {"path": "/new/root/C/img3.jpg", "actual": "C", "correct": "0"},
        {"path": "/new/root/A/img1.jpg", "actual": "A", "correct": "1"},
        {"path": "/new/root/B/img2.jpg", "actual": "B", "correct": "1"},
    ]
    result = _paired_outcomes_historical(anchor, candidate)
    assert result == {
        "count": 3,
        "rescue": 1,
        "damage": 1,
        "both_correct": 1,
        "both_wrong": 0,
        "net_correct": 0,
    }
