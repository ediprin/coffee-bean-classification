import ast
import copy
import json
from pathlib import Path

import torch

from bilinear_lmmd.engine.at_sbn import AuxiliaryTrainingModel
from bilinear_lmmd.engine.mvsc_mvfd_sbn import (
    r0_anchor_multiview_supcon,
    validate_mvsc_mvfd_sbn_config,
)


def _model_cfg():
    return {
        "backbone": "mobilenetv3_large_100",
        "pretrained": False,
        "head": "gap",
        "classifier": "linear",
        "num_classes": 17,
        "out_indices": [3, 4],
        "dropout": 0.2,
    }


def _full_cfg():
    return {
        "seed": 42,
        "data": {
            "augmentation_mode": "preprocessing_study",
            "object_crop": False,
        },
        "model": _model_cfg(),
        "adaptation": {"method": "source_only"},
        "training": {
            "epochs": 50,
            "classification_loss": "cross_entropy",
            "ema_decay": 0.0,
        },
        "preprocessing": {"code": "R0"},
        "feature_distillation": {
            "method": "mvsc_mvfd_sbn",
            "views": ["R0", "C0", "F0", "W0"],
            "primary_view": "R0",
            "teacher_views": ["C0", "F0", "W0"],
            "teacher_aggregation": "equal_mean",
            "teacher_stop_gradient": True,
            "selective_batch_norm": True,
            "aux_ce_weight_each": 0.05,
            "feature_distill_weight": 0.007,
            "feature_distill_loss": "squared_l2",
            "mid_contrastive_weight": 0.05,
            "mid_contrastive_loss": "r0_anchor_multiview_supcon",
            "mid_contrastive_temperature": 0.07,
            "mid_contrastive_anchor": "R0",
            "mid_contrastive_keys": ["C0", "F0", "W0"],
            "mid_feature_position": 0,
            "mid_backbone_out_index": 3,
            "auxiliary_dropout": False,
            "frontends": {
                "R0": {"code": "R0"},
                "C0": {"code": "C0"},
                "F0": {"code": "F0"},
                "W0": {"code": "W0"},
            },
        },
    }


def test_mvsc_detaches_transformed_keys():
    raw = torch.randn(4, 8, 7, 7, requires_grad=True)
    teacher = {
        arm: torch.randn(4, 8, 7, 7, requires_grad=True)
        for arm in ("C0", "F0", "W0")
    }
    labels = torch.tensor([0, 1, 0, 1])

    loss, diagnostics = r0_anchor_multiview_supcon(
        raw,
        teacher,
        labels,
        temperature=0.07,
    )
    loss.backward()

    assert raw.grad is not None
    assert all(feature.grad is None for feature in teacher.values())
    assert diagnostics["positive_keys_mean"] >= 3.0
    assert diagnostics["negative_keys_mean"] > 0.0


def test_mvsc_rewards_class_aligned_mid_features():
    labels = torch.tensor([0, 1])
    teacher_base = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
        ]
    )
    teacher = {
        arm: teacher_base.clone().requires_grad_(True)
        for arm in ("C0", "F0", "W0")
    }
    aligned = teacher_base.clone().requires_grad_(True)
    swapped = teacher_base.flip(0).clone().requires_grad_(True)

    aligned_loss, aligned_diag = r0_anchor_multiview_supcon(
        aligned,
        teacher,
        labels,
        temperature=0.07,
    )
    swapped_loss, swapped_diag = r0_anchor_multiview_supcon(
        swapped,
        teacher,
        labels,
        temperature=0.07,
    )

    assert aligned_loss < swapped_loss
    assert aligned_diag["cosine_gap"] > swapped_diag["cosine_gap"]


def test_mvsc_config_is_frozen():
    cfg = _full_cfg()
    validate_mvsc_mvfd_sbn_config(cfg)

    broken = copy.deepcopy(cfg)
    broken["feature_distillation"]["mid_contrastive_temperature"] = 0.1
    try:
        validate_mvsc_mvfd_sbn_config(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Temperature selain 0.07 harus ditolak untuk v1.")


def test_auxiliary_wrapper_exposes_same_final_embedding_shape():
    torch.manual_seed(42)
    model = AuxiliaryTrainingModel(_model_cfg())
    x = torch.randn(2, 3, 224, 224)
    features = model.base.encoder(x)
    assert len(features) == 2
    embedding = model.base.pool(features)
    assert embedding.shape[0] == 2
    assert embedding.shape[1] == model.base.classifier.in_features


def test_kaggle_notebook_materialization_signature_matches_repo_api():
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "Coffee17_MVSC_MVFD_SBN_Kaggle.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "materialize_preprocessing_development"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 4
    assert [kw.arg for kw in call.keywords] == ["fold"]
