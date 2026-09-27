import ast
import json
import copy
from pathlib import Path

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.engine.at_sbn import AuxiliaryTrainingModel
from bilinear_lmmd.engine.hbp_mvfd_sbn import (
    equal_mean_teacher,
    squared_l2_feature_distillation,
    validate_hbp_mvfd_sbn_config,
)
from bilinear_lmmd.modeling.models import build_model


def _model_cfg():
    return {
        "backbone": "mobilenetv3_large_100",
        "pretrained": False,
        "head": "hbp",
        "classifier": "linear",
        "num_classes": 17,
        "out_indices": [1, 3, 4],
        "projection_dim": 512,
        "dropout": 0.2,
    }


def test_hbp_mvfd_config_is_locked():
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs/hbp_mvfd_sbn/HBP_MVFD_SBN_ALL4.yaml")
    validate_hbp_mvfd_sbn_config(cfg)

    broken = copy.deepcopy(cfg)
    broken["model"]["head"] = "gap"
    try:
        validate_hbp_mvfd_sbn_config(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("HBP-MVFD v1 harus menolak head selain HBP.")


def test_hbp_wrapper_keeps_primary_initialization_identical():
    torch.manual_seed(42)
    base = build_model(copy.deepcopy(_model_cfg()))

    torch.manual_seed(42)
    wrapped = AuxiliaryTrainingModel(copy.deepcopy(_model_cfg()))

    from bilinear_lmmd.core.reproducibility import model_state_fingerprint
    assert wrapped.primary_initial_sha256 == model_state_fingerprint(base)


def test_hbp_embedding_dimension_is_1536():
    torch.manual_seed(42)
    model = build_model(copy.deepcopy(_model_cfg())).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.embedding.shape == (2, 1536)
    assert model.classifier.in_features == 1536


def test_equal_mean_hbp_teacher_is_exact_mean():
    embeddings = {
        "C0": torch.tensor([[1.0, 3.0]]),
        "F0": torch.tensor([[3.0, 5.0]]),
        "W0": torch.tensor([[5.0, 7.0]]),
    }
    teacher = equal_mean_teacher(embeddings)
    assert torch.allclose(teacher, torch.tensor([[3.0, 5.0]]))


def test_hbp_feature_distillation_detaches_teacher():
    student = torch.randn(4, 12, requires_grad=True)
    teacher = torch.randn(4, 12, requires_grad=True)
    loss = squared_l2_feature_distillation(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_kaggle_notebook_materialization_signature_matches_repo_api():
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "Coffee17_HBP_MVFD_SBN_Kaggle.ipynb"
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


def test_kaggle_notebook_uses_hbp_result_schema():
    root = Path(__file__).resolve().parents[2]
    path = root / "notebooks" / "Coffee17_HBP_MVFD_SBN_Kaggle.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'result["DELTA_MVFD_ON_HBP"]' in source
    assert 'result["HBP_R0_CONTROL"]' in source
    assert 'result["HBP_MVFD_SBN_ALL4"]' in source
    assert 'result["delta_vs_r0_control"]' not in source
    assert "result['metrics']" not in source
