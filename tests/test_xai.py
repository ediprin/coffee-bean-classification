import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from bilinear_lmmd.run_xai_analysis import (
    METRIC_NAMES,
    _aggregate_selected,
    pair_prediction_rows,
    select_rows,
)
from bilinear_lmmd.run_final_hbp_xai import (
    pair_final_predictions,
    select_final_xai_rows,
)
from bilinear_lmmd.xai import (
    analyze_explanation,
    attention_overlap,
    colorize_cam,
    explain_tensor,
)


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.second = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.third = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        for layer in (self.first, self.second, self.third):
            nn.init.constant_(layer.weight, 0.05)

    def forward(self, image):
        first = F.relu(self.first(image))
        second = F.relu(self.second(F.avg_pool2d(first, 2)))
        third = F.relu(self.third(F.avg_pool2d(second, 2)))
        return [first, second, third]


class _TinyPool(nn.Module):
    def forward(self, features):
        return torch.cat(
            [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in features],
            dim=1,
        )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _TinyEncoder()
        self.pool = _TinyPool()
        self.dropout = nn.Identity()
        self.classifier = nn.Linear(12, 3, bias=False)
        self.classifier_type = "linear"
        with torch.no_grad():
            self.classifier.weight[0].fill_(1.0)
            self.classifier.weight[1].fill_(0.5)
            self.classifier.weight[2].fill_(-0.5)

    def forward(self, image):
        features = self.encoder(image)
        logits = self.classifier(self.pool(features))
        return SimpleNamespace(logits=logits)


def test_finer_layercam_uses_three_layers_and_gamma_zero_matches_layercam():
    model = _TinyModel().eval()
    image = torch.linspace(0.0, 1.0, 3 * 16 * 16).reshape(1, 3, 16, 16)

    explanation = explain_tensor(model, image, target=0, top_k=2, gamma=0.0)

    assert explanation.references == [1, 2]
    assert explanation.layercam.shape == (16, 16)
    assert explanation.finer_layercam.min() >= 0.0
    assert explanation.finer_layercam.max() <= 1.0
    assert np.allclose(explanation.layercam, explanation.finer_layercam, atol=1e-5)


def test_attention_overlap_and_deletion_metrics_are_finite():
    cam = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 2:7] = True
    cam[mask] = 1.0

    overlap = attention_overlap(cam, mask)

    assert overlap["foreground_mass"] == pytest.approx(1.0)
    assert overlap["background_leakage"] == pytest.approx(0.0)
    assert overlap["foreground_lift"] == pytest.approx(4.0)

    model = _TinyModel().eval()
    image = torch.ones(1, 3, 16, 16)
    _, metrics = analyze_explanation(
        model,
        image,
        np.ones((16, 16), dtype=bool),
        target=0,
        top_k=2,
        gamma=0.6,
        deletion_fraction=0.05,
    )
    for method in ("layercam", "finer_layercam"):
        assert np.isfinite(metrics[method]["relative_confidence_drop"])
        assert metrics[method]["deletion_fraction"] == pytest.approx(0.05)


def _write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "actual", "predicted", "correct", "prob::A", "prob::B"]
        )
        writer.writerows(rows)


def test_prediction_pairing_labels_all_complementarity_outcomes(tmp_path):
    m1 = tmp_path / "m1.csv"
    m5 = tmp_path / "m5.csv"
    _write_predictions(
        m1,
        [
            ["/old/a.jpg", "A", "B", 0, 0.2, 0.8],
            ["/old/b.jpg", "A", "A", 1, 0.8, 0.2],
            ["/old/c.jpg", "A", "A", 1, 0.8, 0.2],
            ["/old/d.jpg", "A", "B", 0, 0.2, 0.8],
        ],
    )
    _write_predictions(
        m5,
        [
            ["/new/a.jpg", "A", "A", 1, 0.8, 0.2],
            ["/new/b.jpg", "A", "B", 0, 0.2, 0.8],
            ["/new/c.jpg", "A", "A", 1, 0.8, 0.2],
            ["/new/d.jpg", "A", "B", 0, 0.2, 0.8],
        ],
    )

    classes, rows = pair_prediction_rows(m1, m5)

    assert classes == ["A", "B"]
    assert {row["outcome"] for row in rows} == {
        "rescued",
        "negative_transfer",
        "both_correct",
        "both_wrong",
    }
    first = select_rows(rows, samples_per_outcome=1, selection_seed=42)
    second = select_rows(rows, samples_per_outcome=1, selection_seed=42)
    assert first == second


def test_xai_aggregate_preserves_outcome_paired_model_delta():
    def model_result(value):
        metrics = {metric: value for metric in METRIC_NAMES}
        return {
            "metrics": {
                "layercam": dict(metrics),
                "finer_layercam": dict(metrics),
            }
        }

    aggregate = _aggregate_selected(
        [{"models": {"M1": model_result(0.2), "M5w01": model_result(0.5)}}]
    )

    assert aggregate["selected_samples"] == 1
    assert aggregate["delta_M5w01_vs_M1"]["finer_layercam"][
        "foreground_mass"
    ] == pytest.approx(0.3)


def test_cam_heatmap_is_rgb_and_preserves_spatial_size():
    heatmap = colorize_cam(np.linspace(0, 1, 35).reshape(5, 7))
    assert heatmap.mode == "RGB"
    assert heatmap.size == (7, 5)


def test_final_hbp_pairing_and_selection_are_deterministic(tmp_path):
    m0 = tmp_path / "m0.csv"
    m1 = tmp_path / "m1.csv"
    _write_predictions(
        m0,
        [
            ["/old/a.jpg", "A", "B", 0, 0.2, 0.8],
            ["/old/b.jpg", "A", "A", 1, 0.8, 0.2],
            ["/old/c.jpg", "A", "A", 1, 0.8, 0.2],
            ["/old/d.jpg", "A", "B", 0, 0.2, 0.8],
        ],
    )
    _write_predictions(
        m1,
        [
            ["/new/a.jpg", "A", "A", 1, 0.8, 0.2],
            ["/new/b.jpg", "A", "B", 0, 0.2, 0.8],
            ["/new/c.jpg", "A", "A", 1, 0.8, 0.2],
            ["/new/d.jpg", "A", "B", 0, 0.2, 0.8],
        ],
    )

    classes, rows = pair_final_predictions(m0, m1)
    first = select_final_xai_rows(rows, 1, 42)
    second = select_final_xai_rows(rows, 1, 42)

    assert classes == ["A", "B"]
    assert {row["outcome"] for row in rows} == {
        "rescued_by_hbp",
        "harmed_by_hbp",
        "both_correct",
        "both_wrong",
    }
    assert first == second
