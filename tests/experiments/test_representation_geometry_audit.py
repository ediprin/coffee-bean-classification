from __future__ import annotations

import numpy as np
import timm

from bilinear_lmmd.experiments.run_representation_geometry_audit import (
    _class_centroids,
    _l2_normalize_numpy,
    _pair_geometry,
)


def test_required_timm_models_exist() -> None:
    assert timm.is_model("mobilenetv3_large_100")
    assert timm.is_model("vit_small_patch14_dinov2.lvd142m")
    assert timm.is_model("convnext_tiny.fb_in22k_ft_in1k")


def test_l2_normalization() -> None:
    x = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    y = _l2_normalize_numpy(x)
    np.testing.assert_allclose(
        np.linalg.norm(y, axis=1),
        np.ones(2),
        rtol=0.0,
        atol=1e-12,
    )


def test_class_centroids_are_unit_normalized() -> None:
    x = _l2_normalize_numpy(
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
    )
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    centers = _class_centroids(x, y, 2)
    np.testing.assert_allclose(
        np.linalg.norm(centers, axis=1),
        np.ones(2),
        rtol=0.0,
        atol=1e-12,
    )


def test_pair_geometry_rewards_separated_compact_classes() -> None:
    x = _l2_normalize_numpy(
        np.asarray(
            [
                [1.0, 0.02],
                [1.0, -0.02],
                [0.02, 1.0],
                [-0.02, 1.0],
            ],
            dtype=np.float64,
        )
    )
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    result = _pair_geometry(
        train_x=x,
        train_y=y,
        classes=["A", "B"],
        pairs=[["A", "B"]],
    )["A__vs__B"]

    assert result["centroid_cosine_distance"] > 0.9
    assert result["left_mean_within_cosine_distance"] < 0.01
    assert result["right_mean_within_cosine_distance"] < 0.01
    assert result["inter_to_pooled_intra_ratio"] > 100.0
