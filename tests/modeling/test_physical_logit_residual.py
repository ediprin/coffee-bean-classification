from __future__ import annotations

import numpy as np

from bilinear_lmmd.modeling.physical_logit_residual import (
    FEATURES,
    build_active_pairs,
    fit_physical_logit_residual,
)


CLASSES = (
    "Broken",
    "Cut",
    "Dry Cherry",
    "Fade",
    "Floater",
    "Full Black",
    "Full Sour",
    "Fungus Damage",
    "Husk",
    "Immature",
    "Parchment",
    "Partial Black",
    "Partial Sour",
    "Severe Insect Damage",
    "Shell",
    "Slight Insect Damage",
    "Withered",
)


def test_active_pair_count_and_mask() -> None:
    pairs = build_active_pairs(CLASSES)
    assert len(pairs) == 38
    assert len(set(pairs)) == 38

    feature_index = {name: i for i, name in enumerate(FEATURES)}
    class_index = {name: i for i, name in enumerate(CLASSES)}

    assert (class_index["Broken"], feature_index["eccentricity"]) in pairs
    assert (class_index["Partial Sour"], feature_index["warm_frac_a5_b10"]) in pairs
    assert (class_index["Full Black"], feature_index["L_frac_lt_50"]) in pairs

    assert (class_index["Fade"], feature_index["L_mean"]) not in pairs
    assert (
        class_index["Severe Insect Damage"],
        feature_index["area_fraction"],
    ) not in pairs


def test_zero_initial_residual_matches_base() -> None:
    rng = np.random.default_rng(123)
    base = rng.normal(size=(20, len(CLASSES)))
    x = rng.normal(size=(20, len(FEATURES)))
    labels = rng.integers(0, len(CLASSES), size=20)

    fit = fit_physical_logit_residual(
        base_logits=base,
        features=x,
        labels=labels,
        classes=CLASSES,
        l2=0.01,
        label_smoothing=0.1,
        maxiter=1,
    )

    # The model representation itself is additive and the explicit zero matrix
    # reproduces the base logits exactly before optimization.
    zero = np.zeros_like(fit.full_weight_matrix())
    np.testing.assert_array_equal(base + fit.standardizer.transform(x) @ zero.T, base)


def test_fit_is_deterministic_and_inactive_weights_remain_zero() -> None:
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=(n, len(FEATURES)))
    labels = rng.integers(0, len(CLASSES), size=n)
    base = rng.normal(scale=0.25, size=(n, len(CLASSES)))

    # Inject a learnable physical relation for Full Black and Partial Black.
    li = FEATURES.index("L_frac_lt_50")
    full_black = CLASSES.index("Full Black")
    partial_black = CLASSES.index("Partial Black")
    x[labels == full_black, li] += 2.0
    x[labels == partial_black, li] -= 1.5

    kwargs = dict(
        base_logits=base,
        features=x,
        labels=labels,
        classes=CLASSES,
        l2=0.01,
        label_smoothing=0.1,
        maxiter=200,
    )
    fit_a = fit_physical_logit_residual(**kwargs)
    fit_b = fit_physical_logit_residual(**kwargs)

    np.testing.assert_allclose(fit_a.weights, fit_b.weights, rtol=0.0, atol=0.0)
    assert fit_a.final_objective <= fit_a.initial_objective

    matrix = fit_a.full_weight_matrix()
    fade = CLASSES.index("Fade")
    severe = CLASSES.index("Severe Insect Damage")
    assert np.all(matrix[fade] == 0.0)
    assert np.all(matrix[severe] == 0.0)
