import torch

from bilinear_lmmd.analysis.nuisance import (
    NUISANCE_KINDS,
    apply_acquisition_nuisance,
    cosine_stability,
    fisher_separability,
    normalized_l2_shift,
)


def test_all_nuisances_preserve_shape_and_range():
    x = torch.linspace(0.0, 1.0, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    for kind in NUISANCE_KINDS:
        y = apply_acquisition_nuisance(x, kind, seed=123)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0


def test_noise_is_deterministic_for_fixed_seed():
    x = torch.full((2, 3, 8, 8), 0.5)
    a = apply_acquisition_nuisance(x, "gaussian_noise", seed=17)
    b = apply_acquisition_nuisance(x, "gaussian_noise", seed=17)
    c = apply_acquisition_nuisance(x, "gaussian_noise", seed=18)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_embedding_metrics_identity():
    x = torch.randn(7, 13)
    assert abs(cosine_stability(x, x) - 1.0) < 1.0e-6
    assert normalized_l2_shift(x, x) < 1.0e-7


def test_fisher_separability_increases_for_clear_clusters():
    labels = torch.tensor([0, 0, 1, 1])
    mixed = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    separated = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    assert fisher_separability(separated, labels) > fisher_separability(mixed, labels)
