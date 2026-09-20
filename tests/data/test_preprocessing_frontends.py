from __future__ import annotations

import pytest
import torch

from bilinear_lmmd.data.preprocessing import (
    AF2Config,
    AF2Frontend,
    ARM_CODES,
    CLAHEConfig,
    CLAHEFrontend,
    RawFrontend,
    WAV1Frontend,
    af2_entropy_threshold,
    build_preprocessing_frontend,
    haar_dwt2,
    haar_idwt2,
    preprocessing_spec,
)


def test_registry_freezes_four_primary_arms():
    assert ARM_CODES == ("R0", "C0", "F0", "W0")
    assert preprocessing_spec("C0")["config"] == {
        "clip_limit": 3.0,
        "tile_grid_size": [8, 8],
    }
    assert preprocessing_spec("F0")["config"]["patch_size"] == 32
    assert preprocessing_spec("F0")["config"]["overlap"] == 0.50
    assert preprocessing_spec("F0")["config"]["gamma"] == 0.10
    assert preprocessing_spec("F0")["config"]["angular_bins"] == 360
    assert preprocessing_spec("W0")["config"]["wavelet_levels"] == 2


def test_raw_is_exact_identity_and_parameter_free():
    image = torch.rand(2, 3, 64, 64)
    module = RawFrontend()
    assert torch.equal(module(image), image)
    assert not list(module.parameters())
    assert not module.state_dict()


def test_clahe_reference_config_and_contract():
    pytest.importorskip("cv2")
    config = CLAHEConfig.from_mapping({"clip_limit": 3.0, "tile_grid_size": [8, 8]})
    assert config.clip_limit == 3.0
    assert config.tile_grid_size == (8, 8)
    torch.manual_seed(19)
    image = torch.rand(1, 3, 64, 64)
    module = CLAHEFrontend(config)
    first = module(image)
    second = module(image)
    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert not list(module.parameters())
    assert not module.state_dict()


def test_af2_entropy_threshold_matches_canonical_equation():
    probability = torch.full((1, 1, 4), 0.25)
    threshold = af2_entropy_threshold(probability, gamma=0.1)
    assert torch.allclose(threshold, torch.tensor([[0.08]]), atol=1e-7, rtol=0.0)


def test_af2_suppresses_low_density_direction():
    module = AF2Frontend(AF2Config())
    m = module.config.patch_size
    center = m // 2
    frequency = torch.zeros(1, 1, m, m, dtype=torch.complex64)
    frequency[0, 0, center, center + 4] = 10.0 + 0.0j
    frequency[0, 0, center + 4, center] = 0.001 + 0.0j
    weight = module._af2_weight(frequency)
    assert weight[0, 0, center, center + 4] > 0.99
    assert weight[0, 0, center + 4, center] == 0.0


def test_af2_forward_is_deterministic_finite_and_canonical_range():
    torch.manual_seed(7)
    image = torch.rand(1, 3, 65, 71)
    module = AF2Frontend()
    first = module(image)
    second = module(image)
    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert torch.isfinite(first).all()
    assert torch.all(first + 1e-7 >= image)
    assert float(first.max()) <= 2.0 + 1e-6
    assert not list(module.parameters())
    assert not module.state_dict()


def test_haar_reconstruction_and_constant_detail_response():
    torch.manual_seed(8)
    image = torch.rand(2, 3, 13, 15)
    bands, shape = haar_dwt2(image)
    reconstructed = haar_idwt2(bands, shape)
    assert torch.allclose(reconstructed, image, atol=2.0e-6, rtol=2.0e-6)
    constant_bands, _ = haar_dwt2(torch.ones(1, 1, 16, 16))
    assert constant_bands[:, :, 1:].abs().max().item() < 1.0e-6


def test_wav1_is_deterministic_finite_active_and_parameter_free():
    torch.manual_seed(3)
    image = torch.rand(1, 3, 65, 63)
    module = WAV1Frontend()
    first = module(image)
    second = module(image)
    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert torch.isfinite(first).all()
    assert torch.all(first + 1e-7 >= image)
    assert float(first.max()) <= 2.0 + 1e-6
    assert not torch.equal(first, image)
    assert not list(module.parameters())
    assert not module.state_dict()


@pytest.mark.parametrize("code", ARM_CODES)
def test_registry_builds_parameter_free_frontends(code):
    module = build_preprocessing_frontend(code)
    assert not list(module.parameters())
    assert not module.state_dict()
