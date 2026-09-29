from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from bilinear_lmmd.engine.dcl_local_learning import (
    _local_permutation,
    deterministic_adaptive_avg_pool2d,
    gpu_training_smoke_test,
    location_targets,
    region_confusion_batch,
    validate_config,
)
from bilinear_lmmd.engine.physical_logit_residual_wr_hbp import (
    configure_strict_determinism,
)


def test_local_permutation_is_valid_and_deterministic() -> None:
    left = _local_permutation(4, seed=123)
    right = _local_permutation(4, seed=123)
    assert torch.equal(left, right)
    assert sorted(left.tolist()) == list(range(16))


def test_region_confusion_preserves_patch_multiset() -> None:
    # Each 2x2 patch is filled with a unique scalar id.
    image = torch.zeros(1, 1, 8, 8)
    patch_id = 0
    for row in range(4):
        for col in range(4):
            image[:, :, row * 2:(row + 1) * 2, col * 2:(col + 1) * 2] = patch_id
            patch_id += 1

    shuffled, permutations = region_confusion_batch(
        image,
        grid_size=4,
        seed=17,
    )
    assert shuffled.shape == image.shape
    assert permutations.shape == (1, 16)
    assert sorted(permutations[0].tolist()) == list(range(16))

    observed = []
    for row in range(4):
        for col in range(4):
            patch = shuffled[0, 0, row * 2:(row + 1) * 2, col * 2:(col + 1) * 2]
            assert torch.all(patch == patch.flatten()[0])
            observed.append(int(patch.flatten()[0].item()))
    assert observed == permutations[0].tolist()


def test_location_targets_match_identity_and_permutation() -> None:
    permutations = torch.tensor(
        [[1, 0, 2, 3,
          5, 4, 6, 7,
          8, 9, 10, 11,
          12, 13, 14, 15]],
        dtype=torch.long,
    )
    identity, shuffled = location_targets(
        permutations,
        grid_size=4,
        device=torch.device("cpu"),
    )
    expected_identity = (torch.arange(16, dtype=torch.float32) - 8) / 16
    expected_shuffled = (permutations[0].float() - 8) / 16
    assert torch.allclose(identity[0], expected_identity)
    assert torch.allclose(shuffled[0], expected_shuffled)


def test_region_confusion_requires_divisible_grid() -> None:
    image = torch.zeros(2, 3, 225, 224)
    try:
        region_confusion_batch(image, grid_size=4, seed=1)
    except ValueError as exc:
        assert "habis dibagi" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-divisible image size.")


def test_deterministic_adaptive_pool_matches_pytorch_forward() -> None:
    x = torch.arange(2 * 3 * 7 * 7, dtype=torch.float32).reshape(2, 3, 7, 7)
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
    observed = deterministic_adaptive_avg_pool2d(x, (4, 4))
    assert torch.allclose(observed, expected, rtol=0.0, atol=1e-6)


def test_deterministic_adaptive_pool_has_backward() -> None:
    x = torch.randn(2, 3, 7, 7, requires_grad=True)
    y = deterministic_adaptive_avg_pool2d(x, (4, 4)).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_local_permutation_is_never_identity() -> None:
    identity = torch.arange(16)
    for seed in range(256):
        permutation = _local_permutation(4, seed=seed)
        assert not torch.equal(permutation, identity)


def test_deterministic_adaptive_pool_backward_matches_pytorch_cpu() -> None:
    base = torch.arange(2 * 3 * 7 * 7, dtype=torch.float32).reshape(2, 3, 7, 7)
    left = base.clone().requires_grad_(True)
    right = base.clone().requires_grad_(True)
    weights = torch.linspace(0.1, 1.0, steps=2 * 3 * 4 * 4).reshape(2, 3, 4, 4)

    (deterministic_adaptive_avg_pool2d(left, (4, 4)) * weights).sum().backward()
    (torch.nn.functional.adaptive_avg_pool2d(right, (4, 4)) * weights).sum().backward()

    assert left.grad is not None
    assert right.grad is not None
    assert torch.allclose(left.grad, right.grad, rtol=0.0, atol=1e-7)


def _locked_config() -> dict:
    path = Path("configs/dcl_local_learning/DCL_LOCAL_V1.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_locked_dcl_config_validates() -> None:
    validate_config(_locked_config())


def test_config_rejects_changed_hard_pair() -> None:
    cfg = _locked_config()
    cfg["evaluation"]["targeted_confusion_pairs"][0][0] = "Changed"
    with pytest.raises(ValueError, match="preregistration"):
        validate_config(cfg)


def test_config_requires_pretrained_backbone() -> None:
    cfg = _locked_config()
    cfg["model"]["pretrained"] = False
    with pytest.raises(ValueError, match="pretrained"):
        validate_config(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_strict_deterministic_training_smoke() -> None:
    configure_strict_determinism(42)
    report = gpu_training_smoke_test(_locked_config(), device="cuda:0")
    assert report["passed"] is True
    assert report["deterministic_algorithms"] is True
    for arm in ("HBP_CE", "HBP_DCL"):
        assert report["arms"][arm]["all_gradients_finite"] is True
        assert report["arms"][arm]["optimizer_step_passed"] is True
