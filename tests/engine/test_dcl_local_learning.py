from __future__ import annotations

import torch

from bilinear_lmmd.engine.dcl_local_learning import (
    _local_permutation,
    deterministic_adaptive_avg_pool2d,
    location_targets,
    region_confusion_batch,
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
