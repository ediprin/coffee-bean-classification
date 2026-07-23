import torch

from bilinear_lmmd.data.region_confusion import (
    normalized_linear_layout,
    region_confusion,
)


def _patch_ids(grid_size: int = 3, patch_size: int = 2) -> torch.Tensor:
    image = torch.empty(1, 1, grid_size * patch_size, grid_size * patch_size)
    for row in range(grid_size):
        for column in range(grid_size):
            index = row * grid_size + column
            image[
                :,
                :,
                row * patch_size : (row + 1) * patch_size,
                column * patch_size : (column + 1) * patch_size,
            ] = float(index)
    return image


def test_region_confusion_preserves_pixels_and_tracks_exact_layout():
    generator = torch.Generator().manual_seed(7)
    images = _patch_ids()
    batch = region_confusion(images, grid_size=3, generator=generator)

    assert batch.images.shape == images.shape
    assert torch.equal(
        images.flatten().sort().values,
        batch.images.flatten().sort().values,
    )
    assert sorted(batch.permutations[0].tolist()) == list(range(9))
    base = normalized_linear_layout(3)
    assert torch.equal(batch.original_layout[0], base)
    assert torch.equal(
        batch.confused_layout[0],
        base[batch.permutations[0]],
    )

    observed = batch.images.reshape(1, 1, 3, 2, 3, 2)
    observed = observed[:, :, :, 0, :, 0].flatten().to(torch.long)
    assert torch.equal(observed, batch.permutations[0])


def test_region_confusion_rejects_non_divisible_image():
    images = torch.randn(2, 3, 10, 12)
    try:
        region_confusion(images, grid_size=3)
    except ValueError as exc:
        assert "habis dibagi" in str(exc)
    else:
        raise AssertionError("Ukuran non-divisible harus ditolak.")
