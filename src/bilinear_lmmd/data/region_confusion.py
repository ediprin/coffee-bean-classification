from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RegionConfusionBatch:
    images: Tensor
    original_layout: Tensor
    confused_layout: Tensor
    permutations: Tensor


def normalized_linear_layout(
    grid_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the scalar region-location target used by DCL.

    Chen et al. encode each cell with its centered linear grid index divided by
    the number of cells. Keeping this one-dimensional law makes the auxiliary
    objective directly comparable with the official implementation.
    """

    if grid_size <= 1:
        raise ValueError("grid_size harus lebih besar dari satu.")
    count = grid_size * grid_size
    midpoint = count // 2
    return (
        torch.arange(count, device=device, dtype=dtype) - float(midpoint)
    ) / float(count)


def _local_permutation(
    grid_size: int,
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> Tensor:
    """Shuffle neighboring patches and rows like the official DCL RCM.

    The authors' implementation walks through every row and repeatedly
    shuffles the last two neighboring patches, then applies the same local
    operation to neighboring rows.  We express the operation directly on
    integer patch identities so the layout target is exact.
    """

    # Keep RNG operations on CPU because rcm_generator is intentionally a CPU
    # generator whose state can be serialized and resumed on any device.
    order = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    for row in range(grid_size):
        for column in range(1, grid_size):
            if int(torch.randint(0, 2, (), generator=generator).item()):
                left = order[row, column - 1].clone()
                order[row, column - 1] = order[row, column]
                order[row, column] = left
    for row in range(1, grid_size):
        if int(torch.randint(0, 2, (), generator=generator).item()):
            upper = order[row - 1].clone()
            order[row - 1] = order[row]
            order[row] = upper
    return order.flatten().to(device)


def region_confusion(
    images: Tensor,
    grid_size: int = 7,
    *,
    generator: torch.Generator | None = None,
) -> RegionConfusionBatch:
    """Create locally shuffled DCL views and exact region-layout targets.

    The operation is applied after the common stochastic image transform, so
    original and confused views share rotation, crop, color jitter, and flip.
    ImageNet normalization commutes with patch permutation, allowing the RCM
    to operate on tensors without changing pixel values.
    """

    if images.ndim != 4:
        raise ValueError("images harus berbentuk [batch, channel, height, width].")
    if grid_size <= 1:
        raise ValueError("grid_size harus lebih besar dari satu.")
    batch, channels, height, width = images.shape
    if height % grid_size or width % grid_size:
        raise ValueError(
            "Ukuran gambar harus habis dibagi grid_size: "
            f"{height}x{width} vs {grid_size}."
        )

    patch_height = height // grid_size
    patch_width = width // grid_size
    patches = (
        images.reshape(
            batch,
            channels,
            grid_size,
            patch_height,
            grid_size,
            patch_width,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_size * grid_size, channels, patch_height, patch_width)
    )
    permutations = torch.stack(
        [
            _local_permutation(
                grid_size,
                generator=generator,
                device=images.device,
            )
            for _ in range(batch)
        ]
    )
    gather_index = permutations[:, :, None, None, None].expand_as(patches)
    shuffled_patches = torch.gather(patches, dim=1, index=gather_index)
    shuffled = (
        shuffled_patches.reshape(
            batch,
            grid_size,
            grid_size,
            channels,
            patch_height,
            patch_width,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )

    base_layout = normalized_linear_layout(
        grid_size,
        device=images.device,
        dtype=images.dtype,
    )
    return RegionConfusionBatch(
        images=shuffled,
        original_layout=base_layout.unsqueeze(0).expand(batch, -1).clone(),
        confused_layout=base_layout[permutations],
        permutations=permutations,
    )
