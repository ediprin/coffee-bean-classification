from __future__ import annotations

import torch
from torch import nn


class RawFrontend(nn.Module):
    """Identity control for the preprocessing study."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(f"Raw frontend membutuhkan BCHW RGB, diterima {tuple(value.shape)}")
        if not torch.is_floating_point(value):
            raise TypeError("Raw frontend membutuhkan tensor floating point")
        return value
