from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AF2Config:
    """Frozen canonical AF2 transfer contract."""

    patch_size: int = 32
    overlap: float = 0.50
    gamma: float = 0.10
    angular_bins: int = 360
    chunk_size: int = 128
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, payload: "AF2Config | dict[str, Any] | None") -> "AF2Config":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.patch_size <= 1:
            raise ValueError("patch_size harus >1")
        if not 0.0 <= result.overlap < 1.0 or result.stride <= 0:
            raise ValueError("overlap tidak menghasilkan stride valid")
        if result.gamma <= 0.0 or result.angular_bins <= 1:
            raise ValueError("gamma/angular_bins tidak valid")
        if result.chunk_size <= 0 or result.eps <= 0.0:
            raise ValueError("chunk_size/eps tidak valid")
        return result

    @property
    def stride(self) -> int:
        return int(round(self.patch_size * (1.0 - self.overlap)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def af2_entropy_threshold(probability: torch.Tensor, gamma: float, eps: float = 1e-8) -> torch.Tensor:
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=-1)
    return float(gamma) / (1.0 + torch.exp(-entropy))


def minmax_spatial(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    low = value.amin(dim=(-2, -1), keepdim=True)
    high = value.amax(dim=(-2, -1), keepdim=True)
    return (value - low) / (high - low).clamp_min(eps)


def afab_gate(raw: torch.Tensor, recovered: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return raw + raw * minmax_spatial(recovered, eps=eps)


class AF2Frontend(nn.Module):
    """Canonical patchwise AF2 angular-frequency frontend.

    Ported with the same choices used in the detection repo: independent RGB
    transforms, 360 degree bins, patch size 32, 50% overlap, hard entropy gate,
    inverse FFT, overlap averaging and residual min-max gating.
    """

    def __init__(self, config: AF2Config | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = AF2Config.from_mapping(config)
        radius, angle_bin = self._build_frequency_geometry(self.config.patch_size, self.config.angular_bins)
        self.register_buffer("frequency_radius", radius, persistent=False)
        self.register_buffer("angle_bin", angle_bin, persistent=False)

    @staticmethod
    def _build_frequency_geometry(patch_size: int, angular_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
        center = patch_size // 2
        coord = torch.arange(patch_size, dtype=torch.float32) - float(center)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        degrees = torch.remainder(torch.rad2deg(torch.atan2(yy, xx)), 360.0)
        bins = torch.floor(degrees / (360.0 / angular_bins)).long().clamp_(0, angular_bins - 1)
        return radius, bins

    def _pad_for_windows(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = image.shape
        m, stride = self.config.patch_size, self.config.stride
        target_h = max(h, m); target_w = max(w, m)
        pad_h = (stride - ((target_h - m) % stride)) % stride
        pad_w = (stride - ((target_w - m) % stride)) % stride
        pad_h += target_h - h; pad_w += target_w - w
        if pad_h or pad_w: image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        return image, (h, w)

    def _af2_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        n, c, m, _ = shifted_frequency.shape
        magnitude = shifted_frequency.abs().reshape(n, c, m * m)
        index = self.angle_bin.to(device=magnitude.device).reshape(1, 1, -1).expand(n, c, -1)
        density = magnitude.new_zeros((n, c, self.config.angular_bins))
        density.scatter_add_(dim=-1, index=index, src=magnitude)
        total = density.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        probability = density / total
        threshold = af2_entropy_threshold(probability, self.config.gamma, self.config.eps)
        normalized_density = density / density.amax(dim=-1, keepdim=True).clamp_min(self.config.eps)
        direction_weight = torch.where(normalized_density <= threshold.unsqueeze(-1), torch.zeros_like(normalized_density), normalized_density)
        pixel_weight = torch.gather(direction_weight, dim=-1, index=index)
        return pixel_weight.reshape(n, c, m, m)

    def _filter_patch_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        original_dtype = patches.dtype; device_type = patches.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            work = patches.float(); frequency = torch.fft.fft2(work, dim=(-2, -1), norm="ortho")
            frequency = torch.fft.fftshift(frequency, dim=(-2, -1)); frequency = frequency * self._af2_weight(frequency)
            recovered = torch.fft.ifft2(torch.fft.ifftshift(frequency, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
        return recovered.to(dtype=original_dtype)

    def _recover_one(self, image: torch.Tensor) -> torch.Tensor:
        padded, original_shape = self._pad_for_windows(image)
        _, channels, hp, wp = padded.shape
        m, stride = self.config.patch_size, self.config.stride
        columns = F.unfold(padded, kernel_size=m, stride=stride); count = columns.shape[-1]
        patches = columns.transpose(1, 2).reshape(count, channels, m, m)
        recovered_columns = torch.empty_like(columns)
        for start in range(0, count, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, count); filtered = self._filter_patch_chunk(patches[start:stop])
            recovered_columns[:, :, start:stop] = filtered.reshape(stop - start, -1).transpose(0, 1).unsqueeze(0)
        recovered_sum = F.fold(recovered_columns, output_size=(hp, wp), kernel_size=m, stride=stride)
        ones = torch.ones((1, m * m, count), device=image.device, dtype=image.dtype)
        divisor = F.fold(ones, output_size=(hp, wp), kernel_size=m, stride=stride)
        recovered = recovered_sum / divisor.clamp_min(self.config.eps)
        h, w = original_shape
        return recovered[:, :, :h, :w]

    def recover(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 3: raise ValueError("AF2 membutuhkan BCHW RGB")
        if not torch.is_floating_point(value): raise TypeError("AF2 membutuhkan tensor floating point")
        return torch.cat([self._recover_one(value[i : i + 1]) for i in range(value.shape[0])], dim=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        recovered = self.recover(value)
        return afab_gate(value, recovered, eps=self.config.eps)
