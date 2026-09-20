from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CLAHEConfig:
    """Frozen LAB-luminance CLAHE control ported from coffee-bean-detection."""

    clip_limit: float = 3.0
    tile_grid_size: tuple[int, int] = (8, 8)

    @classmethod
    def from_mapping(cls, payload: "CLAHEConfig | dict[str, Any] | None") -> "CLAHEConfig":
        if isinstance(payload, cls):
            result = payload
        else:
            values = dict(payload or {})
            if "tile_grid_size" in values:
                values["tile_grid_size"] = tuple(int(v) for v in values["tile_grid_size"])
            result = cls(**values)
        if result.clip_limit <= 0:
            raise ValueError("clip_limit harus positif")
        if len(result.tile_grid_size) != 2 or any(int(v) <= 0 for v in result.tile_grid_size):
            raise ValueError("tile_grid_size harus berisi dua integer positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tile_grid_size"] = list(self.tile_grid_size)
        return payload


class CLAHEFrontend(nn.Module):
    """Exact LAB-L CLAHE reference; deterministic and parameter-free.

    This keeps the detection-repo implementation semantics: BCHW float RGB in
    [0,1] is quantized to uint8, converted RGB->LAB, CLAHE is applied only to
    L, then LAB->RGB and returned as the original floating dtype in [0,1].
    """

    def __init__(self, config: CLAHEConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = CLAHEConfig.from_mapping(config)

    def _one(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"CLAHE membutuhkan RGB CHW, diterima {tuple(image.shape)}")
        try:
            import cv2
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("OpenCV diperlukan untuk CLAHE") from error

        device, dtype = image.device, image.dtype
        rgb = (
            image.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.config.clip_limit),
            tileGridSize=tuple(int(v) for v in self.config.tile_grid_size),
        )
        l_enhanced = clahe.apply(l_channel)
        enhanced_rgb = cv2.cvtColor(
            cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2RGB
        )
        output = torch.from_numpy(enhanced_rgb.copy()).permute(2, 0, 1)
        return output.to(device=device, dtype=dtype).div_(255.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"CLAHE membutuhkan BCHW, diterima {tuple(value.shape)}")
        if value.shape[1] != 3:
            raise ValueError("CLAHE dikunci untuk input RGB 3-channel")
        if not torch.is_floating_point(value):
            raise TypeError("CLAHE memerlukan tensor floating point")
        return torch.stack([self._one(value[index]) for index in range(value.shape[0])], dim=0)
