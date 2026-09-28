from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .registry import build_preprocessing_frontend


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("ImageNet normalization membutuhkan BCHW RGB")
    mean = value.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = value.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (value - mean) / std


def _frontend_payload(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"enabled", "code", "method", "execution_device"}
    return {key: value for key, value in config.items() if key not in ignored}


@dataclass
class PreprocessingRuntime:
    """Apply a frozen frontend before ImageNet normalization.

    C0 remains on CPU because its canonical implementation uses OpenCV. F0/W0
    run on the target tensor device. R0 is the identity control. Frontends are
    parameter-free and executed under no_grad because gradients with respect to
    input pixels are not part of the preprocessing-study estimand.
    """

    code: str
    device: torch.device
    frontend: torch.nn.Module

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        device: torch.device,
    ) -> "PreprocessingRuntime":
        if not bool(config.get("enabled", False)):
            raise ValueError("preprocessing-study memerlukan preprocessing.enabled=true")
        code = str(config.get("code", "")).upper()
        frontend = build_preprocessing_frontend(code, _frontend_payload(config))
        if code in {"F0", "W0"}:
            frontend = frontend.to(device)
        else:
            frontend = frontend.cpu()
        frontend.eval()
        return cls(code=code, device=device, frontend=frontend)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Preprocessing runtime membutuhkan BCHW RGB")
        if not torch.is_floating_point(images):
            raise TypeError("Preprocessing runtime membutuhkan tensor floating point")
        if float(images.min()) < -1e-6 or float(images.max()) > 1.0 + 1e-6:
            raise ValueError(
                "Input preprocessing-study harus ToTensor RGB [0,1] sebelum frontend."
            )

        with torch.no_grad():
            if self.code == "C0":
                work = self.frontend(images.detach().cpu())
                work = work.to(self.device, non_blocking=True)
            else:
                work = images.to(self.device, non_blocking=True)
                work = self.frontend(work)
        return imagenet_normalize(work)
