from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn

from .raw import RawFrontend
from .clahe import CLAHEConfig, CLAHEFrontend
from .af2 import AF2Config, AF2LuminanceFrontend
from .wav1 import WAV1Config, WAV1Frontend


ARM_CODES = ("R0", "C0", "F0", "W0")

_SPECS: dict[str, dict[str, Any]] = {
    "R0": {
        "code": "R0",
        "method": "raw",
        "execution_device": "same_as_input",
        "origin": "classification control",
        "output_contract": "identity",
        "config": {},
    },
    "C0": {
        "code": "C0",
        "method": "clahe_lab_luminance",
        "execution_device": "cpu_preferred",
        "origin": {
            "paper": "Mohanty et al. 2026",
            "operation": "CIELAB L-channel CLAHE",
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8],
        },
        "output_contract": "float RGB in [0,1]",
        "config": CLAHEConfig().to_dict(),
    },
    "F0": {
        "code": "F0",
        "method": "af2_luminance_shared_gate",
        "execution_device": "gpu_preferred",
        "origin": {
            "frequency_basis": "Xu et al. 2025 AFAB-2 + patch-wise DFT",
            "color_texture_basis": "Maenpaa and Pietikainen 2004",
            "matched_detection_reference": {
                "repository": "ediprin/coffee-bean-detection",
                "commit": "6ef389c23932e44fe4135c32d471b3008b1cbf39",
                "path": "src/coffee_detector/af2_luminance/operator.py",
            },
        },
        "output_contract": "shared Rec.709 luminance gate; raw RGB residual; no clipping",
        "config": AF2Config().to_dict(),
    },
    "W0": {
        "code": "W0",
        "method": "haar4_visushrink_soft_reconstruction",
        "execution_device": "gpu_preferred",
        "origin": {
            "paper": "Yang et al. 2025",
            "operation": "RGB channel-wise Haar DWT, VisuShrink soft threshold, inverse DWT",
        },
        "output_contract": "four-level reconstructed RGB; no post-hoc clipping",
        "config": WAV1Config().to_dict(),
    },
}


def preprocessing_spec(code: str) -> dict[str, Any]:
    code = str(code).upper()
    if code not in _SPECS:
        raise ValueError(f"Preprocessing code harus salah satu {ARM_CODES}")
    return deepcopy(_SPECS[code])


def build_preprocessing_frontend(
    code: str,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    code = str(code).upper()
    if code == "R0":
        if config:
            raise ValueError("R0 tidak menerima parameter preprocessing")
        return RawFrontend()
    if code == "C0":
        return CLAHEFrontend(CLAHEConfig.from_mapping(config))
    if code == "F0":
        return AF2LuminanceFrontend(AF2Config.from_mapping(config))
    if code == "W0":
        return WAV1Frontend(WAV1Config.from_mapping(config))
    raise ValueError(f"Preprocessing code harus salah satu {ARM_CODES}")
