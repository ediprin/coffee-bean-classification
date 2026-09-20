from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn

from .raw import RawFrontend
from .clahe import CLAHEConfig, CLAHEFrontend
from .af2 import AF2Config, AF2Frontend
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
            "repository": "ediprin/coffee-bean-detection",
            "ref": "agent/af2-clahe-control",
            "path": "src/coffee_detector/classical_enhancement/operator.py",
        },
        "output_contract": "float RGB [0,1]",
        "config": CLAHEConfig().to_dict(),
    },
    "F0": {
        "code": "F0",
        "method": "af2_canonical",
        "execution_device": "gpu_preferred",
        "origin": {
            "repository": "ediprin/coffee-bean-detection",
            "ref": "agent/af2-spectral-factorization",
            "path": "src/coffee_detector/afab/operator.py",
        },
        "output_contract": "canonical residual gate; input [0,1] -> theoretical [0,2]",
        "config": AF2Config().to_dict(),
    },
    "W0": {
        "code": "W0",
        "method": "wav1_haar_detail_energy",
        "execution_device": "gpu_preferred",
        "origin": {
            "repository": "ediprin/coffee-bean-detection",
            "ref": "agent/af2-rad-wavelet-refinement",
            "path": "src/coffee_detector/af2_spectral/operator.py",
            "arm": "WAV1",
        },
        "output_contract": "canonical residual gate; input [0,1] -> theoretical [0,2]",
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
        return AF2Frontend(AF2Config.from_mapping(config))
    if code == "W0":
        return WAV1Frontend(WAV1Config.from_mapping(config))
    raise ValueError(f"Preprocessing code harus salah satu {ARM_CODES}")
