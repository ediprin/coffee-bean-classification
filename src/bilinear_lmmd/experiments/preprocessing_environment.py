from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from bilinear_lmmd.core.reproducibility import canonical_json_sha256, sha256_file


PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "opencv-python-headless",
    "opencv-python",
    "scikit-learn",
    "PyYAML",
    "Pillow",
    "scipy",
    "tqdm",
    "huggingface_hub",
)


def _installed_versions() -> dict[str, str]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def current_environment() -> dict:
    software = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": _installed_versions(),
        "torch_cuda_version": torch.version.cuda,
    }
    return {
        "software": software,
        "software_sha256": canonical_json_sha256(software),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def freeze_environment(
    output: Path,
    lock_output: Path,
    *,
    authorize_freeze: bool = False,
) -> dict:
    if not authorize_freeze:
        raise RuntimeError("Runtime freeze memerlukan --authorize-freeze.")
    current = current_environment()
    lock_lines = [
        f"{name}=={version}"
        for name, version in sorted(current["software"]["packages"].items())
    ]
    lock_output.parent.mkdir(parents=True, exist_ok=True)
    lock_output.write_text("\n".join(lock_lines) + "\n", encoding="utf-8")
    payload = {
        "format": "bilinear_lmmd.preprocessing.runtime_environment.v1",
        "decision": "FROZEN_PREPROCESSING_RUNTIME",
        **current,
        "lock_file": str(lock_output),
        "lock_sha256": sha256_file(lock_output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def verify_environment(reference_path: Path) -> dict:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if reference.get("decision") != "FROZEN_PREPROCESSING_RUNTIME":
        raise RuntimeError("Environment reference belum frozen.")
    current = current_environment()
    if current["software_sha256"] != reference.get("software_sha256"):
        raise RuntimeError(
            "Runtime software berbeda dari frozen environment: "
            f"{current['software_sha256']} != {reference.get('software_sha256')}"
        )
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock-output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--authorize-freeze", action="store_true")
    args = parser.parse_args()
    if args.reference is not None:
        print(json.dumps(verify_environment(args.reference), indent=2))
        return
    if args.output is None or args.lock_output is None:
        parser.error("Freeze mode memerlukan --output dan --lock-output.")
    freeze_environment(
        args.output,
        args.lock_output,
        authorize_freeze=args.authorize_freeze,
    )


if __name__ == "__main__":
    main()
