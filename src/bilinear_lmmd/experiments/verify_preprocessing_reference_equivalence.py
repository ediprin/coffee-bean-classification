from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

from bilinear_lmmd.data.preprocessing import (
    AF2Config,
    AF2Frontend,
    CLAHEConfig,
    CLAHEFrontend,
    WAV1Config,
    WAV1Frontend,
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Reference repo bukan Git checkout yang dapat diverifikasi: {repo}"
        ) from exc


def _package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat memuat module reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_modules(reference_repo: Path, arm: str):
    coffee = reference_repo / "src/coffee_detector"
    if not coffee.is_dir():
        raise FileNotFoundError(
            f"Reference repo tidak memiliki src/coffee_detector: {reference_repo}"
        )

    _package("coffee_detector", coffee)
    if arm == "C0":
        return _load_module(
            "_coffee_reference_clahe_operator",
            coffee / "classical_enhancement/operator.py",
        )

    _package("coffee_detector.afab", coffee / "afab")
    afab = _load_module(
        "coffee_detector.afab.operator",
        coffee / "afab/operator.py",
    )
    if arm == "F0":
        return afab

    _package("coffee_detector.af2_spectral", coffee / "af2_spectral")
    _load_module(
        "coffee_detector.af2_spectral.config",
        coffee / "af2_spectral/config.py",
    )
    return _load_module(
        "coffee_detector.af2_spectral.operator",
        coffee / "af2_spectral/operator.py",
    )


def verify_reference_equivalence(
    arm: str,
    reference_repo: Path,
    output: Path,
    *,
    expected_reference_commit: str,
    seed: int = 20260919,
    size: int = 65,
) -> dict:
    arm = arm.upper()
    if arm not in {"C0", "F0", "W0"}:
        raise ValueError("Reference equivalence hanya untuk C0/F0/W0")
    reference_repo = Path(reference_repo).expanduser().resolve()
    actual_commit = _git_commit(reference_repo)
    if actual_commit != expected_reference_commit:
        raise RuntimeError(
            f"Reference commit berubah: {actual_commit} != "
            f"{expected_reference_commit}"
        )

    reference_module = _load_reference_modules(reference_repo, arm)
    torch.manual_seed(seed)
    probe = torch.rand(2, 3, size, size)

    if arm == "C0":
        ours = CLAHEFrontend(CLAHEConfig())
        reference = reference_module.CLAHEInputEnhancer(
            reference_module.CLAHEConfig()
        )
    elif arm == "F0":
        ours = AF2Frontend(AF2Config())
        reference = reference_module.AFABInputEnhancer(
            reference_module.AFABConfig(
                mode="af2",
                patch_size=32,
                overlap=0.50,
                gamma=0.10,
                angular_bins=360,
                chunk_size=128,
                eps=1.0e-8,
            )
        )
    else:
        config_module = sys.modules["coffee_detector.af2_spectral.config"]
        ours = WAV1Frontend(WAV1Config())
        reference = reference_module.SpectralInputEnhancer(
            config_module.frozen_arm_config("WAV1")
        )

    with torch.inference_mode():
        left = ours(probe.clone())
        right = reference(probe.clone())

    exact = torch.equal(left, right)
    max_abs = float((left - right).abs().max())
    result = {
        "format": "bilinear_lmmd.preprocessing.reference_equivalence.v2",
        "arm": arm,
        "reference_repo": str(reference_repo),
        "reference_git_commit": actual_commit,
        "expected_reference_git_commit": expected_reference_commit,
        "seed": seed,
        "probe_shape": list(probe.shape),
        "exact_equal": exact,
        "max_abs_error": max_abs,
        "decision": "PASS" if exact else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not exact:
        raise RuntimeError(
            f"Port {arm} tidak bitwise-equivalent: max_abs={max_abs}"
        )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("C0", "F0", "W0"))
    parser.add_argument("--reference-repo", required=True, type=Path)
    parser.add_argument("--expected-reference-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--size", type=int, default=65)
    args = parser.parse_args()
    verify_reference_equivalence(
        args.arm,
        args.reference_repo,
        args.output,
        expected_reference_commit=args.expected_reference_commit,
        seed=args.seed,
        size=args.size,
    )


if __name__ == "__main__":
    main()
