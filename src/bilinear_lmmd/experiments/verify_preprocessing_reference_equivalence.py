from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

from bilinear_lmmd.data.preprocessing import AF2Config, AF2LuminanceFrontend


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


def _load_f0_reference(reference_repo: Path):
    coffee = reference_repo / "src/coffee_detector"
    if not coffee.is_dir():
        raise FileNotFoundError(
            f"Reference repo tidak memiliki src/coffee_detector: {reference_repo}"
        )
    _package("coffee_detector", coffee)
    _package("coffee_detector.afab", coffee / "afab")
    afab = _load_module(
        "coffee_detector.afab.operator",
        coffee / "afab/operator.py",
    )
    _package("coffee_detector.af2_luminance", coffee / "af2_luminance")
    luminance = _load_module(
        "coffee_detector.af2_luminance.operator",
        coffee / "af2_luminance/operator.py",
    )
    return afab, luminance


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
    if arm != "F0":
        raise ValueError(
            "Code-reference equivalence hanya berlaku untuk F0 luminance."
        )
    reference_repo = Path(reference_repo).expanduser().resolve()
    actual_commit = _git_commit(reference_repo)
    if actual_commit != expected_reference_commit:
        raise RuntimeError(
            f"Reference commit berubah: {actual_commit} != "
            f"{expected_reference_commit}"
        )

    afab, luminance = _load_f0_reference(reference_repo)
    torch.manual_seed(seed)
    probe = torch.rand(2, 3, size, size)

    ours = AF2LuminanceFrontend(AF2Config())
    reference = luminance.AF2LuminanceInputEnhancer(
        afab.AFABConfig(
            mode="af2",
            patch_size=32,
            overlap=0.50,
            gamma=0.10,
            angular_bins=360,
            chunk_size=128,
            eps=1.0e-8,
        )
    )

    with torch.inference_mode():
        left = ours(probe.clone())
        right = reference(probe.clone())

    exact = torch.equal(left, right)
    max_abs = float((left - right).abs().max())
    result = {
        "format": "bilinear_lmmd.preprocessing.reference_equivalence.v3",
        "arm": "F0",
        "method": "af2_luminance_shared_gate",
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
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not exact:
        raise RuntimeError(
            f"Port F0 luminance tidak bitwise-equivalent: max_abs={max_abs}"
        )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("F0",))
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
