from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bilinear_lmmd.data.preprocessing import (
    ARM_CODES,
    build_preprocessing_frontend,
    preprocessing_spec,
)
from bilinear_lmmd.experiments.preprocessing_contract import validate_primary_configs
from bilinear_lmmd.modeling.models import build_model


def _tensor_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _model_state(seed: int, model_cfg: dict) -> tuple[torch.nn.Module, str]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_model(model_cfg).cpu().eval()
    return model, _tensor_fingerprint(model.state_dict())


def _frontend_payload(config: dict) -> dict:
    ignored = {"enabled", "code", "method", "execution_device"}
    return {key: value for key, value in config.items() if key not in ignored}


def _chromaticity(value: torch.Tensor) -> torch.Tensor:
    return value / value.sum(dim=1, keepdim=True).clamp_min(1e-8)


def run_preprocessing_static_preflight(
    output: Path,
    *,
    seed: int = 42,
    probe_size: int = 64,
) -> dict:
    config_gate = validate_primary_configs()
    configs = config_gate["configs"]

    model_hashes = {}
    model_parameter_counts = {}
    for code in ARM_CODES:
        model, fingerprint = _model_state(seed, dict(configs[code]["model"]))
        model_hashes[code] = fingerprint
        model_parameter_counts[code] = sum(
            p.numel() for p in model.parameters()
        )

    probe = (
        torch.linspace(0.001, 1.0, steps=3 * probe_size * probe_size)
        .reshape(1, 3, probe_size, probe_size)
    )

    arms = {}
    for code in ARM_CODES:
        configured = configs[code]["preprocessing"]
        payload = _frontend_payload(configured)
        spec = preprocessing_spec(code)
        frontend = build_preprocessing_frontend(code, payload)
        with torch.inference_mode():
            first = frontend(probe.clone())
            second = frontend(probe.clone())

        trainable = sum(
            p.numel() for p in frontend.parameters() if p.requires_grad
        )
        gates = {
            "configured_code_exact": configured.get("code") == code,
            "configured_method_exact": configured.get("method") == spec["method"],
            "configured_parameters_match_reference": payload == spec["config"],
            "shape_preserved": tuple(first.shape) == tuple(probe.shape),
            "dtype_preserved": first.dtype == probe.dtype,
            "finite": bool(torch.isfinite(first).all()),
            "deterministic": bool(torch.equal(first, second)),
            "trainable_parameters_zero": trainable == 0,
            "persistent_state_zero": len(frontend.state_dict()) == 0,
        }
        if code == "R0":
            gates["identity_exact"] = bool(torch.equal(first, probe))
        elif code == "C0":
            gates["range_0_1"] = (
                float(first.min()) >= 0.0
                and float(first.max()) <= 1.0
            )
            gates["active"] = float((first - probe).abs().max()) > 0.0
        elif code == "F0":
            gates["canonical_non_decreasing"] = bool(
                torch.all(first + 1e-7 >= probe)
            )
            gates["canonical_upper_bound_2"] = (
                float(first.max()) <= 2.0 + 1e-6
            )
            gates["shared_gate_chromaticity"] = bool(
                torch.allclose(
                    _chromaticity(first),
                    _chromaticity(probe),
                    atol=2e-6,
                    rtol=2e-6,
                )
            )
            gates["active"] = float((first - probe).abs().max()) > 0.0
        else:
            gates["active"] = float((first - probe).abs().max()) > 0.0

        arms[code] = {
            "spec": spec,
            "configured_preprocessing": configured,
            "gates": gates,
            "min": float(first.min()),
            "max": float(first.max()),
            "mean": float(first.mean()),
            "std": float(first.std()),
            "fraction_lt_0": float((first < 0.0).float().mean()),
            "fraction_gt_1": float((first > 1.0).float().mean()),
        }

    gates = {
        "common_config_sha256_exact": bool(
            config_gate["common_config_sha256"]
        ),
        "common_initialized_model_state_sha256": (
            len(set(model_hashes.values())) == 1
        ),
        "common_model_parameter_count": (
            len(set(model_parameter_counts.values())) == 1
        ),
        "all_frontends_valid": all(
            all(row["gates"].values()) for row in arms.values()
        ),
    }
    result = {
        "format": "bilinear_lmmd.preprocessing.static_preflight.v3",
        "decision": (
            "PASS_PREPROCESSING_STATIC_CONTRACT"
            if all(gates.values())
            else "FAIL"
        ),
        "seed": seed,
        "common_config_sha256": config_gate["common_config_sha256"],
        "arm_config_sha256": config_gate["arm_config_sha256"],
        "model_state_sha256": model_hashes,
        "model_parameter_count": model_parameter_counts,
        "gates": gates,
        "arms": arms,
        "training_executed": False,
        "test_images_accessed": False,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if result["decision"] != "PASS_PREPROCESSING_STATIC_CONTRACT":
        raise RuntimeError(f"Static preflight gagal: {result}")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-size", type=int, default=64)
    args = parser.parse_args()
    run_preprocessing_static_preflight(
        args.output,
        seed=args.seed,
        probe_size=args.probe_size,
    )


if __name__ == "__main__":
    main()
