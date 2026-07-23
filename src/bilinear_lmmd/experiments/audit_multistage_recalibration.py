from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model


CONFIGS = {
    "BE2G": Path("configs/backbones/BE2G_efficientnetv2_gap_source.yaml"),
    "BE2H": Path("configs/backbones/BE2H_efficientnetv2_hbp_source.yaml"),
    "MSF0": Path(
        "configs/finegrained/MSF0_efficientnetv2_fixed_multistage.yaml"
    ),
    "MSF1": Path(
        "configs/finegrained/MSF1_efficientnetv2_adaptive_multistage.yaml"
    ),
    "MSFC": Path(
        "configs/finegrained/MSFC_efficientnetv2_channel_control.yaml"
    ),
}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def audit_models(
    output: str | Path,
    models: list[str],
    device_name: str,
    image_size: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    cpu_threads: int,
) -> dict:
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats harus positif dan warmup tidak boleh negatif.")
    if batch_size <= 0 or image_size <= 0:
        raise ValueError("batch_size dan image_size harus positif.")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads harus positif.")
    unknown = sorted(set(models) - set(CONFIGS))
    if unknown:
        raise ValueError(f"Model audit tidak dikenal: {unknown}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    rows: dict[str, dict] = {}

    for code in models:
        cfg = load_config(CONFIGS[code])
        # This is an architecture audit, not a weight download or training run.
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"]).to(device).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        with torch.inference_mode():
            result = model(images)
            for _ in range(warmup):
                model(images)
            _synchronize(device)
            start = time.perf_counter()
            for _ in range(repeats):
                model(images)
            _synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0 / repeats

        row = {
            "config": str(CONFIGS[code]),
            "device": str(device),
            "batch_size": batch_size,
            "image_size": image_size,
            "parameters": parameters,
            "trainable_parameters": trainable,
            "size_fp32_mb": parameters * 4 / 1024**2,
            "latency_ms_per_batch": latency_ms,
            "throughput_images_per_second": batch_size * 1000.0 / latency_ms,
            "logits_shape": list(result.logits.shape),
            "embedding_shape": list(result.embedding.shape),
        }
        if result.gate_weights is not None:
            weights = result.gate_weights
            row.update(
                {
                    "gate_shape": list(weights.shape),
                    "gate_stage_sum_max_error": float(
                        (weights.sum(dim=1) - 1.0).abs().max().cpu()
                    ),
                    "initial_gate_min": float(weights.min().cpu()),
                    "initial_gate_max": float(weights.max().cpu()),
                }
            )
        rows[code] = row
        del model

    if "BE2G" in rows:
        baseline_parameters = rows["BE2G"]["parameters"]
        for code in ("MSF0", "MSF1", "MSFC"):
            if code not in rows:
                continue
            rows[code]["parameter_delta_vs_BE2G"] = (
                rows[code]["parameters"] - baseline_parameters
            )
            rows[code]["parameter_delta_pct_vs_BE2G"] = 100.0 * (
                rows[code]["parameters"] / baseline_parameters - 1.0
            )

    payload = {
        "audit": "multistage_recalibration_architecture",
        "training_performed": False,
        "cpu_threads": cpu_threads if device.type == "cpu" else None,
        "models": rows,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== STATIC MULTISTAGE AUDIT ===")
    for code, row in rows.items():
        print(
            f"{code:5s} params={row['parameters']:,} "
            f"size={row['size_fp32_mb']:.2f}MB "
            f"latency={row['latency_ms_per_batch']:.2f}ms "
            f"logits={row['logits_shape']} embedding={row['embedding_shape']}"
        )
    print(f"TRAINING: tidak | SAVED: {destination.resolve()}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(CONFIGS),
        default=list(CONFIGS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    audit_models(
        output=args.output,
        models=args.models,
        device_name=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        cpu_threads=args.cpu_threads,
    )


if __name__ == "__main__":
    main()
