from __future__ import annotations

import argparse
import json
import time

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import build_model
from bilinear_lmmd.engine.train import resolve_device


def benchmark(
    config_path: str,
    warmup: int,
    iterations: int,
    batch_size: int = 1,
) -> dict[str, float | str | int | None]:
    if batch_size <= 0:
        raise ValueError("batch_size harus lebih besar dari nol.")
    cfg = load_config(config_path)
    # Efficiency depends on architecture, not pretrained values. Avoid an
    # unnecessary Hub request so the benchmark is reproducible offline.
    cfg["model"]["pretrained"] = False
    device = resolve_device(cfg["device"])
    model = build_model(cfg["model"]).to(device).eval()
    size = int(cfg["data"]["image_size"])
    sample = torch.randn(batch_size, 3, size, size, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        synchronize()
    latency_ms = (time.perf_counter() - start) * 1000 / iterations
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "backbone": cfg["model"]["backbone"],
        "head": cfg["model"]["head"],
        "device": str(device),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "model_size_fp32_mb": parameters * 4 / 1024**2,
        "batch_size": batch_size,
        "latency_ms_batch": latency_ms,
        "latency_ms_per_image": latency_ms / batch_size,
        "throughput_images_per_second": batch_size * 1000 / latency_ms,
        "peak_memory_mb": peak_memory_mb,
        # Backward compatibility for existing final-report consumers.
        "latency_ms_batch1": latency_ms if batch_size == 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ukur parameter, ukuran, dan latency model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(args.config, args.warmup, args.iterations, args.batch_size),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
