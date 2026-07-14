from __future__ import annotations

import argparse
import json
import time

import torch

from .config import load_config
from .models import build_model
from .train import resolve_device


def benchmark(config_path: str, warmup: int, iterations: int) -> dict[str, float | str]:
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])
    model = build_model(cfg["model"]).to(device).eval()
    size = int(cfg["data"]["image_size"])
    sample = torch.randn(1, 3, size, size, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        synchronize()
    latency_ms = (time.perf_counter() - start) * 1000 / iterations
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "backbone": cfg["model"]["backbone"],
        "head": cfg["model"]["head"],
        "device": str(device),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "model_size_fp32_mb": parameters * 4 / 1024**2,
        "latency_ms_batch1": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ukur parameter, ukuran, dan latency model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.config, args.warmup, args.iterations), indent=2))


if __name__ == "__main__":
    main()
