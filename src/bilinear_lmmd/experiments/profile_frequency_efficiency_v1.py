from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from bilinear_lmmd.config import load_config
from bilinear_lmmd.modeling.frequency_screening import (
    FREQUENCY_CANDIDATES,
    build_frequency_screening_model,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _conv_linear_macs_per_image(model: nn.Module, image_size: int) -> int:
    total = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal total
        out = output
        batch = int(out.shape[0])
        out_per_image = out.numel() // batch
        kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
        total += (
            out_per_image
            * (int(module.in_channels) // int(module.groups))
            * kernel
        )

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal total
        out = output
        batch = int(out.shape[0])
        out_per_image = out.numel() // batch
        total += out_per_image * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        model(torch.randn(1, 3, image_size, image_size, device=device))
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return int(total)


def profile_candidate(
    cfg: dict,
    candidate: str,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict:
    local_cfg = copy.deepcopy(cfg)
    # Profiling must never depend on downloading pretrained weights.
    local_cfg["model"]["pretrained"] = False

    torch.manual_seed(42)
    model = build_frequency_screening_model(candidate, local_cfg).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )

    # Real forward/backward smoke before any expensive Coffee17 training.
    model.train()
    smoke = torch.randn(2, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    output = model(smoke, labels=labels)
    loss = torch.nn.functional.cross_entropy(output.logits, labels)
    loss.backward()
    _sync(device)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite backward preflight for {candidate}.")

    model.zero_grad(set_to_none=True)
    model.eval()
    macs = _conv_linear_macs_per_image(model, 224)

    sample = torch.randn(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        _sync(device)

        elapsed_ms = []
        for _ in range(iterations):
            _sync(device)
            start = time.perf_counter()
            model(sample)
            _sync(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    peak_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            model(sample)
        _sync(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))

    median_ms = float(statistics.median(elapsed_ms))
    ordered = sorted(elapsed_ms)
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    p95_ms = float(ordered[p95_index])

    return {
        "candidate": candidate,
        "anchor": FREQUENCY_CANDIDATES[candidate]["anchor"],
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "state_size_mb": state_bytes / (1024.0 * 1024.0),
        "conv_linear_macs_per_image": int(macs),
        "conv_linear_macs_g": macs / 1.0e9,
        "latency_batch1_median_ms": median_ms,
        "latency_batch1_p95_ms": p95_ms,
        "throughput_batch1_img_s": 1000.0 / median_ms,
        "peak_cuda_memory_mb": (
            None if peak_memory is None else peak_memory / (1024.0 * 1024.0)
        ),
        "forward_backward_preflight": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static/runtime preflight for focused Coffee17 efficiency candidates."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(FREQUENCY_CANDIDATES),
        default=["B0", "H384", "C1", "C2", "C3", "C4"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 10:
        raise ValueError("Gunakan warmup >= 1 dan iterations >= 10.")

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diminta tetapi GPU tidak tersedia.")
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)

    cfg = load_config(args.config)
    payload = {
        "format": "bilinear_lmmd.focused_efficiency.profile.v1",
        "device": str(device),
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "note": (
            "conv_linear_macs counts Conv2d+Linear MACs only; use measured "
            "latency/throughput as the primary runtime-efficiency evidence."
        ),
        "results": [
            profile_candidate(
                cfg,
                candidate,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for candidate in args.candidates
        ],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
