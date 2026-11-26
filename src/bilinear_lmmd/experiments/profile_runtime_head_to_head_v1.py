"""No-training, synthetic-input, interleaved CUDA architecture benchmark."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.profile_frequency_efficiency_v1 import _conv_linear_macs_per_image
from bilinear_lmmd.modeling.frequency_screening import build_frequency_screening_model

ARMS = ("B0", "W3", "C2", "B1", "W4")
EXPECTED_PARAMS = dict(zip(ARMS, (2988289, 2992169, 2993089, 3562305, 3566185)))
ANCHOR = {"B0": "B0", "W3": "B0", "C2": "B0", "B1": "B1", "W4": "B1"}


def balanced_orders(arms=ARMS, seed=42):
    """Two full cyclic Latin squares, second in reverse order."""
    base = list(arms)
    random.Random(seed).shuffle(base)
    return [
        seq[i:] + seq[:i]
        for seq in (base, base[::-1])
        for i in range(len(seq))
    ]


def stats(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Expected finite positive timing observations")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "mean": statistics.mean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def command(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build(cfg, arm, device):
    torch.manual_seed(42)
    model = build_frequency_screening_model(arm, cfg).to(device=device, dtype=torch.float32).eval()
    count = sum(p.numel() for p in model.parameters())
    if count != EXPECTED_PARAMS[arm]:
        raise RuntimeError(f"{arm}: parameter count drift {count} != {EXPECTED_PARAMS[arm]}")
    return model


def sync(device):
    torch.cuda.synchronize(device)


@torch.inference_mode()
def validate_forward(model, sample):
    out = model(sample)
    if tuple(out.logits.shape) != (1, 17) or not torch.isfinite(out.logits).all().item():
        raise RuntimeError("Invalid inference output")


@torch.inference_mode()
def latency_block(model, sample, device, n, warmup):
    for _ in range(warmup):
        model(sample)
    sync(device)
    times = []
    for _ in range(n):
        start = time.perf_counter_ns()
        model(sample)
        sync(device)
        times.append((time.perf_counter_ns() - start) / 1e6)
    return times


@torch.inference_mode()
def throughput_block(model, sample, device, n, warmup):
    for _ in range(warmup):
        model(sample)
    sync(device)
    start = time.perf_counter_ns()
    for _ in range(n):
        model(sample)
    sync(device)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return {"images": n, "seconds": elapsed, "images_per_second": n / elapsed}


@torch.inference_mode()
def isolated_memory(cfg, arm, device, image_size, warmup):
    """Only one model and input resident. Allocated, not reserved, bytes."""
    gc.collect()
    torch.cuda.empty_cache()
    sync(device)
    empty = torch.cuda.memory_allocated(device)
    model = build(cfg, arm, device)
    gen = torch.Generator(device=device).manual_seed(42)
    sample = torch.randn(1, 3, image_size, image_size, generator=gen, device=device)
    for _ in range(warmup):
        model(sample)
    sync(device)
    resident = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    model(sample)
    sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    result = {
        "empty_allocated_bytes": empty,
        "resident_allocated_bytes": resident,
        "peak_allocated_bytes": peak,
        "model_input_and_forward_peak_bytes": peak - empty,
        "forward_increment_bytes": peak - resident,
    }
    del model, sample, gen
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(config, device, output, required_commit, warmup=30, block_warmup=10, samples=50):
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("CUDA required; no automatic CPU fallback")
    if min(warmup, block_warmup, samples) < 1:
        raise ValueError("Positive warmups and sample count required")
    root = Path(__file__).resolve().parents[3]
    commit = command(["git", "rev-parse", "HEAD"], cwd=root)
    if commit != required_commit:
        raise RuntimeError(f"Scientific commit mismatch: {commit}")
    if command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root) != "":
        raise RuntimeError("Tracked repository files modified")
    cfg = copy.deepcopy(load_config(config))
    cfg["model"]["pretrained"] = False
    if cfg["model"]["backbone"] != "mobilenetv3_large_100" or int(cfg["model"]["num_classes"]) != 17:
        raise ValueError("Unexpected backbone or class count")
    image_size = int(cfg["data"]["image_size"])
    if image_size != 224 or int(cfg["model"]["projection_dim"]) != 512:
        raise ValueError("Expected 224 input and HBP512")
    if cfg["focused_efficiency"]["frecsa_kernel_size"] != 7 or cfg["focused_efficiency"]["frecsa_bias"] != 0.5:
        raise ValueError("Unexpected FReCSA defaults")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    torch.cuda.set_device(device)
    models = {arm: build(cfg, arm, device) for arm in ARMS}
    # Identical fixed synthetic tensor; model-only timing excludes transfers/preprocessing.
    gen = torch.Generator(device=device).manual_seed(42)
    sample = torch.randn(1, 3, image_size, image_size, generator=gen, device=device)
    records = {}
    for arm, model in models.items():
        validate_forward(model, sample)
        macs = _conv_linear_macs_per_image(model, image_size)
        records[arm] = {
            "candidate": arm,
            "anchor": ANCHOR[arm],
            "parameter_count": EXPECTED_PARAMS[arm],
            "state_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
            "conv_linear_macs_per_image": macs,
            "latency_blocks": [],
            "throughput_blocks": [],
        }
        for _ in range(warmup):
            model(sample)
    del model
    sync(device)
    orders = balanced_orders()
    for index, order in enumerate(orders):
        print(f"Latency round {index+1}/{len(orders)}: {order}", flush=True)
        for arm in order:
            times = latency_block(models[arm], sample, device, samples, block_warmup)
            records[arm]["latency_blocks"].append({"round": index+1, "samples_ms": times, "stats_ms": stats(times)})
    # Separate measurement: no per-image synchronization in throughput blocks.
    for index, order in enumerate(orders[::-1]):
        print(f"Throughput round {index+1}/{len(orders)}: {order}", flush=True)
        for arm in order:
            block = throughput_block(models[arm], sample, device, samples, block_warmup)
            records[arm]["throughput_blocks"].append({"round": index+1, **block})
    del models, sample, gen
    gc.collect()
    torch.cuda.empty_cache()
    for arm in ARMS:
        item = records[arm]
        blocks = item["latency_blocks"]
        item["latency_ms"] = stats([v for b in blocks for v in b["samples_ms"]])
        item["round_median_ms"] = stats([b["stats_ms"]["median"] for b in blocks])
        t = item["throughput_blocks"]
        item["sustained_batch1_img_s"] = sum(b["images"] for b in t) / sum(b["seconds"] for b in t)
        item["round_throughput_img_s"] = stats([b["images_per_second"] for b in t])
        item["isolated_memory"] = isolated_memory(cfg, arm, device, image_size, warmup)
    for arm in ARMS:
        item = records[arm]
        ref = records[ANCHOR[arm]]
        item["paired_round_latency_delta_percent"] = [
            100 * (a["stats_ms"]["median"] / b["stats_ms"]["median"] - 1)
            for a, b in zip(item["latency_blocks"], ref["latency_blocks"])
        ]
    payload = {
        "format": "coffee17.runtime_head_to_head.v1",
        "git_commit": commit,
        "config_file_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "effective_config": cfg,
        "training_performed": False,
        "dataset_accessed": False,
        "outer_test_accessed": False,
        "weights": "random_seed42_no_pretrained_no_checkpoint",
        "scope": "FP32 eager model forward on resident synthetic input; excludes I/O, normalization, softmax, transfer; not end-to-end deployment",
        "mac_scope": "Conv2d and Linear only; excludes pooling, BN, wavelet and elementwise ops",
        "memory_scope": "isolated single-model CUDA allocated peak, not reserved or process VRAM",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "timm": importlib.metadata.version("timm"),
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device), "device": str(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "nvidia_smi": command(["nvidia-smi"]),
            "tf32": False, "cudnn_benchmark": False, "strict_determinism": False,
        },
        "batch_size": 1, "image_size": image_size, "warmup": warmup,
        "block_warmup": block_warmup, "samples_per_block": samples,
        "latency_orders": orders, "throughput_orders": orders[::-1],
        "results": [records[a] for a in ARMS],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    for arm in ARMS:
        r = records[arm]
        print(arm, r["latency_ms"], "img/s", r["sustained_batch1_img_s"], flush=True)
    print("READY:", output, flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--required-commit", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--block-warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=50)
    a = p.parse_args()
    run(a.config, torch.device(a.device), a.output, a.required_commit, a.warmup, a.block_warmup, a.samples)


if __name__ == "__main__":
    main()
