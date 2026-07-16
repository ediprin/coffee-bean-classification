from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark


CONFIGS = {
    "CBD1": Path("configs/CBD1_mobilenetv3_hbp_source.yaml"),
    "CBDC1": Path("configs/CBDC1_mobilenetv3_capacity_residual_hbp_source.yaml"),
    "CBDD2": Path("configs/CBDD2_mobilenetv3_decoupled_gap_hbp_learned_source.yaml"),
}


def run_efficiency(
    output: Path,
    models: list[str],
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
) -> dict:
    results = {}
    for model in models:
        results[model] = {}
        for batch_size in batch_sizes:
            print(f"BENCHMARK: {model} | batch={batch_size}", flush=True)
            results[model][str(batch_size)] = benchmark(
                str(CONFIGS[model]), warmup, iterations, batch_size
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== EFISIENSI DECOUPLED CBD ===")
    for model in models:
        first = next(iter(results[model].values()))
        batch1 = results[model].get("1")
        largest_batch = max(batch_sizes)
        batch_large = results[model][str(largest_batch)]
        latency = (
            f"{batch1['latency_ms_batch']:.3f}ms"
            if batch1 is not None
            else "n/a"
        )
        print(
            f"{model}: params={first['parameters']:,} "
            f"size={first['model_size_fp32_mb']:.2f}MB "
            f"latency_b1={latency} "
            f"throughput_b{largest_batch}="
            f"{batch_large['throughput_images_per_second']:.1f} img/s"
        )
    print(f"SAVED: {output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CBD1/CBDC1/CBDD2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS)
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    run_efficiency(
        args.output, args.models, args.batch_sizes, args.warmup, args.iterations
    )


if __name__ == "__main__":
    main()
