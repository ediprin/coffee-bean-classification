from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms

from bilinear_lmmd.core.reproducibility import sha256_file
from bilinear_lmmd.data.preprocessing import ARM_CODES, build_preprocessing_frontend
from bilinear_lmmd.engine.train import resolve_device
from bilinear_lmmd.experiments.preprocessing_contract import validate_primary_configs


def _frontend_payload(config: dict) -> dict:
    ignored = {"enabled", "code", "method", "execution_device"}
    return {key: value for key, value in config.items() if key not in ignored}


def _chromaticity(value: torch.Tensor) -> torch.Tensor:
    positive = value.clamp_min(0.0)
    return positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _one_metrics(raw: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if raw.shape != candidate.shape:
        raise RuntimeError("Observability output mengubah shape")
    if not bool(torch.isfinite(candidate).all()):
        raise RuntimeError("Observability menemukan NaN/Inf")

    flat = candidate.flatten()
    delta = candidate - raw
    raw_np = raw[0].permute(1, 2, 0).cpu().numpy()
    out_np = candidate[0].permute(1, 2, 0).cpu().numpy()
    ssim = float(
        structural_similarity(
            raw_np,
            out_np,
            channel_axis=2,
            data_range=1.0,
        )
    )
    chroma = float(
        (_chromaticity(candidate) - _chromaticity(raw))
        .abs()
        .mean()
    )
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "p01": float(torch.quantile(flat, 0.01)),
        "p50": float(torch.quantile(flat, 0.50)),
        "p99": float(torch.quantile(flat, 0.99)),
        "fraction_lt_0": float((flat < 0.0).float().mean()),
        "fraction_gt_1": float((flat > 1.0).float().mean()),
        "mean_abs_delta": float(delta.abs().mean()),
        "mse": float(delta.square().mean()),
        "ssim": ssim,
        "nonnegative_chromaticity_l1": chroma,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def run_observability_audit(
    canonical_root: Path,
    raw_manifest_path: Path,
    output_dir: Path,
    *,
    device_name: str = "auto",
    image_size: int = 224,
    expected_count: int = 979,
) -> dict:
    canonical_root = Path(canonical_root).expanduser().resolve()
    raw_manifest_path = Path(raw_manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("images", [])
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Observability mengharapkan {expected_count} originals, ditemukan {len(rows)}"
        )

    config_gate = validate_primary_configs()
    configs = config_gate["configs"]
    device = resolve_device(device_name)
    resize = transforms.Resize((image_size, image_size), antialias=True)
    to_tensor = transforms.ToTensor()

    frontends = {}
    for arm in ARM_CODES:
        cfg = configs[arm]["preprocessing"]
        frontend = build_preprocessing_frontend(
            arm, _frontend_payload(cfg)
        ).eval()
        if arm in {"F0", "W0"}:
            frontend = frontend.to(device)
        frontends[arm] = frontend

    per_image: list[dict] = []
    class_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    arm_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for index, row in enumerate(rows, start=1):
        identity = row["identity"]
        path = canonical_root / identity
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            raw = to_tensor(resize(image.convert("RGB"))).unsqueeze(0)

        for arm in ARM_CODES:
            with torch.inference_mode():
                if arm == "C0":
                    candidate = frontends[arm](raw).cpu()
                elif arm in {"F0", "W0"}:
                    candidate = frontends[arm](raw.to(device)).cpu()
                else:
                    candidate = frontends[arm](raw).cpu()

            metrics = _one_metrics(raw, candidate)
            result_row = {
                "identity": identity,
                "class": row["class"],
                "arm": arm,
                **metrics,
            }
            per_image.append(result_row)
            for key, value in metrics.items():
                arm_values[arm][key].append(value)
                class_values[f"{row['class']}::{arm}"][key].append(value)

        if index % 50 == 0 or index == len(rows):
            print(f"OBSERVABILITY {index}/{len(rows)} originals", flush=True)

    summary_arms = {}
    for arm in ARM_CODES:
        values = arm_values[arm]
        summary_arms[arm] = {
            key: (
                float(min(v))
                if key == "min"
                else float(max(v))
                if key == "max"
                else _mean(v)
            )
            for key, v in values.items()
        }

    class_summary = {}
    for class_arm, metrics in sorted(class_values.items()):
        class_name, arm = class_arm.split("::", 1)
        class_summary.setdefault(class_name, {})[arm] = {
            key: _mean(values) for key, values in metrics.items()
        }

    gates = {
        "all_originals_processed": len(rows) == expected_count,
        "r0_identity": summary_arms["R0"]["mean_abs_delta"] == 0.0,
        "c0_valid_range": (
            summary_arms["C0"]["min"] >= 0.0
            and summary_arms["C0"]["max"] <= 1.0
        ),
        "f0_canonical_range": (
            summary_arms["F0"]["min"] >= -1e-7
            and summary_arms["F0"]["max"] <= 2.0 + 1e-6
        ),
        "f0_shared_gate_preserves_chromaticity": (
            summary_arms["F0"]["nonnegative_chromaticity_l1"] <= 2e-6
        ),
        "w0_active": summary_arms["W0"]["mean_abs_delta"] > 0.0,
        "all_outputs_finite": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "preprocessing_observability_per_image.csv"
    fieldnames = list(per_image[0].keys())
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)

    result = {
        "format": "bilinear_lmmd.preprocessing.observability.v1",
        "decision": (
            "PASS_PREPROCESSING_OBSERVABILITY_AUDIT"
            if all(gates.values())
            else "FAIL"
        ),
        "image_count": len(rows),
        "image_size": image_size,
        "device": str(device),
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "arm_config_sha256": config_gate["arm_config_sha256"],
        "gates": gates,
        "arms": summary_arms,
        "classes": class_summary,
        "per_image_table": str(table_path),
        "training_executed": False,
        "model_accessed": False,
        "test_split_materialized": False,
    }
    summary_path = output_dir / "preprocessing_observability.json"
    summary_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["decision"] != "PASS_PREPROCESSING_OBSERVABILITY_AUDIT":
        raise RuntimeError(f"Observability audit gagal: {gates}")
    print(json.dumps(result["arms"], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--expected-count", type=int, default=979)
    args = parser.parse_args()
    run_observability_audit(
        args.canonical_root,
        args.raw_manifest,
        args.output_dir,
        device_name=args.device,
        image_size=args.image_size,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()
