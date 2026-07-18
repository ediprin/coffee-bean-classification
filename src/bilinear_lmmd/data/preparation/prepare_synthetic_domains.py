from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from bilinear_lmmd.data.attribute_features import segment_bean


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DOMAIN_NAMES = ("illumination", "sensor", "background", "combined")
SPLITS = ("train", "val", "test")


def _seed_for(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _collect_source(source_root: Path) -> tuple[list[str], list[tuple[str, str, Path]]]:
    split_classes: dict[str, list[str]] = {}
    samples: list[tuple[str, str, Path]] = []
    for split in SPLITS:
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split source tidak ditemukan: {split_root}")
        classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        if not classes:
            raise ValueError(f"Tidak ada folder kelas di {split_root}")
        split_classes[split] = classes
        for class_name in classes:
            images = sorted(
                path
                for path in (split_root / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not images:
                raise ValueError(f"Kelas kosong: {split_root / class_name}")
            samples.extend((split, class_name, path) for path in images)

    expected = split_classes["train"]
    for split, classes in split_classes.items():
        if classes != expected:
            raise ValueError(
                f"Daftar kelas source/{split} berbeda. Seluruh split harus identik."
            )
    return expected, samples


def _illumination(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gamma = float(
        rng.uniform(0.55, 0.82) if rng.random() < 0.5 else rng.uniform(1.22, 1.70)
    )
    brightness = float(rng.uniform(0.78, 1.18))
    channel_gains = rng.uniform(0.84, 1.16, size=3).astype(np.float32)
    height, width = rgb.shape[:2]
    axis = int(rng.integers(0, 2))
    reverse = bool(rng.integers(0, 2))
    shadow_edge = float(rng.uniform(0.48, 0.78))
    length = width if axis == 1 else height
    gradient = np.linspace(shadow_edge, 1.0, length, dtype=np.float32)
    if reverse:
        gradient = gradient[::-1]
    shading = gradient[None, :, None] if axis == 1 else gradient[:, None, None]
    shifted = np.power(np.clip(rgb, 0.0, 1.0), gamma)
    shifted = shifted * brightness * channel_gains[None, None, :] * shading
    output = Image.fromarray(np.uint8(np.clip(shifted, 0.0, 1.0) * 255.0))
    return output, {
        "gamma": gamma,
        "brightness": brightness,
        "channel_gains": channel_gains.tolist(),
        "shadow_axis": "horizontal" if axis == 1 else "vertical",
        "shadow_reverse": reverse,
        "shadow_edge": shadow_edge,
    }


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _sensor(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    image = image.convert("RGB")
    width, height = image.size
    scale = float(rng.uniform(0.42, 0.72))
    low_size = (max(16, round(width * scale)), max(16, round(height * scale)))
    blur_radius = float(rng.uniform(0.35, 1.35))
    noise_std = float(rng.uniform(3.0, 10.0))
    jpeg_quality = int(rng.integers(38, 71))
    degraded = image.resize(low_size, Image.Resampling.BILINEAR).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    degraded = degraded.filter(ImageFilter.GaussianBlur(blur_radius))
    array = np.asarray(degraded, dtype=np.float32)
    array += rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
    degraded = Image.fromarray(np.uint8(np.clip(array, 0.0, 255.0)))
    degraded = _jpeg_roundtrip(degraded, jpeg_quality)
    return degraded, {
        "downsample_scale": scale,
        "blur_radius": blur_radius,
        "noise_std": noise_std,
        "jpeg_quality": jpeg_quality,
    }


def _synthetic_background(
    height: int, width: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    palettes = np.asarray(
        [
            (65, 47, 35),
            (115, 83, 55),
            (48, 70, 82),
            (85, 91, 82),
            (160, 145, 120),
            (42, 43, 47),
        ],
        dtype=np.float32,
    )
    palette_index = int(rng.integers(0, len(palettes)))
    base = palettes[palette_index]
    direction = rng.uniform(-24.0, 24.0, size=3).astype(np.float32)
    x = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :, None]
    y = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None, None]
    gradient = x * direction[None, None, :] + y * direction[None, None, :]
    noise_std = float(rng.uniform(1.5, 5.5))
    noise = rng.normal(0.0, noise_std, size=(height, width, 1)).astype(np.float32)
    background = np.clip(base[None, None, :] + gradient + noise, 0.0, 255.0)
    return background, {
        "palette_index": palette_index,
        "base_rgb": base.tolist(),
        "gradient_rgb": direction.tolist(),
        "noise_std": noise_std,
    }


def _background(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32) / 255.0
    mask = segment_bean(rgb)
    height, width = mask.shape
    background, parameters = _synthetic_background(height, width, rng)
    feather_radius = max(1.0, min(height, width) * 0.006)
    alpha_image = Image.fromarray(np.uint8(mask) * 255).filter(
        ImageFilter.GaussianBlur(feather_radius)
    )
    alpha = np.asarray(alpha_image, dtype=np.float32)[..., None] / 255.0
    composed = rgb_u8.astype(np.float32) * alpha + background * (1.0 - alpha)
    parameters.update(
        {
            "mask_area_fraction": float(mask.mean()),
            "feather_radius": feather_radius,
        }
    )
    return Image.fromarray(np.uint8(np.clip(composed, 0.0, 255.0))), parameters


def transform_domain(
    image: Image.Image, domain: str, rng: np.random.Generator
) -> tuple[Image.Image, dict]:
    if domain == "illumination":
        return _illumination(image, rng)
    if domain == "sensor":
        return _sensor(image, rng)
    if domain == "background":
        return _background(image, rng)
    if domain == "combined":
        current, background = _background(image, rng)
        current, illumination = _illumination(current, rng)
        current, sensor = _sensor(current, rng)
        return current, {
            "background": background,
            "illumination": illumination,
            "sensor": sensor,
        }
    raise ValueError(f"Domain tidak dikenal: {domain}")


def prepare_synthetic_domains(
    source_root: Path,
    output_root: Path,
    domains: list[str],
    seed: int = 42,
) -> dict:
    domains = list(dict.fromkeys(domains))
    unknown = sorted(set(domains).difference(DOMAIN_NAMES))
    if unknown:
        raise ValueError(f"Domain tidak dikenal: {unknown}")
    if not domains:
        raise ValueError("Minimal satu domain harus dipilih.")

    source_root = source_root.resolve()
    metadata_path = output_root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested = {
            "source_root": str(source_root),
            "domains": domains,
            "seed": seed,
        }
        observed = {key: metadata.get(key) for key in requested}
        if metadata.get("status") == "complete" and observed == requested:
            print(f"SKIP: synthetic domains sudah lengkap: {output_root}")
            return metadata
        raise FileExistsError(
            f"Metadata {metadata_path} tidak cocok dengan permintaan baru. "
            "Gunakan output-root lain atau pindahkan folder lama secara manual."
        )
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tetapi tidak memiliki metadata lengkap. "
            "Pindahkan folder parsial secara manual sebelum mengulang."
        )

    classes, samples = _collect_source(source_root)
    output_root.mkdir(parents=True)
    manifest_path = output_root / "manifest.jsonl"
    counts = {domain: {split: 0 for split in SPLITS} for domain in domains}
    mask_failures: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for domain in domains:
            domain_root = output_root / domain
            progress = tqdm(samples, desc=f"generate {domain}", unit="image")
            for split, class_name, source_path in progress:
                source_destination = (
                    domain_root / "source" / split / class_name / source_path.name
                )
                _link_or_copy(source_path, source_destination)
                target_name = f"{source_path.stem}__{domain}.jpg"
                target_path = domain_root / "target" / split / class_name / target_name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                relative_source = source_path.relative_to(source_root).as_posix()
                item_seed = _seed_for(seed, domain, relative_source)
                rng = np.random.default_rng(item_seed)
                try:
                    with Image.open(source_path) as opened:
                        transformed, parameters = transform_domain(opened, domain, rng)
                except ValueError as error:
                    if domain not in {"background", "combined"}:
                        raise
                    with Image.open(source_path) as opened:
                        transformed, parameters = _illumination(opened, rng)
                    parameters = {
                        "fallback": "illumination",
                        "segmentation_error": str(error),
                        "illumination": parameters,
                    }
                    mask_failures.append(
                        {
                            "domain": domain,
                            "source": relative_source,
                            "error": str(error),
                        }
                    )
                transformed.convert("RGB").save(target_path, format="JPEG", quality=95)
                counts[domain][split] += 1
                manifest.write(
                    json.dumps(
                        {
                            "domain": domain,
                            "split": split,
                            "class": class_name,
                            "source": relative_source,
                            "target": target_path.relative_to(output_root).as_posix(),
                            "item_seed": item_seed,
                            "parameters": parameters,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    metadata = {
        "status": "complete",
        "protocol": "controlled_synthetic_domain_shift",
        "claim_scope": "synthetic robustness and UDA sanity-check; not real-world validation",
        "source_root": str(source_root),
        "domains": domains,
        "seed": seed,
        "classes": classes,
        "source_samples": len(samples),
        "counts": counts,
        "mask_failure_count": len(mask_failures),
        "mask_failures": mask_failures,
        "leakage_policy": (
            "The original source split is preserved. A transformed image stays in the "
            "same train/val/test split as its source identity."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n=== CONTROLLED SYNTHETIC DOMAINS ===")
    print(f"Source : {source_root}")
    print(f"Output : {output_root}")
    print(f"Domain : {', '.join(domains)}")
    print(f"Sampel : {len(samples)} per domain")
    print(f"Mask fallback: {len(mask_failures)}")
    print("KLAIM  : synthetic robustness/UDA sanity-check, bukan validasi dunia nyata")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat controlled synthetic domain shifts tanpa kebocoran split"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAIN_NAMES,
        default=list(DOMAIN_NAMES),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_synthetic_domains(
        source_root=args.source_root,
        output_root=args.output_root,
        domains=args.domains,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
