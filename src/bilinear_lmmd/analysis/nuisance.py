from __future__ import annotations

import io
import math

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


NUISANCE_KINDS = (
    "brightness_low",
    "brightness_high",
    "gamma_dark",
    "gamma_bright",
    "white_balance_warm",
    "white_balance_cool",
    "shadow_left",
    "shadow_right",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_q50",
)


def _validate(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Nuisance input harus BCHW RGB.")
    if not torch.is_floating_point(images):
        raise TypeError("Nuisance input harus floating point.")
    if float(images.min()) < -1.0e-6 or float(images.max()) > 1.0 + 1.0e-6:
        raise ValueError("Nuisance input harus berada pada [0,1].")


def _shadow_mask(
    images: torch.Tensor,
    *,
    left_dark: bool,
    minimum: float = 0.65,
) -> torch.Tensor:
    width = images.shape[-1]
    ramp = torch.linspace(
        minimum,
        1.0,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    if not left_dark:
        ramp = torch.flip(ramp, dims=(0,))
    return ramp.view(1, 1, 1, width)


def _jpeg_one(image: torch.Tensor, quality: int) -> torch.Tensor:
    cpu = image.detach().clamp(0.0, 1.0).cpu()
    pil = TF.to_pil_image(cpu)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)
    recovered = Image.open(buffer).convert("RGB")
    return TF.to_tensor(recovered).to(device=image.device, dtype=image.dtype)


def apply_acquisition_nuisance(
    images: torch.Tensor,
    kind: str,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Deterministic camera/illumination proxy perturbations.

    These operations are diagnostic proxies, not a claim that they reproduce a
    physical camera pipeline. They are applied before R0/C0/F0/W0 frontends.
    """

    _validate(images)
    kind = str(kind)
    if kind not in NUISANCE_KINDS:
        raise ValueError(f"Nuisance tidak dikenal: {kind}")

    if kind == "brightness_low":
        out = images * 0.80
    elif kind == "brightness_high":
        out = images * 1.20
    elif kind == "gamma_dark":
        out = images.clamp_min(0.0).pow(1.25)
    elif kind == "gamma_bright":
        out = images.clamp_min(0.0).pow(0.80)
    elif kind == "white_balance_warm":
        gains = images.new_tensor((1.10, 1.00, 0.90)).view(1, 3, 1, 1)
        out = images * gains
    elif kind == "white_balance_cool":
        gains = images.new_tensor((0.90, 1.00, 1.10)).view(1, 3, 1, 1)
        out = images * gains
    elif kind == "shadow_left":
        out = images * _shadow_mask(images, left_dark=True)
    elif kind == "shadow_right":
        out = images * _shadow_mask(images, left_dark=False)
    elif kind == "gaussian_blur":
        out = TF.gaussian_blur(images, kernel_size=[5, 5], sigma=[1.2, 1.2])
    elif kind == "gaussian_noise":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(
            images.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=images.device, dtype=images.dtype)
        out = images + 0.03 * noise
    elif kind == "jpeg_q50":
        out = torch.stack([_jpeg_one(image, quality=50) for image in images], dim=0)
    else:  # pragma: no cover
        raise AssertionError(kind)

    return out.clamp(0.0, 1.0)


def cosine_stability(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
    if clean.shape != perturbed.shape or clean.ndim != 2:
        raise ValueError("Embedding harus NxD dan berukuran sama.")
    return float(
        F.cosine_similarity(clean, perturbed, dim=1).mean().item()
    )


def normalized_l2_shift(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
    if clean.shape != perturbed.shape or clean.ndim != 2:
        raise ValueError("Embedding harus NxD dan berukuran sama.")
    clean_n = F.normalize(clean, p=2, dim=1)
    perturbed_n = F.normalize(perturbed, p=2, dim=1)
    return float(torch.linalg.vector_norm(clean_n - perturbed_n, dim=1).mean().item())


def fisher_separability(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> float:
    """Between-class / within-class scatter on L2-normalized embeddings."""

    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings harus NxD dan labels harus N.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Jumlah embedding dan label berbeda.")

    z = F.normalize(embeddings.float(), p=2, dim=1)
    labels = labels.long()
    global_mean = z.mean(dim=0)
    between = z.new_zeros(())
    within = z.new_zeros(())
    total = float(z.shape[0])

    for cls in labels.unique(sorted=True):
        mask = labels == cls
        zc = z[mask]
        mean_c = zc.mean(dim=0)
        between = between + float(zc.shape[0]) * (mean_c - global_mean).square().sum()
        within = within + (zc - mean_c).square().sum()

    return float((between / total / (within / total + eps)).item())


def entropy_from_cosine(stability: float) -> float:
    """Optional bounded instability score for reporting only."""
    stability = max(-1.0, min(1.0, float(stability)))
    p = (stability + 1.0) / 2.0
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))
