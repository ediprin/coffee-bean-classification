from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pywt
from PIL import Image
from scipy import ndimage as ndi
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from skimage import color, feature, filters, measure

from bilinear_lmmd.analysis.coffee17_physical_descriptors import (
    bean_mask_from_background_otsu,
)


FEATURE_NAMES = (
    "Average_Blue", "Average_B", "Average_Red", "Average_Green",
    "GLCM_SumOfSquaresVariance_Mean", "GLCM_SumVariance_Mean",
    "GT_th_3.0_freq_0.4_mean", "GT_th_0.0_freq_0.4_mean",
    "GT_th_1.0_freq_0.4_mean", "GT_th_2.0_freq_0.4_mean",
    "Average_L", "GLCM_SumEntropy_Mean", "GLCM_SumAverage_Mean",
    "GLCM_ASM_Mean", "GLCM_Entropy_Mean", "LBP_R_1_P_8_energy",
    "LBP_R_1_P_8_entropy", "Average_A", "GLCM_DifferenceVariance_Mean",
    "LBP_R_2_P_16_entropy", "LBP_R_2_P_16_energy",
    "GLCM_Correlation_Mean", "LBP_R_3_P_24_entropy",
    "LBP_R_3_P_24_energy", "GLCM_DifferenceEntropy_Mean",
    "GT_th_2.0_freq_0.05_mean", "GLCM_MaximalCorrelationCoefficient_Mean",
    "LTE_LL_7", "GT_th_0.0_freq_0.05_mean", "GLCM_Information2_Mean",
    "GLCM_Information1_Mean", "GLCM_InverseDifferenceMoment_Mean",
    "DWT_bior3.3_level_2_ad_mean", "GT_th_1.0_freq_0.05_mean",
    "GT_th_0.0_freq_0.4_std", "DWT_bior3.3_level_3_ad_mean",
    "GT_th_3.0_freq_0.05_mean", "DWT_bior3.3_level_3_ad_std",
    "LTE_SS_7", "GT_th_1.0_freq_0.4_std", "Area",
    "DWT_bior3.3_level_2_ad_std", "DWT_bior3.3_level_1_da_mean",
    "DWT_bior3.3_level_1_dd_mean", "GT_th_3.0_freq_0.4_std",
    "LTE_ES_7", "DWT_bior3.3_level_2_da_mean",
    "DWT_bior3.3_level_1_ad_mean", "GLCM_Contrast_Mean",
    "DWT_bior3.3_level_2_dd_mean", "LTE_LE_7",
    "DWT_bior3.3_level_3_da_mean", "LTE_EE_7",
    "DWT_bior3.3_level_3_dd_std", "LTE_LS_7",
    "DWT_bior3.3_level_3_da_std", "DWT_bior3.3_level_3_dd_mean",
    "GT_th_2.0_freq_0.4_std", "DWT_bior3.3_level_2_dd_std",
    "DWT_bior3.3_level_2_da_std", "DWT_bior3.3_level_1_ad_std",
    "DWT_bior3.3_level_1_dd_std", "GT_th_2.0_freq_0.05_std",
    "Circularity", "Perimeter", "DWT_bior3.3_level_1_da_std",
    "Diameter", "GT_th_1.0_freq_0.05_std", "GT_th_3.0_freq_0.05_std",
    "GT_th_0.0_freq_0.05_std", "Eccentricity",
)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _masked_gray(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(rgb).astype(np.float64)
    fill = float(np.median(gray[mask]))
    return np.where(mask, gray, fill)


def _masked_glcm(gray: np.ndarray, mask: np.ndarray, levels: int = 32) -> np.ndarray:
    quant = np.clip(np.floor(gray * levels), 0, levels - 1).astype(np.int32)
    matrices = []
    for dy, dx in ((0, 1), (-1, 1), (-1, 0), (-1, -1)):
        y0a, y1a = max(0, -dy), min(gray.shape[0], gray.shape[0] - dy)
        x0a, x1a = max(0, -dx), min(gray.shape[1], gray.shape[1] - dx)
        y0b, y1b = y0a + dy, y1a + dy
        x0b, x1b = x0a + dx, x1a + dx
        valid = mask[y0a:y1a, x0a:x1a] & mask[y0b:y1b, x0b:x1b]
        a = quant[y0a:y1a, x0a:x1a][valid]
        b = quant[y0b:y1b, x0b:x1b][valid]
        matrix = np.zeros((levels, levels), dtype=np.float64)
        if a.size:
            np.add.at(matrix, (a, b), 1.0)
            np.add.at(matrix, (b, a), 1.0)
        total = float(matrix.sum())
        if total > 0:
            matrix /= total
        matrices.append(matrix)
    return np.mean(matrices, axis=0)


def _entropy(p: np.ndarray) -> float:
    values = p[p > 0]
    return float(-(values * np.log2(values)).sum())


def _glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = _masked_glcm(gray, mask)
    levels = p.shape[0]
    idx = np.arange(levels, dtype=np.float64)
    i, j = np.meshgrid(idx, idx, indexing="ij")
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    mux = float((idx * px).sum())
    muy = float((idx * py).sum())
    sigx = math.sqrt(max(float(((idx - mux) ** 2 * px).sum()), 0.0))
    sigy = math.sqrt(max(float(((idx - muy) ** 2 * py).sum()), 0.0))

    asm = float((p * p).sum())
    contrast = float((((i - j) ** 2) * p).sum())
    corr = float((((i - mux) * (j - muy) * p).sum()) / (sigx * sigy + 1e-12))
    invdiff = float((p / (1.0 + (i - j) ** 2)).sum())
    entropy = _entropy(p)
    sos_var = float((((i - mux) ** 2) * p).sum())

    psum = np.zeros(2 * levels - 1, dtype=np.float64)
    pdiff = np.zeros(levels, dtype=np.float64)
    for a in range(levels):
        for b in range(levels):
            psum[a + b] += p[a, b]
            pdiff[abs(a - b)] += p[a, b]
    sum_idx = np.arange(2 * levels - 1, dtype=np.float64)
    diff_idx = np.arange(levels, dtype=np.float64)
    sum_avg = float((sum_idx * psum).sum())
    sum_entropy = _entropy(psum)
    sum_var = float((((sum_idx - sum_avg) ** 2) * psum).sum())
    diff_avg = float((diff_idx * pdiff).sum())
    diff_var = float((((diff_idx - diff_avg) ** 2) * pdiff).sum())
    diff_entropy = _entropy(pdiff)

    hx = _entropy(px)
    hy = _entropy(py)
    pxpy = px[:, None] * py[None, :]
    hxy1 = float(-(p[p > 0] * np.log2(np.maximum(pxpy[p > 0], 1e-12))).sum())
    hxy2 = _entropy(pxpy)
    imc1 = float((entropy - hxy1) / max(hx, hy, 1e-12))
    imc2 = float(math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * max(hxy2 - entropy, 0.0)))))

    q = np.zeros_like(p)
    for a in range(levels):
        for b in range(levels):
            denom = px * py[b]
            valid = denom > 1e-12
            q[a, b] = float(np.sum((p[a, valid] * p[b, valid]) / denom[valid]))
    eig = np.sort(np.real(np.linalg.eigvals(q)))
    mcc = float(math.sqrt(max(eig[-2], 0.0))) if eig.size >= 2 else 0.0

    return {
        "GLCM_SumOfSquaresVariance_Mean": sos_var,
        "GLCM_SumVariance_Mean": sum_var,
        "GLCM_SumEntropy_Mean": sum_entropy,
        "GLCM_SumAverage_Mean": sum_avg,
        "GLCM_ASM_Mean": asm,
        "GLCM_Entropy_Mean": entropy,
        "GLCM_DifferenceVariance_Mean": diff_var,
        "GLCM_Correlation_Mean": corr,
        "GLCM_DifferenceEntropy_Mean": diff_entropy,
        "GLCM_MaximalCorrelationCoefficient_Mean": mcc,
        "GLCM_Information2_Mean": imc2,
        "GLCM_Information1_Mean": imc1,
        "GLCM_InverseDifferenceMoment_Mean": invdiff,
        "GLCM_Contrast_Mean": contrast,
    }


def _lbp_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for radius, points in ((1, 8), (2, 16), (3, 24)):
        lbp = feature.local_binary_pattern(gray, points, radius, method="uniform")
        values = lbp[mask].astype(np.int32)
        hist = np.bincount(values, minlength=points + 2).astype(np.float64)
        hist /= max(float(hist.sum()), 1.0)
        energy = float(np.square(hist).sum())
        entropy = _entropy(hist)
        out[f"LBP_R_{radius}_P_{points}_energy"] = energy
        out[f"LBP_R_{radius}_P_{points}_entropy"] = entropy
    return out


def _laws_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    # Classical Laws 5-tap vectors; "_7" is retained to match Tulsi's feature
    # naming and denotes the local energy window used below.
    vectors = {
        "L": np.array([1, 4, 6, 4, 1], dtype=np.float64),
        "E": np.array([-1, -2, 0, 2, 1], dtype=np.float64),
        "S": np.array([-1, 0, 2, 0, -1], dtype=np.float64),
    }
    local_mean = ndi.uniform_filter(gray, size=15, mode="reflect")
    centered = gray - local_mean
    specs = (("L", "L"), ("S", "S"), ("E", "S"), ("L", "E"), ("E", "E"), ("L", "S"))
    out: dict[str, float] = {}
    for left, right in specs:
        kernel = np.outer(vectors[left], vectors[right])
        response = ndi.convolve(centered, kernel, mode="reflect")
        energy = ndi.uniform_filter(np.abs(response), size=7, mode="reflect")
        out[f"LTE_{left}{right}_7"] = float(np.mean(energy[mask]))
    return out


def _dwt_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    # Fill background before the wavelet transform to avoid a synthetic white
    # rectangle edge dominating coefficients near the segmented bean.
    filled = np.where(mask, gray, float(np.median(gray[mask])))
    coeffs = pywt.wavedec2(filled, wavelet="bior3.3", level=3, mode="symmetric")
    details_by_level = {3: coeffs[1], 2: coeffs[2], 1: coeffs[3]}
    out: dict[str, float] = {}
    # PyWavelets detail tuple is (cH, cV, cD); frozen names follow ad/da/dd.
    for level in (1, 2, 3):
        cH, cV, cD = details_by_level[level]
        for suffix, array in (("ad", cH), ("da", cV), ("dd", cD)):
            out[f"DWT_bior3.3_level_{level}_{suffix}_mean"] = float(np.mean(array))
            out[f"DWT_bior3.3_level_{level}_{suffix}_std"] = float(np.std(array))
    return out


def _gabor_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for theta in (0.0, 1.0, 2.0, 3.0):
        theta_tag = f"{theta:.1f}"
        for frequency in (0.4, 0.05):
            freq_tag = "0.4" if frequency == 0.4 else "0.05"
            real, imag = filters.gabor(gray, frequency=frequency, theta=theta)
            magnitude = np.hypot(real, imag)
            values = magnitude[mask]
            prefix = f"GT_th_{theta_tag}_freq_{freq_tag}"
            out[f"{prefix}_mean"] = float(np.mean(values))
            out[f"{prefix}_std"] = float(np.std(values))
    return out


def _shape_features(mask: np.ndarray) -> dict[str, float]:
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        raise ValueError("Bean mask kosong.")
    region = max(regions, key=lambda r: r.area)
    area = float(region.area)
    perimeter = float(measure.perimeter(mask, neighborhood=8))
    circularity = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-12))
    return {
        "Area": area,
        "Circularity": circularity,
        "Perimeter": perimeter,
        "Diameter": float(region.equivalent_diameter_area),
        "Eccentricity": float(region.eccentricity),
    }


def extract_tulsi71(path: Path) -> np.ndarray:
    """Extract the frozen 71-D Tulsi-aligned descriptor vector.

    The descriptor *families and feature names* follow Tulsi et al. (2026).
    Coffee17 uses the project's already-audited label-free bean segmentation
    rather than Tulsi's red-channel Otsu, whose failure mode on pale beans with
    dark spots was established before this experiment.
    """

    rgb = _load_rgb(Path(path))
    mask = bean_mask_from_background_otsu(rgb)
    if int(mask.sum()) < 64:
        raise ValueError(f"Bean mask terlalu kecil: {path}")

    gray = _masked_gray(rgb, mask)
    lab = color.rgb2lab(rgb)
    features: dict[str, float] = {
        # Tulsi appendix uses Average_Blue for CIELAB b* and Average_B for RGB B.
        "Average_Blue": float(np.mean(lab[..., 2][mask])),
        "Average_B": float(np.mean(rgb[..., 2][mask] * 255.0)),
        "Average_Red": float(np.mean(rgb[..., 0][mask] * 255.0)),
        "Average_Green": float(np.mean(rgb[..., 1][mask] * 255.0)),
        "Average_L": float(np.mean(lab[..., 0][mask])),
        "Average_A": float(np.mean(lab[..., 1][mask])),
    }
    features.update(_glcm_features(gray, mask))
    features.update(_lbp_features(gray, mask))
    features.update(_laws_features(gray, mask))
    features.update(_dwt_features(gray, mask))
    features.update(_gabor_features(gray, mask))
    features.update(_shape_features(mask))

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError("Tulsi71 feature hilang: " + ", ".join(missing))
    values = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    if values.shape != (71,) or not np.isfinite(values).all():
        raise RuntimeError(f"Tulsi71 invalid untuk {path}: shape={values.shape}")
    return values


def extract_feature_matrix(paths: list[Path]) -> np.ndarray:
    return np.stack([extract_tulsi71(Path(path)) for path in paths], axis=0)


def mrmr_select(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 20,
    random_state: int = 42,
) -> list[int]:
    """Deterministic MIQ-style mRMR selection fitted on the training fold only."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("mRMR mengharapkan matriks N x 71.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Jumlah row x/y berbeda.")
    if not 1 <= k <= x.shape[1]:
        raise ValueError("k mRMR tidak valid.")

    # Robust scaling only for numerical stability of kNN MI estimators.
    med = np.median(x, axis=0)
    scale = np.median(np.abs(x - med), axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - med) / scale

    relevance = mutual_info_classif(
        z, y, discrete_features=False, random_state=random_state
    )

    selected: list[int] = [int(np.argmax(relevance))]
    redundancy_sum = np.zeros(z.shape[1], dtype=np.float64)

    while len(selected) < k:
        newest = selected[-1]
        if float(np.var(z[:, newest])) > 1e-12:
            against_newest = mutual_info_regression(
                z,
                z[:, newest],
                discrete_features=False,
                random_state=random_state,
            )
            redundancy_sum += np.asarray(against_newest, dtype=np.float64)

        best_index = -1
        best_score = -np.inf
        for candidate in range(z.shape[1]):
            if candidate in selected:
                continue
            red = float(redundancy_sum[candidate] / len(selected))
            score = float(relevance[candidate]) - red
            if score > best_score + 1e-15 or (
                abs(score - best_score) <= 1e-15
                and (best_index < 0 or candidate < best_index)
            ):
                best_score = score
                best_index = candidate
        if best_index < 0:
            raise RuntimeError("mRMR gagal memilih feature berikutnya.")
        selected.append(best_index)
    return selected
