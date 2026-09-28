from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


FEATURES = (
    "area_fraction",
    "perimeter_over_sqrt_area",
    "circularity",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "L_mean",
    "warm_frac_a5_b10",
    "L_frac_lt_50",
)

GEOMETRY_FEATURES = (
    "area_fraction",
    "perimeter_over_sqrt_area",
    "circularity",
    "eccentricity",
    "solidity",
    "aspect_ratio",
)

SOUR_FEATURES = ("L_mean", "warm_frac_a5_b10")
BLACK_FEATURES = ("L_mean", "L_frac_lt_50")

SHAPE_CLASSES = ("Broken", "Cut", "Immature", "Shell", "Withered")
SOUR_CLASSES = ("Full Sour", "Partial Sour")
BLACK_CLASSES = ("Full Black", "Partial Black")


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


@dataclass(frozen=True)
class PhysicalResidualFit:
    classes: tuple[str, ...]
    features: tuple[str, ...]
    active_pairs: tuple[tuple[int, int], ...]
    standardizer: Standardizer
    weights: np.ndarray
    l2: float
    label_smoothing: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    initial_objective: float
    final_objective: float

    def full_weight_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.classes), len(self.features)), dtype=np.float64)
        for value, (class_index, feature_index) in zip(self.weights, self.active_pairs):
            matrix[class_index, feature_index] = float(value)
        return matrix

    def residual_logits(self, x: np.ndarray) -> np.ndarray:
        standardized = self.standardizer.transform(np.asarray(x, dtype=np.float64))
        return standardized @ self.full_weight_matrix().T

    def adjusted_logits(self, base_logits: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.asarray(base_logits, dtype=np.float64) + self.residual_logits(x)


def build_active_pairs(classes: list[str] | tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    classes = tuple(classes)
    feature_index = {name: index for index, name in enumerate(FEATURES)}

    pairs: list[tuple[int, int]] = []
    for class_name in SHAPE_CLASSES:
        ci = classes.index(class_name)
        for feature in GEOMETRY_FEATURES:
            pairs.append((ci, feature_index[feature]))

    for class_name in SOUR_CLASSES:
        ci = classes.index(class_name)
        for feature in SOUR_FEATURES:
            pairs.append((ci, feature_index[feature]))

    for class_name in BLACK_CLASSES:
        ci = classes.index(class_name)
        for feature in BLACK_FEATURES:
            pairs.append((ci, feature_index[feature]))

    return tuple(pairs)


def fit_standardizer(x: np.ndarray) -> Standardizer:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    return Standardizer(mean=mean, scale=scale)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cross_entropy_with_smoothing(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    smoothing: float,
) -> tuple[float, np.ndarray]:
    n, c = logits.shape
    probs = _softmax(logits)

    targets = np.full((n, c), smoothing / c, dtype=np.float64)
    targets[np.arange(n), labels] += 1.0 - smoothing

    log_probs = np.log(np.clip(probs, 1.0e-15, 1.0))
    loss = -float(np.sum(targets * log_probs) / n)
    grad_logits = (probs - targets) / n
    return loss, grad_logits


def fit_physical_logit_residual(
    *,
    base_logits: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[str] | tuple[str, ...],
    l2: float = 0.01,
    label_smoothing: float = 0.1,
    maxiter: int = 500,
) -> PhysicalResidualFit:
    """Fit a tiny masked linear residual on fixed WR-HBP logits.

    The optimization is deterministic CPU L-BFGS-B over only the active
    class-feature pairs. Zero initialization gives exact WR-HBP logits before
    fitting. The WR-HBP model itself is never changed by this residual fit.
    """

    base_logits = np.asarray(base_logits, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = tuple(classes)

    if base_logits.ndim != 2 or base_logits.shape[1] != len(classes):
        raise ValueError("base_logits shape tidak cocok dengan classes.")
    if features.ndim != 2 or features.shape[1] != len(FEATURES):
        raise ValueError("features shape tidak cocok dengan frozen FEATURES.")
    if len(base_logits) != len(features) or len(labels) != len(features):
        raise ValueError("Jumlah sample logits/features/labels berbeda.")
    if l2 < 0.0:
        raise ValueError("l2 harus nonnegative.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing harus di [0,1).")

    standardizer = fit_standardizer(features)
    x = standardizer.transform(features)
    active_pairs = build_active_pairs(classes)

    def unpack(w: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(classes), len(FEATURES)), dtype=np.float64)
        for value, (ci, fi) in zip(w, active_pairs):
            matrix[ci, fi] = value
        return matrix

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        matrix = unpack(w)
        adjusted = base_logits + x @ matrix.T
        ce, grad_logits = _cross_entropy_with_smoothing(
            adjusted,
            labels,
            smoothing=label_smoothing,
        )
        grad_matrix = grad_logits.T @ x

        active_grad = np.asarray(
            [grad_matrix[ci, fi] for ci, fi in active_pairs],
            dtype=np.float64,
        )
        penalty = 0.5 * l2 * float(np.dot(w, w))
        gradient = active_grad + l2 * w
        return ce + penalty, gradient

    initial = np.zeros(len(active_pairs), dtype=np.float64)
    initial_objective, _ = objective(initial)

    result = minimize(
        fun=lambda w: objective(w),
        x0=initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maxiter),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "maxls": 50,
        },
    )

    final_objective, _ = objective(result.x)

    return PhysicalResidualFit(
        classes=classes,
        features=tuple(FEATURES),
        active_pairs=active_pairs,
        standardizer=standardizer,
        weights=np.asarray(result.x, dtype=np.float64),
        l2=float(l2),
        label_smoothing=float(label_smoothing),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
    )
