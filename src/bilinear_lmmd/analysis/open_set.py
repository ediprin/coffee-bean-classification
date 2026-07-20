from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import weibull_min
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def standard_knownness_scores(
    logits: np.ndarray,
    energy_temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return post-hoc scores with one orientation: larger means more known."""

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits harus matriks [N, K] dengan K >= 2.")
    if energy_temperature <= 0:
        raise ValueError("energy_temperature harus positif.")
    probabilities = softmax(logits)
    scaled = logits / energy_temperature
    maximum = scaled.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(
        np.exp(scaled - maximum).sum(axis=1)
    )
    return {
        "msp": probabilities.max(axis=1),
        "mls": logits.max(axis=1),
        # Liu et al. define free energy with a negative sign for OOD scoring.
        # Here we expose its negative so all scores are knownness scores.
        "energy": energy_temperature * logsumexp,
    }


def fit_class_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    prototypes = []
    for class_index in range(num_classes):
        mask = (labels == class_index) & (predictions == class_index)
        if not mask.any():
            raise ValueError(
                f"Tidak ada train sample benar untuk prototype kelas {class_index}."
            )
        prototype = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(prototype)
        if norm <= 1e-12:
            raise ValueError(f"Prototype kelas {class_index} memiliki norm nol.")
        prototypes.append(prototype / norm)
    return np.vstack(prototypes)


def prototype_knownness(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return (normalized @ prototypes.T).max(axis=1)


@dataclass(frozen=True)
class WeibullClassModel:
    mean_activation: np.ndarray
    shape: float
    scale: float
    tail_count: int


@dataclass(frozen=True)
class ViMModel:
    """Fitted Virtual-logit Matching state.

    ``residual_basis`` contains the eigenvectors outside the principal
    subspace. Scores use the project reference orientation: larger means more
    in-distribution/known.
    """

    origin: np.ndarray
    residual_basis: np.ndarray
    alpha: float
    principal_dimension: int


def vim_default_principal_dimension(feature_dimension: int) -> int:
    """Return the dimension heuristic used by the official ViM benchmark."""

    if feature_dimension < 2:
        raise ValueError("Dimensi feature ViM minimal 2.")
    if feature_dimension >= 2048:
        dimension = 1000
    elif feature_dimension >= 768:
        dimension = 512
    else:
        dimension = feature_dimension // 2
    return min(max(dimension, 1), feature_dimension - 1)


def _stable_logsumexp(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))


def fit_vim(
    train_embeddings: np.ndarray,
    train_logits: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    principal_dimension: int | None = None,
) -> ViMModel:
    """Fit ViM using known training data only.

    The implementation follows Wang et al.'s official benchmark: classifier
    origin shift, uncentred empirical covariance, residual eigen-space, and
    virtual-logit scale matched to the mean maximum training logit.
    """

    embeddings = np.asarray(train_embeddings, dtype=np.float64)
    logits = np.asarray(train_logits, dtype=np.float64)
    weight = np.asarray(classifier_weight, dtype=np.float64)
    bias = np.asarray(classifier_bias, dtype=np.float64)
    if embeddings.ndim != 2 or logits.ndim != 2:
        raise ValueError("Embedding dan logit train ViM harus matriks.")
    if len(embeddings) != len(logits) or len(embeddings) < 2:
        raise ValueError("Jumlah embedding/logit train ViM tidak valid.")
    if weight.shape != (logits.shape[1], embeddings.shape[1]):
        raise ValueError("Bentuk weight classifier tidak cocok untuk ViM.")
    if bias.shape != (logits.shape[1],):
        raise ValueError("Bentuk bias classifier tidak cocok untuk ViM.")

    feature_dimension = embeddings.shape[1]
    dimension = (
        vim_default_principal_dimension(feature_dimension)
        if principal_dimension is None
        else int(principal_dimension)
    )
    if not 0 < dimension < feature_dimension:
        raise ValueError("Principal dimension ViM harus antara 1 dan F-1.")

    origin = -(np.linalg.pinv(weight) @ bias)
    shifted = embeddings - origin
    covariance = shifted.T @ shifted / len(shifted)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    residual_basis = np.ascontiguousarray(eigenvectors[:, order[dimension:]])
    residual_norm = np.linalg.norm(shifted @ residual_basis, axis=1)
    mean_residual = float(residual_norm.mean())
    if not np.isfinite(mean_residual) or mean_residual <= 1e-12:
        raise ValueError("Mean residual norm ViM nol atau tidak finite.")
    alpha = float(logits.max(axis=1).mean() / mean_residual)
    if not np.isfinite(alpha):
        raise ValueError("Alpha ViM tidak finite.")
    return ViMModel(
        origin=origin,
        residual_basis=residual_basis,
        alpha=alpha,
        principal_dimension=dimension,
    )


def vim_knownness(
    embeddings: np.ndarray,
    logits: np.ndarray,
    model: ViMModel,
) -> np.ndarray:
    """Compute ViM knownness; larger values indicate a known sample."""

    embeddings = np.asarray(embeddings, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if embeddings.ndim != 2 or logits.ndim != 2:
        raise ValueError("Embedding dan logit ViM harus matriks.")
    if len(embeddings) != len(logits):
        raise ValueError("Jumlah embedding dan logit ViM berbeda.")
    if embeddings.shape[1] != len(model.origin):
        raise ValueError("Dimensi embedding tidak cocok dengan model ViM.")
    residual = np.linalg.norm(
        (embeddings - model.origin) @ model.residual_basis,
        axis=1,
    )
    return _stable_logsumexp(logits) - model.alpha * residual


def _activation_distance(
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
) -> float:
    euclidean = float(np.linalg.norm(first - second))
    if metric == "euclidean":
        return euclidean
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
    cosine = 1.0 - float(np.dot(first, second)) / denominator
    if metric == "cosine":
        return cosine
    if metric == "eucos":
        # Scaling used by the reference OpenMax implementation.
        return euclidean / 200.0 + cosine
    raise ValueError("OpenMax distance harus euclidean, cosine, atau eucos.")


def fit_openmax(
    activations: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    tail_size: int = 20,
    distance_metric: str = "eucos",
) -> list[WeibullClassModel]:
    """Fit class MAV/Weibull tails using correctly classified known train data."""

    if tail_size <= 0:
        raise ValueError("tail_size harus positif.")
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    models: list[WeibullClassModel] = []
    for class_index in range(num_classes):
        mask = (labels == class_index) & (predictions == class_index)
        correct = activations[mask]
        if len(correct) < 2:
            raise ValueError(
                f"OpenMax memerlukan >=2 train sample benar untuk kelas {class_index}."
            )
        mean_activation = correct.mean(axis=0)
        distances = np.asarray(
            [
                _activation_distance(row, mean_activation, distance_metric)
                for row in correct
            ]
        )
        tail = np.sort(distances)[-min(tail_size, len(distances)) :]
        tail = np.maximum(tail, 1e-12)
        shape, _, scale = weibull_min.fit(tail, floc=0.0)
        models.append(
            WeibullClassModel(
                mean_activation=mean_activation,
                shape=float(shape),
                scale=max(float(scale), 1e-12),
                tail_count=len(tail),
            )
        )
    return models


def openmax_knownness(
    activations: np.ndarray,
    models: list[WeibullClassModel],
    alpha_rank: int = 10,
    distance_metric: str = "eucos",
) -> np.ndarray:
    """Compute one minus the OpenMax unknown probability.

    This follows the rank-weighted activation recalibration from OpenMax. The
    input activation vector is the ordinary classifier logit vector.
    """

    activations = np.asarray(activations, dtype=np.float64)
    num_classes = activations.shape[1]
    if len(models) != num_classes:
        raise ValueError("Jumlah model Weibull harus sama dengan dimensi aktivasi.")
    alpha_rank = min(max(int(alpha_rank), 1), num_classes)
    knownness = np.empty(len(activations), dtype=np.float64)
    for row_index, activation in enumerate(activations):
        ranked = np.argsort(activation)[::-1][:alpha_rank]
        rank_weights = {
            int(class_index): (alpha_rank - rank) / alpha_rank
            for rank, class_index in enumerate(ranked)
        }
        revised = activation.copy()
        unknown_activation = 0.0
        for class_index, model in enumerate(models):
            distance = _activation_distance(
                activation, model.mean_activation, distance_metric
            )
            outlier_probability = weibull_min.cdf(
                distance, model.shape, loc=0.0, scale=model.scale
            )
            reduction = outlier_probability * rank_weights.get(class_index, 0.0)
            revised[class_index] = activation[class_index] * (1.0 - reduction)
            unknown_activation += activation[class_index] - revised[class_index]
        joined = np.concatenate((revised, np.asarray([unknown_activation])))
        joined -= joined.max()
        probabilities = np.exp(joined)
        probabilities /= probabilities.sum()
        knownness[row_index] = 1.0 - probabilities[-1]
    return knownness


def threshold_from_known_validation(
    known_scores: np.ndarray,
    acceptance_target: float = 0.95,
) -> float:
    if not 0.0 < acceptance_target < 1.0:
        raise ValueError("acceptance_target harus berada di antara 0 dan 1.")
    values = np.asarray(known_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("known_scores validation harus vektor non-kosong.")
    return float(
        np.quantile(values, 1.0 - acceptance_target, method="lower")
    )


def _oscr_area(
    known_scores: np.ndarray,
    known_correct: np.ndarray,
    known_labels: np.ndarray,
    unknown_scores: np.ndarray,
    unknown_labels: np.ndarray,
    macro: bool,
) -> float:
    thresholds = np.concatenate(
        ([np.inf], np.unique(np.concatenate((known_scores, unknown_scores)))[::-1], [-np.inf])
    )
    ccr_values: list[float] = []
    fpr_values: list[float] = []
    known_classes = np.unique(known_labels)
    unknown_classes = np.unique(unknown_labels)
    for threshold in thresholds:
        known_accepted = known_scores >= threshold
        unknown_accepted = unknown_scores >= threshold
        if macro:
            ccr = np.mean(
                [
                    np.mean(known_correct[known_labels == label] & known_accepted[known_labels == label])
                    for label in known_classes
                ]
            )
            fpr = np.mean(
                [
                    np.mean(unknown_accepted[unknown_labels == label])
                    for label in unknown_classes
                ]
            )
        else:
            ccr = np.mean(known_correct & known_accepted)
            fpr = np.mean(unknown_accepted)
        ccr_values.append(float(ccr))
        fpr_values.append(float(fpr))
    # Thresholds descend, so FPR is non-decreasing. Duplicate x values are
    # valid vertical pieces and sklearn's trapezoidal auc handles them.
    return float(auc(np.asarray(fpr_values), np.asarray(ccr_values)))


def open_set_metrics(
    known_labels: np.ndarray,
    known_predictions: np.ndarray,
    known_scores: np.ndarray,
    unknown_labels: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
    known_class_names: list[str] | None = None,
    unknown_class_names: list[str] | None = None,
) -> dict:
    known_labels = np.asarray(known_labels, dtype=np.int64)
    known_predictions = np.asarray(known_predictions, dtype=np.int64)
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_labels = np.asarray(unknown_labels, dtype=np.int64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    if not (
        len(known_labels) == len(known_predictions) == len(known_scores)
        and len(unknown_labels) == len(unknown_scores)
    ):
        raise ValueError("Panjang label, prediction, dan score tidak konsisten.")
    known_correct = known_predictions == known_labels
    binary_labels = np.concatenate(
        (np.ones(len(known_scores), dtype=int), np.zeros(len(unknown_scores), dtype=int))
    )
    all_scores = np.concatenate((known_scores, unknown_scores))
    fpr, tpr, _ = roc_curve(binary_labels, all_scores)
    candidates = fpr[tpr >= 0.95]
    fpr95 = float(candidates.min()) if len(candidates) else 1.0
    known_names = known_class_names or [str(index) for index in np.unique(known_labels)]
    unknown_names = unknown_class_names or [str(index) for index in np.unique(unknown_labels)]
    unknown_acceptance = {
        unknown_names[int(class_index)]: float(
            np.mean(unknown_scores[unknown_labels == class_index] >= threshold)
        )
        for class_index in np.unique(unknown_labels)
    }
    return {
        "known_macro_f1": float(
            f1_score(
                known_labels,
                known_predictions,
                labels=list(range(len(known_names))),
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": float(roc_auc_score(binary_labels, all_scores)),
        "aupr_in": float(average_precision_score(binary_labels, all_scores)),
        "aupr_out": float(
            average_precision_score(1 - binary_labels, -all_scores)
        ),
        "fpr95": fpr95,
        "oscr": _oscr_area(
            known_scores,
            known_correct,
            known_labels,
            unknown_scores,
            unknown_labels,
            macro=False,
        ),
        "macro_oscr_research": _oscr_area(
            known_scores,
            known_correct,
            known_labels,
            unknown_scores,
            unknown_labels,
            macro=True,
        ),
        "threshold": float(threshold),
        "known_acceptance": float(np.mean(known_scores >= threshold)),
        "known_correct_acceptance": float(
            np.mean(known_correct & (known_scores >= threshold))
        ),
        "unknown_rejection": float(np.mean(unknown_scores < threshold)),
        "balanced_rejection_accuracy": float(
            0.5
            * (
                np.mean(known_scores >= threshold)
                + np.mean(unknown_scores < threshold)
            )
        ),
        "unknown_acceptance_per_class": unknown_acceptance,
    }
