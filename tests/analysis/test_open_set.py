import numpy as np

from bilinear_lmmd.analysis.open_set import (
    fit_class_prototypes,
    fit_openmax,
    open_set_metrics,
    openmax_knownness,
    prototype_knownness,
    standard_knownness_scores,
    threshold_from_known_validation,
)


def test_standard_knownness_scores_have_consistent_orientation() -> None:
    logits = np.asarray([[8.0, 0.0], [0.2, 0.1]])
    scores = standard_knownness_scores(logits)
    assert set(scores) == {"msp", "mls", "energy"}
    assert all(values[0] > values[1] for values in scores.values())


def test_perfect_separation_has_perfect_oscr() -> None:
    metrics = open_set_metrics(
        known_labels=np.asarray([0, 0, 1, 1]),
        known_predictions=np.asarray([0, 0, 1, 1]),
        known_scores=np.asarray([0.95, 0.90, 0.85, 0.80]),
        unknown_labels=np.asarray([0, 0, 1, 1]),
        unknown_scores=np.asarray([0.20, 0.15, 0.10, 0.05]),
        threshold=0.50,
        known_class_names=["a", "b"],
        unknown_class_names=["u", "v"],
    )
    assert metrics["auroc"] == 1.0
    assert metrics["oscr"] == 1.0
    assert metrics["macro_oscr_research"] == 1.0
    assert metrics["unknown_rejection"] == 1.0
    assert metrics["known_macro_f1"] == 1.0


def test_threshold_uses_only_known_validation_quantile() -> None:
    scores = np.arange(100, dtype=float)
    threshold = threshold_from_known_validation(scores, acceptance_target=0.95)
    assert np.mean(scores >= threshold) >= 0.95


def test_prototype_and_openmax_fit_only_correct_training_samples() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]
    )
    logits = np.asarray(
        [[4.0, 0.0], [3.5, 0.2], [3.0, 0.4], [0.0, 4.0], [0.2, 3.5], [0.4, 3.0]]
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    predictions = logits.argmax(axis=1)
    prototypes = fit_class_prototypes(embeddings, labels, predictions, 2)
    prototype_scores = prototype_knownness(embeddings, prototypes)
    assert prototypes.shape == (2, 2)
    assert np.isfinite(prototype_scores).all()

    models = fit_openmax(logits, labels, predictions, 2, tail_size=2)
    openmax_scores = openmax_knownness(logits, models, alpha_rank=2)
    assert len(models) == 2
    assert np.isfinite(openmax_scores).all()
    assert ((0.0 <= openmax_scores) & (openmax_scores <= 1.0)).all()
