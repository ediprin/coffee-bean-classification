import pytest

from bilinear_lmmd.config import DEFAULTS
from bilinear_lmmd.train import classification_metrics


def test_classification_metrics_include_hard_tail():
    metrics = classification_metrics(
        labels=[0, 0, 1, 1, 2, 2],
        predictions=[0, 0, 1, 0, 2, 1],
        class_names=["easy", "hard_a", "hard_b"],
        hard_groups={"hard": ["hard_a", "hard_b"]},
    )
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["hard_class_f1"] == pytest.approx(7 / 12)
    assert metrics["worst_class_f1"] == pytest.approx(0.5)
    assert set(metrics["per_class"]) == {"easy", "hard_a", "hard_b"}


def test_predeclared_hard_subset_contains_exactly_eight_classes():
    groups = DEFAULTS["evaluation"]["hard_groups"]
    members = list(dict.fromkeys(name for group in groups.values() for name in group))
    assert members == [
        "Partial Black",
        "Partial Sour",
        "Full Sour",
        "Withered",
        "Immature",
        "Cut",
        "Slight Insect Damage",
        "Severe Insect Damage",
    ]
