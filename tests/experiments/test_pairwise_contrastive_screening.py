from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_pairwise_contrastive_screening import (
    screening_decision,
)


def _summary(macro: float, hard: float, worst: float) -> dict:
    return {
        "macro_f1": {"delta_mean": macro},
        "hard_class_f1": {"delta_mean": hard},
        "worst_class_f1": {"delta_mean": worst},
    }


def test_pairwise_gate_requires_all_metrics():
    assert screening_decision(_summary(0.01, 0.01, -0.01))["decision"] == "PASS"
    assert screening_decision(_summary(-0.01, 0.01, 0.01))["decision"] == "FAIL"


def test_p1_p2_only_differ_in_confusion_aware_fields():
    p1 = load_config("configs/finegrained/CP1_efficientnetv2_gap_supcon.yaml")
    p2 = load_config("configs/finegrained/CP2_efficientnetv2_gap_confusion_pairwise.yaml")
    assert p1["model"] == p2["model"]
    ignored = {"pairwise_mode", "confusion_strength", "output_dir"}
    for key, value in p1["training"].items():
        if key not in ignored:
            assert value == p2["training"][key]
    assert p1["training"]["pairwise_mode"] == "standard"
    assert p2["training"]["pairwise_mode"] == "confusion_aware"
