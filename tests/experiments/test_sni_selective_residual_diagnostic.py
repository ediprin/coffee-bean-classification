from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.experiments.run_sni_selective_residual_diagnostic import (
    MODEL_CONFIGS,
    SEED,
    _audit_capacity,
    screening_decision,
)


def test_diagnostic_is_short_seed42_validation_only_and_capacity_matched():
    assert SEED == 42
    assert tuple(MODEL_CONFIGS) == ("SNIDG", "SNIDH")
    audits = _audit_capacity()
    assert audits["SNIDG"]["parameters"] == audits["SNIDH"]["parameters"]
    assert (
        audits["SNIDG"]["trainable_after_freeze"]
        == audits["SNIDH"]["trainable_after_freeze"]
    )
    for path in MODEL_CONFIGS.values():
        cfg = load_config(path)
        assert cfg["training"]["epochs"] == 10
        assert cfg["model"]["pretrained"] is False
        assert cfg["adaptation"]["method"] == "source_only"
        assert cfg["training"]["trainable_modules"] == [
            "bean_pool",
            "bean_residual_classifier",
            "residual_dropout",
        ]


def test_diagnostic_requires_all_three_validation_criteria():
    summary = {
        "macro_f1": {"delta_mean": 0.01},
        "hard_class_f1": {"delta_mean": 0.01},
        "worst_class_f1": {"delta_mean": -0.01},
    }
    assert screening_decision(summary)["decision"] == "PASS"
    summary["hard_class_f1"]["delta_mean"] = 0.0
    assert screening_decision(summary)["decision"] == "FAIL"
