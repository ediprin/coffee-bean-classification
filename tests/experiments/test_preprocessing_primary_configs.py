from bilinear_lmmd.experiments.preprocessing_contract import (
    ARMS,
    validate_primary_configs,
)


def test_primary_configs_differ_only_by_preprocessing_and_output():
    result = validate_primary_configs()
    assert set(result["configs"]) == set(ARMS)
    assert len(set(result["arm_config_sha256"].values())) == 4
    for arm, cfg in result["configs"].items():
        assert cfg["preprocessing"]["code"] == arm
        assert cfg["data"]["augmentation_mode"] == "preprocessing_study"
        assert cfg["model"]["backbone"] == "mobilenetv3_large_100"
        assert cfg["model"]["head"] == "gap"
        assert cfg["adaptation"]["method"] == "source_only"
        assert cfg["training"]["epochs"] == 50
