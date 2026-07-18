from importlib import import_module
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_source_modules_are_grouped_by_concern() -> None:
    package_root = REPOSITORY_ROOT / "src" / "bilinear_lmmd"
    root_modules = sorted(path.name for path in package_root.glob("*.py"))
    assert root_modules == ["__init__.py"]

    expected_packages = {
        "analysis",
        "core",
        "data",
        "engine",
        "experiments",
        "modeling",
        "reporting",
    }
    actual_packages = {
        path.name
        for path in package_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    assert expected_packages.issubset(actual_packages)


def test_configs_are_grouped_by_dataset_or_study() -> None:
    config_root = REPOSITORY_ROOT / "configs"
    assert list(config_root.glob("*.yaml")) == []
    assert {
        "backbones",
        "cbd",
        "coffee17",
        "granularity",
        "paper",
        "roast",
        "usk",
    }.issubset({path.name for path in config_root.iterdir() if path.is_dir()})


def test_primary_command_modules_remain_importable() -> None:
    modules = (
        "bilinear_lmmd.engine.train",
        "bilinear_lmmd.engine.evaluate_checkpoint",
        "bilinear_lmmd.data.preparation.prepare_coffee17",
        "bilinear_lmmd.experiments.run_backbone_screening",
        "bilinear_lmmd.reporting.aggregate_ablation",
    )
    for module_name in modules:
        assert import_module(module_name) is not None
