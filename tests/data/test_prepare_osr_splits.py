from pathlib import Path

import yaml

from bilinear_lmmd.data.preparation.prepare_osr_splits import prepare_osr_splits


CLASSES = [
    "Broken",
    "Cut",
    "Dry Cherry",
    "Fade",
    "Floater",
    "Full Black",
    "Full Sour",
    "Fungus Damage",
    "Husk",
    "Immature",
    "Parchment",
    "Partial Black",
    "Partial Sour",
    "Severe Insect Damage",
    "Shell",
    "Slight Insect Damage",
    "Withered",
]


def _source(root: Path) -> Path:
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            path = root / "source" / split / class_name / f"{split}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{split}:{class_name}".encode())
    return root


def _protocol(path: Path) -> Path:
    payload = {
        "protocol_id": "test-osr-v1",
        "version": 1,
        "status": "frozen",
        "dataset": {
            "expected_classes": 17,
            "unknown_classes_per_split": 3,
            "balanced_test_samples_per_class": 1,
            "known_acceptance_target": 0.95,
            "balance_seed": 7,
        },
        "scores": {
            "energy_temperature": 1.0,
            "openmax_tail_size": 2,
            "openmax_alpha_rank": 2,
            "openmax_distance": "eucos",
        },
        "splits": {
            "near": {"unknown_classes": CLASSES[:3], "rationale": "near"},
            "medium": {"unknown_classes": CLASSES[3:6], "rationale": "medium"},
            "far": {"unknown_classes": CLASSES[6:9], "rationale": "far"},
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_prepare_osr_splits_prevents_unknown_training_leakage(tmp_path: Path) -> None:
    source = _source(tmp_path / "coffee17")
    protocol = _protocol(tmp_path / "protocol.yaml")
    output = tmp_path / "prepared"

    audit = prepare_osr_splits(source, output, protocol)

    assert audit["status"] == "complete"
    for tier, values in audit["splits"].items():
        assert len(values["known_classes"]) == 14
        assert len(values["unknown_classes"]) == 3
        assert values["balanced_counts"] == {
            "known": 14,
            "unknown": 3,
            "per_class": 1,
        }
        for class_name in values["unknown_classes"]:
            assert not (output / tier / "source" / "train" / class_name).exists()
            assert not (output / tier / "source" / "val" / class_name).exists()
            assert (output / tier / "unknown" / "test" / class_name).is_dir()


def test_prepare_osr_splits_is_idempotent_for_same_protocol(tmp_path: Path) -> None:
    source = _source(tmp_path / "coffee17")
    protocol = _protocol(tmp_path / "protocol.yaml")
    output = tmp_path / "prepared"
    first = prepare_osr_splits(source, output, protocol)
    second = prepare_osr_splits(source, output, protocol)
    assert first == second
