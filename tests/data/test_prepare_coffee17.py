from pathlib import Path

from PIL import Image

from bilinear_lmmd.data.preparation.prepare_coffee17 import discover_directory_samples


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_discover_directory_samples_finds_nested_classes_and_aliases(tmp_path):
    raw = tmp_path / "kaggle-input"
    _image(raw / "owner" / "dataset" / "Broken" / "a.png", (1, 2, 3))
    _image(raw / "owner" / "dataset" / "Broken" / "b.png", (2, 3, 4))
    _image(
        raw / "owner" / "dataset" / "Fungus Damange" / "c.png",
        (3, 4, 5),
    )
    _image(raw / "other-dataset" / "Dark" / "ignored.png", (4, 5, 6))

    samples = discover_directory_samples(
        raw,
        expected_counts={"Broken": 2, "Fungus Damage": 1},
    )

    assert set(samples) == {"Broken", "Fungus Damage"}
    assert [path.name for path in samples["Broken"]] == ["a.png", "b.png"]
    assert samples["Fungus Damage"][0].name == "c.png"
