from pathlib import Path

from PIL import Image
import torch

from bilinear_lmmd.data.multisource import build_multisource_loaders
from bilinear_lmmd.modeling.ontology import load_ontology


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (value, value, value)).save(path)


def test_multisource_loader_maps_labels_and_balances_datasets(tmp_path: Path) -> None:
    for split in ("train", "val"):
        _image(tmp_path / "fine" / split / "A" / "a.jpg", 20)
        _image(tmp_path / "fine" / split / "B" / "b.jpg", 40)
        _image(tmp_path / "coarse" / split / "AB" / "ab.jpg", 60)
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        """
canonical_classes: [A, B]
datasets:
  fine:
    labels: {A: [A], B: [B]}
  coarse:
    labels: {AB: [A, B]}
""",
        encoding="utf-8",
    )
    ontology = load_ontology(ontology_path)
    loaders = build_multisource_loaders(
        {
            "image_size": 16,
            "batch_size": 2,
            "workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "balance_datasets": True,
            "sources": [
                {"name": "fine", "root": str(tmp_path / "fine")},
                {"name": "coarse", "root": str(tmp_path / "coarse")},
            ],
        },
        ontology,
        seed=42,
    )
    assert loaders.train_counts == {"fine": 2, "coarse": 1}
    weights = torch.as_tensor(loaders.train.sampler.weights)
    assert torch.isclose(weights[:2].sum(), weights[2:].sum())
    _, compatibility, dataset_indices, _ = next(iter(loaders.train))
    assert compatibility.shape[1] == 2
    assert dataset_indices.dtype == torch.int64
