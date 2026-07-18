from pathlib import Path

from PIL import Image
from torch import nn

from bilinear_lmmd.engine import train_omsl as train_module
from bilinear_lmmd.modeling.models import ModelOutput


class TinyOntologyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, images):
        embedding = images.mean(dim=(2, 3))
        return ModelOutput(logits=self.classifier(embedding), embedding=embedding)


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (value, value, value)).save(path)


def test_train_omsl_smoke_writes_resumable_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    for split in ("train", "val"):
        _image(tmp_path / "fine" / split / "A" / "a.jpg", 20)
        _image(tmp_path / "fine" / split / "B" / "b.jpg", 220)
        _image(tmp_path / "coarse" / split / "AB" / "ab.jpg", 100)
    (tmp_path / "ontology.yaml").write_text(
        """
canonical_classes: [A, B]
datasets:
  fine: {labels: {A: [A], B: [B]}}
  coarse: {labels: {AB: [A, B]}}
""",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
seed: 42
device: cpu
ontology: {{path: ontology.yaml}}
data:
  image_size: 16
  batch_size: 2
  workers: 0
  pin_memory: false
  drop_last: false
  sources:
    - {{name: fine, root: "{(tmp_path / 'fine').as_posix()}"}}
    - {{name: coarse, root: "{(tmp_path / 'coarse').as_posix()}"}}
model: {{num_classes: 2, classifier: linear}}
taxonomy_contrastive: {{weight: 0.1}}
training:
  epochs: 1
  output_dir: "{output.as_posix()}"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_module, "build_model", lambda _: TinyOntologyModel())
    report = train_module.train_omsl(config)
    assert report["method"] == "OMSL-TC"
    assert report["best_epoch"] == 1
    for name in ("last.pt", "best.pt", "history.json", "metrics.json"):
        assert (output / name).is_file()
