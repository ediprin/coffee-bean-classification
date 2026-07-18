from pathlib import Path

import pytest
import torch

from bilinear_lmmd.modeling.ontology import compatible_leaf_mask, load_ontology


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_ontology_builds_binary_observation_matrix(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ontology.yaml",
        """
canonical_classes: [Full Black, Partial Black, Sour]
datasets:
  coarse:
    labels:
      Black: [Full Black, Partial Black]
      Sour: [Sour]
""",
    )
    ontology = load_ontology(path)
    assert ontology.canonical_classes == ("Full Black", "Partial Black", "Sour")
    assert torch.equal(
        compatible_leaf_mask(ontology, "coarse", "Black"),
        torch.tensor([True, True, False]),
    )


def test_provisional_mapping_requires_explicit_permission(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ontology.yaml",
        """
canonical_classes: [A, B]
datasets:
  source:
    labels:
      parent: {leaves: [A, B], status: provisional}
""",
    )
    with pytest.raises(ValueError, match="provisional"):
        load_ontology(path)
    assert load_ontology(path, allow_provisional=True).datasets["source"].statuses == (
        "provisional",
    )


def test_overlapping_observed_labels_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ontology.yaml",
        """
canonical_classes: [A, B]
datasets:
  source:
    labels:
      parent: [A, B]
      child: [A]
""",
    )
    with pytest.raises(ValueError, match="tumpang tindih"):
        load_ontology(path)
