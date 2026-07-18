from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor


ALLOWED_MAPPING_STATUS = {"exact", "reviewed", "provisional"}


@dataclass(frozen=True)
class DatasetOntology:
    name: str
    observed_labels: tuple[str, ...]
    compatibility: Tensor
    statuses: tuple[str, ...]

    @property
    def label_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.observed_labels)}


@dataclass(frozen=True)
class OntologySpec:
    canonical_classes: tuple[str, ...]
    datasets: dict[str, DatasetOntology]

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.canonical_classes)}


def _mapping_payload(value: Any) -> tuple[list[str], str]:
    if isinstance(value, list):
        return [str(item) for item in value], "reviewed"
    if not isinstance(value, dict):
        raise ValueError("Mapping label harus berupa list leaf atau dictionary.")
    leaves = value.get("leaves")
    if not isinstance(leaves, list):
        raise ValueError("Mapping dictionary membutuhkan field 'leaves' berupa list.")
    status = str(value.get("status", "provisional")).lower()
    if status not in ALLOWED_MAPPING_STATUS:
        raise ValueError(
            f"Status mapping {status!r} tidak valid; gunakan {sorted(ALLOWED_MAPPING_STATUS)}."
        )
    return [str(item) for item in leaves], status


def load_ontology(
    path: str | Path,
    *,
    allow_provisional: bool = False,
) -> OntologySpec:
    """Load and validate a fixed observed-label to canonical-leaf ontology.

    Within one dataset the observed labels must form a non-overlapping partition
    of the canonical leaves that it annotates. Coverage may be partial because a
    dataset is not required to contain every canonical class.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    canonical = tuple(str(item) for item in payload.get("canonical_classes", []))
    if len(canonical) < 2 or len(set(canonical)) != len(canonical):
        raise ValueError("canonical_classes harus unik dan berisi minimal dua kelas.")
    canonical_index = {name: index for index, name in enumerate(canonical)}

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("Ontology membutuhkan mapping 'datasets' yang tidak kosong.")

    datasets: dict[str, DatasetOntology] = {}
    for dataset_name, dataset_payload in raw_datasets.items():
        labels = (dataset_payload or {}).get("labels")
        if not isinstance(labels, dict) or not labels:
            raise ValueError(f"Dataset {dataset_name!r} tidak memiliki mapping labels.")
        observed_labels = tuple(str(name) for name in labels)
        compatibility = torch.zeros(
            (len(observed_labels), len(canonical)), dtype=torch.bool
        )
        statuses: list[str] = []
        assigned: dict[str, str] = {}
        for observed_index, (observed_name, value) in enumerate(labels.items()):
            leaves, status = _mapping_payload(value)
            if not leaves or len(set(leaves)) != len(leaves):
                raise ValueError(
                    f"Mapping {dataset_name}/{observed_name} kosong atau berduplikat."
                )
            unknown = [leaf for leaf in leaves if leaf not in canonical_index]
            if unknown:
                raise ValueError(
                    f"Leaf tidak dikenal pada {dataset_name}/{observed_name}: {unknown}"
                )
            if status == "provisional" and not allow_provisional:
                raise ValueError(
                    f"Mapping provisional ditolak: {dataset_name}/{observed_name}. "
                    "Audit mapping atau aktifkan allow_provisional secara eksplisit."
                )
            for leaf in leaves:
                if leaf in assigned:
                    raise ValueError(
                        f"Leaf {leaf!r} tumpang tindih pada dataset {dataset_name!r}: "
                        f"{assigned[leaf]!r} dan {observed_name!r}."
                    )
                assigned[leaf] = str(observed_name)
                compatibility[observed_index, canonical_index[leaf]] = True
            statuses.append(status)
        datasets[str(dataset_name)] = DatasetOntology(
            name=str(dataset_name),
            observed_labels=observed_labels,
            compatibility=compatibility,
            statuses=tuple(statuses),
        )
    return OntologySpec(canonical_classes=canonical, datasets=datasets)


def compatible_leaf_mask(
    ontology: OntologySpec,
    dataset_name: str,
    observed_label: str,
) -> Tensor:
    if dataset_name not in ontology.datasets:
        raise KeyError(f"Dataset tidak ada dalam ontology: {dataset_name}")
    dataset = ontology.datasets[dataset_name]
    try:
        index = dataset.label_to_index[observed_label]
    except KeyError as exc:
        raise KeyError(
            f"Label {observed_label!r} tidak dipetakan untuk {dataset_name!r}."
        ) from exc
    return dataset.compatibility[index].clone()
