from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets

from bilinear_lmmd.data.loaders import _transforms
from bilinear_lmmd.modeling.ontology import DatasetOntology, OntologySpec


class OntologyImageFolder(Dataset):
    """ImageFolder that returns an observed-label compatibility mask."""

    def __init__(
        self,
        root: str | Path,
        dataset_index: int,
        ontology: DatasetOntology,
        transform,
    ):
        self.root = Path(root)
        self.dataset_index = int(dataset_index)
        self.ontology = ontology
        self.dataset = datasets.ImageFolder(self.root, transform=transform)
        unknown = sorted(set(self.dataset.classes) - set(ontology.observed_labels))
        missing = sorted(set(ontology.observed_labels) - set(self.dataset.classes))
        if unknown or missing:
            raise ValueError(
                f"Kelas folder {self.root} tidak sama dengan ontology {ontology.name}. "
                f"unknown={unknown}, missing={missing}"
            )
        label_to_observed = ontology.label_to_index
        self.folder_target_to_observed = {
            folder_index: label_to_observed[label]
            for label, folder_index in self.dataset.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, folder_target = self.dataset[index]
        observed_target = self.folder_target_to_observed[int(folder_target)]
        compatibility = self.ontology.compatibility[observed_target]
        return (
            image,
            compatibility.clone(),
            self.dataset_index,
            observed_target,
        )


@dataclass(frozen=True)
class MultiSourceLoaders:
    train: DataLoader
    validation: dict[str, DataLoader]
    ontology: OntologySpec
    dataset_names: tuple[str, ...]
    train_counts: dict[str, int]


def _source_transform(data_cfg: dict, train: bool):
    return _transforms(
        int(data_cfg.get("image_size", 224)),
        train=train,
        rotation_angles=[float(x) for x in data_cfg.get("rotation_angles", [0])],
        object_crop=bool(data_cfg.get("object_crop", False)),
        object_crop_margin=float(data_cfg.get("object_crop_margin", 0.1)),
        augmentation_mode=str(data_cfg.get("augmentation_mode", "standard")),
    )


def build_multisource_loaders(
    data_cfg: dict,
    ontology: OntologySpec,
    *,
    seed: int,
) -> MultiSourceLoaders:
    sources = data_cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data.sources harus berupa list dataset yang tidak kosong.")
    names = tuple(str(source["name"]) for source in sources)
    if len(set(names)) != len(names):
        raise ValueError("Nama data.sources harus unik.")
    unknown = [name for name in names if name not in ontology.datasets]
    if unknown:
        raise ValueError(f"Dataset belum ada dalam ontology: {unknown}")

    train_datasets: list[OntologyImageFolder] = []
    validation: dict[str, DataLoader] = {}
    train_counts: dict[str, int] = {}
    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 32)),
        "num_workers": int(data_cfg.get("workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    for dataset_index, source in enumerate(sources):
        name = str(source["name"])
        root = Path(source["root"])
        train_root = root / str(source.get("train_split", "train"))
        val_root = root / str(source.get("val_split", "val"))
        missing = [str(path) for path in (train_root, val_root) if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Folder multi-source belum lengkap:\n- " + "\n- ".join(missing))
        train_dataset = OntologyImageFolder(
            train_root,
            dataset_index,
            ontology.datasets[name],
            _source_transform(data_cfg, train=True),
        )
        val_dataset = OntologyImageFolder(
            val_root,
            dataset_index,
            ontology.datasets[name],
            _source_transform(data_cfg, train=False),
        )
        train_datasets.append(train_dataset)
        train_counts[name] = len(train_dataset)
        validation[name] = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    combined = ConcatDataset(train_datasets)
    if bool(data_cfg.get("balance_datasets", True)):
        sample_weights: list[float] = []
        for dataset in train_datasets:
            sample_weights.extend([1.0 / len(dataset)] * len(dataset))
        generator = torch.Generator().manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(combined),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train = DataLoader(
        combined,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=bool(data_cfg.get("drop_last", True)),
        **loader_kwargs,
    )
    return MultiSourceLoaders(
        train=train,
        validation=validation,
        ontology=ontology,
        dataset_names=names,
        train_counts=train_counts,
    )
