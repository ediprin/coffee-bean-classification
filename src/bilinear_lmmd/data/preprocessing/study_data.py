from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], byteorder="big", signed=False)

def identity_from_path(path: str | Path) -> str:
    path = Path(path); return f"{path.parent.name}/{path.name}"

def deterministic_rotation_angle(*, seed:int, epoch:int, identity:str, angles:list[float])->float:
    values=angles or [0.0]; index=_stable_u64(f"{seed}|{epoch}|{identity}|rotation")%len(values); return float(values[index])

class PreprocessingStudyImageFolder(datasets.ImageFolder):
    def __init__(self, root:str|Path, *, image_size:int, train:bool, seed:int, rotation_angles:list[float])->None:
        super().__init__(root=str(root), transform=None); self.image_size=int(image_size); self.train_mode=bool(train); self.base_seed=int(seed); self.rotation_angles=[float(value) for value in rotation_angles] or [0.0]; self.epoch=0; self.resize=transforms.Resize((self.image_size,self.image_size))
    def set_epoch(self,epoch:int)->None: self.epoch=int(epoch)
    def rotation_for_index(self,index:int)->float:
        path,_=self.samples[index]; return deterministic_rotation_angle(seed=self.base_seed,epoch=self.epoch,identity=identity_from_path(path),angles=self.rotation_angles)
    def __getitem__(self,index:int):
        path,target=self.samples[index]; image:Image.Image=self.loader(path).convert("RGB")
        if self.train_mode: image=TF.rotate(image,self.rotation_for_index(index),fill=255)
        image=self.resize(image); return TF.to_tensor(image),target

class EpochDeterministicSampler(Sampler[int]):
    def __init__(self,dataset,seed:int)->None: self.dataset=dataset; self.seed=int(seed); self.epoch=0
    def set_epoch(self,epoch:int)->None: self.epoch=int(epoch)
    def __iter__(self)->Iterator[int]:
        generator=torch.Generator(); generator.manual_seed(_stable_u64(f"{self.seed}|{self.epoch}|shuffle")%(2**63-1)); yield from torch.randperm(len(self.dataset),generator=generator).tolist()
    def __len__(self)->int: return len(self.dataset)

@dataclass(frozen=True)
class StudyLoaders:
    train:DataLoader; val:DataLoader; classes:list[str]

def build_preprocessing_study_loaders(data_cfg:dict,*,seed:int)->StudyLoaders:
    root=Path(data_cfg["root"]); source=root/str(data_cfg.get("source","source")); train_name=str(data_cfg.get("train_split","train")); val_name=str(data_cfg.get("val_split","val")); train_root=source/train_name; val_root=source/val_name
    if (source/"test").exists(): raise RuntimeError("Development runtime preprocessing-study tidak boleh mengekspos source/test.")
    for path in (train_root,val_root):
        if not path.is_dir(): raise FileNotFoundError(f"Split development tidak ditemukan: {path}")
    if bool(data_cfg.get("object_crop",False)): raise ValueError("Primary preprocessing-study mengunci object_crop=false.")
    image_size=int(data_cfg.get("image_size",224)); rotation_angles=[float(value) for value in data_cfg.get("rotation_angles",[0,45,90,135,180,225,270])]
    train_dataset=PreprocessingStudyImageFolder(train_root,image_size=image_size,train=True,seed=seed,rotation_angles=rotation_angles)
    val_dataset=PreprocessingStudyImageFolder(val_root,image_size=image_size,train=False,seed=seed,rotation_angles=[0.0])
    if train_dataset.class_to_idx!=val_dataset.class_to_idx: raise ValueError("Pemetaan kelas train/val berbeda.")
    sampler=EpochDeterministicSampler(train_dataset,seed); common={"batch_size":int(data_cfg.get("batch_size",32)),"num_workers":int(data_cfg.get("workers",4)),"pin_memory":True}
    train_loader=DataLoader(train_dataset,sampler=sampler,shuffle=False,drop_last=True,**common); val_loader=DataLoader(val_dataset,shuffle=False,drop_last=False,**common)
    return StudyLoaders(train=train_loader,val=val_loader,classes=train_dataset.classes)

def set_study_epoch(loader:DataLoader,epoch:int)->None:
    if hasattr(loader.dataset,"set_epoch"): loader.dataset.set_epoch(epoch)
    sampler=getattr(loader,"sampler",None)
    if hasattr(sampler,"set_epoch"): sampler.set_epoch(epoch)

def build_preprocessing_evaluation_loader(data_cfg:dict,*,split:str,seed:int)->tuple[DataLoader,list[str]]:
    root=Path(data_cfg["root"]); source=root/str(data_cfg.get("source","source")); split_root=source/split
    if not split_root.is_dir(): raise FileNotFoundError(f"Evaluation split tidak ditemukan: {split_root}")
    dataset=PreprocessingStudyImageFolder(split_root,image_size=int(data_cfg.get("image_size",224)),train=False,seed=seed,rotation_angles=[0.0])
    loader=DataLoader(dataset,batch_size=int(data_cfg.get("batch_size",32)),num_workers=int(data_cfg.get("workers",4)),pin_memory=True,shuffle=False,drop_last=False)
    return loader,dataset.classes
