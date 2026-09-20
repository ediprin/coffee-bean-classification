from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F
from torch import nn

from .af2 import afab_gate, minmax_spatial

@dataclass(frozen=True)
class WAV1Config:
    wavelet_levels:int=2; eps:float=1.0e-8
    @classmethod
    def from_mapping(cls,payload:"WAV1Config | dict[str, Any] | None")->"WAV1Config":
        result=payload if isinstance(payload,cls) else cls(**dict(payload or {}))
        if result.wavelet_levels<=0: raise ValueError("wavelet_levels harus positif")
        if result.eps<=0.0: raise ValueError("eps harus positif")
        return result
    def to_dict(self)->dict[str,Any]: return asdict(self)

def rgb_luminance(value:torch.Tensor)->torch.Tensor:
    if value.ndim!=4 or value.shape[1]!=3: raise ValueError("luminance membutuhkan BCHW RGB")
    coefficients=value.new_tensor((0.2126,0.7152,0.0722)).view(1,3,1,1); return (value*coefficients).sum(dim=1,keepdim=True)

def haar_dwt2(value:torch.Tensor)->tuple[torch.Tensor,tuple[int,int]]:
    if value.ndim!=4: raise ValueError("Haar DWT membutuhkan BCHW")
    h,w=value.shape[-2:]; pad_h,pad_w=h%2,w%2; work=F.pad(value,(0,pad_w,0,pad_h),mode="replicate") if pad_h or pad_w else value
    root=1.0/math.sqrt(2.0); low=value.new_tensor((root,root)); high=value.new_tensor((-root,root))
    filters=torch.stack((torch.outer(low,low),torch.outer(low,high),torch.outer(high,low),torch.outer(high,high)))
    channels=value.shape[1]; weight=filters[:,None].repeat(channels,1,1,1); output=F.conv2d(work,weight,stride=2,groups=channels); output=output.reshape(value.shape[0],channels,4,output.shape[-2],output.shape[-1]); return output,(h,w)

def haar_idwt2(bands:torch.Tensor,shape:tuple[int,int])->torch.Tensor:
    if bands.ndim!=5 or bands.shape[2]!=4: raise ValueError("Haar inverse membutuhkan BC4HW")
    root=1.0/math.sqrt(2.0); low=bands.new_tensor((root,root)); high=bands.new_tensor((-root,root)); filters=torch.stack((torch.outer(low,low),torch.outer(low,high),torch.outer(high,low),torch.outer(high,high)))
    b,channels,_,h,w=bands.shape; weight=filters[:,None].repeat(channels,1,1,1); packed=bands.reshape(b,channels*4,h,w); output=F.conv_transpose2d(packed,weight,stride=2,groups=channels); return output[...,:shape[0],:shape[1]]

class WAV1Frontend(nn.Module):
    def __init__(self,config:WAV1Config|dict[str,Any]|None=None)->None: super().__init__(); self.config=WAV1Config.from_mapping(config)
    def recover(self,value:torch.Tensor)->torch.Tensor:
        if value.ndim!=4 or value.shape[1]!=3: raise ValueError("WAV1 membutuhkan BCHW RGB")
        if not torch.is_floating_point(value): raise TypeError("WAV1 membutuhkan tensor floating point")
        luminance=rgb_luminance(value); target=luminance.shape[-2:]; current=luminance; energies=[]
        for _ in range(self.config.wavelet_levels):
            bands,_=haar_dwt2(current); current=bands[:,:,0]; detail=bands[:,:,1:].square().sum(2).add(self.config.eps).sqrt(); detail=F.interpolate(detail,size=target,mode="bilinear",align_corners=False); energies.append(minmax_spatial(detail,self.config.eps))
        cue=torch.stack(energies).mean(0); return cue.expand_as(value)
    def forward(self,value:torch.Tensor)->torch.Tensor: return afab_gate(value,self.recover(value),eps=self.config.eps)
