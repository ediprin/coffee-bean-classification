from __future__ import annotations
import hashlib,json,random,subprocess
from pathlib import Path
from typing import Any
import numpy as np
import torch

def canonical_json_sha256(value: Any)->str:
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: str|Path)->str:
    digest=hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def model_state_fingerprint(model: torch.nn.Module)->str:
    digest=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous(); digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def seed_everything(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def capture_rng_state()->dict:
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.random.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng_state(state:dict)->None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None: torch.cuda.set_rng_state_all(state["torch_cuda"])

def current_git_commit(repo_root: str|Path|None=None)->str|None:
    cwd=Path(repo_root).resolve() if repo_root is not None else None
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=cwd,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None
