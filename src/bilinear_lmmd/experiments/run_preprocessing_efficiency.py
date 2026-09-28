from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.engine.train import resolve_device
from bilinear_lmmd.experiments.preprocessing_contract import ARMS, CONFIGS
from bilinear_lmmd.modeling.models import build_model

def _load_probe_batch(canonical_root,clean_manifest,*,image_size,samples):
    manifest=json.loads(Path(clean_manifest).read_text(encoding="utf-8")); rows=sorted(manifest["images"],key=lambda row:row["identity"])[:samples]
    if not rows: raise RuntimeError("Clean manifest tidak memiliki image.")
    tensors=[]
    for row in rows:
        path=Path(canonical_root)/row["identity"]
        with Image.open(path) as image: tensors.append(TF.to_tensor(image.convert("RGB").resize((image_size,image_size))))
    return torch.stack(tensors)
def _sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)
def _time(fn,*,warmup,iterations,device):
    with torch.inference_mode():
        for _ in range(warmup): fn()
        _sync(device); start=time.perf_counter()
        for _ in range(iterations): fn()
        _sync(device)
    return (time.perf_counter()-start)*1000.0/iterations

def benchmark_preprocessing(canonical_root,clean_manifest,output,*,batch_sizes,warmup=10,iterations=50,samples=64,device_name="auto"):
    if warmup<0 or iterations<=0: raise ValueError("Warmup/iterations tidak valid.")
    device=resolve_device(device_name); base_cfg=load_config(CONFIGS["R0"]); image_size=int(base_cfg["data"]["image_size"]); probes=_load_probe_batch(canonical_root,clean_manifest,image_size=image_size,samples=max(samples,max(batch_sizes)))
    model_cfg=dict(base_cfg["model"]); model_cfg["pretrained"]=False; model=build_model(model_cfg).to(device).eval(); parameters=sum(p.numel() for p in model.parameters())
    results={"format":"bilinear_lmmd.preprocessing.efficiency.v1","device":str(device),"gpu_name":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"model_parameters":parameters,"model_size_fp32_mb":parameters*4/1024**2,"warmup":warmup,"iterations":iterations,"batch_sizes":{}}
    for batch_size in batch_sizes:
        if batch_size<=0: raise ValueError("Batch size harus positif.")
        cpu_batch=probes[:batch_size].contiguous(); results["batch_sizes"][str(batch_size)]={}
        for arm in ARMS:
            cfg=load_config(CONFIGS[arm]); runtime=PreprocessingRuntime.from_config(cfg["preprocessing"],device)
            pre=_time(lambda:runtime(cpu_batch),warmup=warmup,iterations=iterations,device=device); prepared=runtime(cpu_batch); cnn=_time(lambda:model(prepared),warmup=warmup,iterations=iterations,device=device); e2e=_time(lambda:model(runtime(cpu_batch)),warmup=warmup,iterations=iterations,device=device)
            results["batch_sizes"][str(batch_size)][arm]={"frontend_execution_device":"cpu_opencv_then_transfer" if arm=="C0" else ("identity_then_transfer" if arm=="R0" else str(device)),"pre_model_ms_batch":pre,"pre_model_ms_per_image":pre/batch_size,"cnn_ms_batch":cnn,"cnn_ms_per_image":cnn/batch_size,"end_to_end_ms_batch":e2e,"end_to_end_ms_per_image":e2e/batch_size,"throughput_images_per_second":batch_size*1000.0/e2e}
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8"); print(json.dumps(results,indent=2),flush=True); return results

def main():
    p=argparse.ArgumentParser(); p.add_argument("--canonical-root",required=True,type=Path); p.add_argument("--clean-manifest",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--batch-sizes",nargs="+",type=int,default=[1,16]); p.add_argument("--warmup",type=int,default=10); p.add_argument("--iterations",type=int,default=50); p.add_argument("--samples",type=int,default=64); p.add_argument("--device",default="auto"); a=p.parse_args(); benchmark_preprocessing(a.canonical_root,a.clean_manifest,a.output,batch_sizes=a.batch_sizes,warmup=a.warmup,iterations=a.iterations,samples=a.samples,device_name=a.device)
if __name__=="__main__": main()
