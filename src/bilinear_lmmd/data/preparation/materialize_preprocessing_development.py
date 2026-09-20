from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
FORMAT="bilinear_lmmd.coffee17.preprocessing_development.v1"
def _sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def _load(p:Path)->dict: return json.loads(Path(p).read_text(encoding="utf-8"))
def _link(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)

def materialize_preprocessing_development(canonical_root:Path,clean_manifest_path:Path,fold_manifest_path:Path,destination:Path,*,fold:int)->dict:
    canonical_root=Path(canonical_root).resolve(); destination=Path(destination).resolve()
    clean=_load(clean_manifest_path); folds=_load(fold_manifest_path); name=f"fold_{fold}"
    if folds.get("decision")!="PASS" or name not in folds.get("assignments",{}): raise RuntimeError("Fold manifest tidak valid")
    if folds.get("clean_content_sha256")!=clean.get("clean_content_sha256"): raise RuntimeError("Clean/fold fingerprint mismatch")
    ids={r["identity"]:r for r in clean["images"]}; splits=folds["assignments"][name]
    contract={"format":FORMAT,"decision":"PASS","fold":fold,"raw_content_sha256":clean["raw_content_sha256"],"clean_content_sha256":clean["clean_content_sha256"],"clean_manifest_file_sha256":_sha(Path(clean_manifest_path)),"fold_manifest_file_sha256":_sha(Path(fold_manifest_path)),"train_count":len(splits["train"]),"val_count":len(splits["val"]),"locked_test_count":len(splits["test"]),"test_images_extracted":False,"test_directory_absent":True,"model_accessed":False,"training_executed":False}
    cp=destination/"development_contract.json"
    if destination.exists() and any(destination.iterdir()):
        if cp.is_file() and _load(cp)==contract and not (destination/"source/test").exists(): return contract
        raise FileExistsError(f"Development destination berbeda/parsial: {destination}")
    for split in ("train","val"):
        for identity in splits[split]:
            row=ids[identity]; src=canonical_root/identity
            if not src.is_file() or _sha(src)!=row["sha256"]: raise RuntimeError(f"Source berubah: {identity}")
            _link(src,destination/"source"/split/identity)
    if (destination/"source/test").exists(): raise RuntimeError("Development tidak boleh punya test")
    cp.parent.mkdir(parents=True,exist_ok=True); cp.write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    return contract

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--canonical-root",required=True,type=Path); ap.add_argument("--clean-manifest",required=True,type=Path); ap.add_argument("--fold-manifest",required=True,type=Path); ap.add_argument("--destination",required=True,type=Path); ap.add_argument("--fold",required=True,type=int)
    a=ap.parse_args(); materialize_preprocessing_development(a.canonical_root,a.clean_manifest,a.fold_manifest,a.destination,fold=a.fold)
if __name__=="__main__": main()
