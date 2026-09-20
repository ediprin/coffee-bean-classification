from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
FORMAT="bilinear_lmmd.coffee17.preprocessing_test_runtime.v1"
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

def materialize_preprocessing_test(canonical_root:Path,clean_manifest_path:Path,fold_manifest_path:Path,authority_path:Path,destination:Path,*,fold:int,authorize_test:bool=False)->dict:
    if not authorize_test: raise RuntimeError("Outer-test belum diotorisasi secara eksplisit")
    clean=_load(clean_manifest_path); folds=_load(fold_manifest_path); authority=_load(authority_path); name=f"fold_{fold}"
    if authority.get("decision")!="AUTHORIZE_OOF_TEST_EVALUATION" or authority.get("test_images_accessed") is not False or authority.get("further_primary_tuning_authorized") is not False: raise RuntimeError("Authority OOF tidak valid")
    if name not in folds.get("assignments",{}): raise RuntimeError("Fold tidak valid")
    if folds.get("clean_content_sha256")!=clean.get("clean_content_sha256"): raise RuntimeError("Clean/fold fingerprint mismatch")
    if authority.get("clean_content_sha256") not in (None,clean["clean_content_sha256"]): raise RuntimeError("Authority clean population mismatch")
    ids={r["identity"]:r for r in clean["images"]}; test=folds["assignments"][name]["test"]; destination=Path(destination).resolve()
    contract={"format":FORMAT,"decision":"PASS","fold":fold,"clean_content_sha256":clean["clean_content_sha256"],"clean_manifest_file_sha256":_sha(Path(clean_manifest_path)),"fold_manifest_file_sha256":_sha(Path(fold_manifest_path)),"authority_sha256":_sha(Path(authority_path)),"test_count":len(test),"training_executed":False,"test_images_accessed":True}
    cp=destination/"test_contract.json"
    if destination.exists() and any(destination.iterdir()):
        if cp.is_file() and _load(cp)==contract: return contract
        raise FileExistsError(f"Test destination berbeda/parsial: {destination}")
    root=Path(canonical_root).resolve()
    for identity in test:
        row=ids[identity]; src=root/identity
        if not src.is_file() or _sha(src)!=row["sha256"]: raise RuntimeError(f"Test source berubah: {identity}")
        _link(src,destination/"source/test"/identity)
    cp.parent.mkdir(parents=True,exist_ok=True); cp.write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    return contract

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--canonical-root",required=True,type=Path); ap.add_argument("--clean-manifest",required=True,type=Path); ap.add_argument("--fold-manifest",required=True,type=Path); ap.add_argument("--authority",required=True,type=Path); ap.add_argument("--destination",required=True,type=Path); ap.add_argument("--fold",required=True,type=int); ap.add_argument("--authorize-test",action="store_true")
    a=ap.parse_args(); materialize_preprocessing_test(a.canonical_root,a.clean_manifest,a.fold_manifest,a.authority,a.destination,fold=a.fold,authorize_test=a.authorize_test)
if __name__=="__main__": main()
