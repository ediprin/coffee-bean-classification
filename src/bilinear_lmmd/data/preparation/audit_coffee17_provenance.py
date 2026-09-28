from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image
from bilinear_lmmd.data.preparation.prepare_coffee17 import CLASS_ALIASES, DATASET_URL, EXPECTED_COUNTS, IMAGE_SUFFIXES

FORMAT="bilinear_lmmd.coffee17.provenance.v1"

def _sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def _sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
def _json_sha(v:Any)->str:
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def audit_coffee17_provenance(archive:Path, output_dir:Path, *, canonical_root:Path|None=None, expected_counts:dict[str,int]|None=None)->dict:
    archive=Path(archive).expanduser().resolve(); output_dir=Path(output_dir).expanduser().resolve()
    expected=dict(expected_counts or EXPECTED_COUNTS)
    if not archive.is_file() or not zipfile.is_zipfile(archive): raise FileNotFoundError(f"ZIP Coffee17 tidak valid: {archive}")
    source_names=set(expected)|{a for a,c in CLASS_ALIASES.items() if c in expected}
    records=[]; payload={}; seen={}; ignored=0
    with zipfile.ZipFile(archive) as z:
        for item in sorted(z.infolist(), key=lambda x:x.filename.lower()):
            if item.is_dir(): continue
            path=Path(item.filename)
            if path.suffix.lower() not in IMAGE_SUFFIXES: ignored+=1; continue
            matches={CLASS_ALIASES.get(part,part) for part in path.parts[:-1] if part in source_names}
            if not matches: ignored+=1; continue
            if len(matches)!=1: raise ValueError(f"Path kelas ambigu: {item.filename}")
            cls=next(iter(matches)); identity=f"{cls}/{path.name}"
            if identity in seen: raise ValueError(f"Filename collision canonical: {identity}")
            data=z.read(item)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.load(); w,h=im.size; mode=im.mode
            except Exception as e: raise ValueError(f"Gambar gagal didecode: {item.filename}") from e
            records.append({"identity":identity,"class":cls,"filename":path.name,"sha256":_sha_bytes(data),"bytes":len(data),"width":int(w),"height":int(h),"mode":str(mode)})
            payload[identity]=data; seen[identity]=item.filename
    records.sort(key=lambda r:(r["class"],r["filename"],r["sha256"]))
    counts=dict(sorted(Counter(r["class"] for r in records).items()))
    if counts!=expected: raise ValueError(f"Class-count berbeda: {counts} != {expected}")
    compact=[{k:r[k] for k in ("class","filename","sha256","bytes","width","height")} for r in records]
    content_sha=_json_sha(compact)
    manifest={"format":"bilinear_lmmd.coffee17.raw_manifest.v1","dataset_slug":"sujitraarw/coffee-green-bean-with-17-defects-original","raw_content_sha256":content_sha,"images":records}
    prov={"format":FORMAT,"decision":"PASS","dataset_slug":"sujitraarw/coffee-green-bean-with-17-defects-original","dataset_url":DATASET_URL,"archive":str(archive),"archive_sha256":_sha_file(archive),"raw_content_sha256":content_sha,"raw_manifest_sha256":_json_sha(manifest),"image_count":len(records),"class_count":len(expected),"class_counts":expected,"class_aliases":dict(CLASS_ALIASES),"dimension_counts":dict(sorted(Counter(f'{r["width"]}x{r["height"]}' for r in records).items())),"ignored_noncanonical_files":ignored,"split_created":False,"model_accessed":False,"training_executed":False,"test_images_accessed_by_model":False}
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"coffee17_raw_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (output_dir/"coffee17_provenance.json").write_text(json.dumps(prov,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if canonical_root is not None:
        canonical_root=Path(canonical_root).expanduser().resolve()
        if canonical_root.exists() and any(canonical_root.iterdir()):
            existing=sorted(p for p in canonical_root.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if len(existing)!=len(records): raise FileExistsError(f"Canonical root existing berisi {len(existing)} image, diharapkan {len(records)}: {canonical_root}")
            expected_by_identity={r["identity"]:r for r in records}
            for p in existing:
                identity=f"{p.parent.name}/{p.name}"; row=expected_by_identity.get(identity)
                if row is None or _sha_file(p)!=row["sha256"]: raise RuntimeError(f"Canonical root existing berbeda dari archive provenance: {identity}")
            print(f"REUSE CANONICAL ROOT: {canonical_root}", flush=True)
        else:
            for r in records:
                dst=canonical_root/r["class"]/r["filename"]; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(payload[r["identity"]])
    return prov

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--archive",required=True,type=Path); ap.add_argument("--output-dir",required=True,type=Path); ap.add_argument("--canonical-root",type=Path)
    a=ap.parse_args(); audit_coffee17_provenance(a.archive,a.output_dir,canonical_root=a.canonical_root)
if __name__=="__main__": main()
