from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from PIL import Image
from bilinear_lmmd.data.preparation.prepare_coffee17 import IMAGE_SUFFIXES

FORMAT="bilinear_lmmd.coffee17.preprocessing_folds.v1"
def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def _json_sha(v:Any)->str:
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _load(p:Path)->dict: return json.loads(Path(p).read_text(encoding="utf-8"))
def _records(root:Path)->list[dict]:
    out=[]
    for c in sorted(x for x in Path(root).iterdir() if x.is_dir()):
        for p in sorted(c.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES: continue
            with Image.open(p) as im: im.load(); w,h=im.size
            out.append({"identity":f"{c.name}/{p.name}","class":c.name,"filename":p.name,"sha256":_sha_file(p),"bytes":p.stat().st_size,"width":int(w),"height":int(h)})
    return sorted(out,key=lambda r:r["identity"])
def _content_sha(rs:list[dict])->str:
    return _json_sha([{k:r[k] for k in ("class","filename","sha256","bytes","width","height")} for r in rs])
def _clean(rs:list[dict]):
    groups=defaultdict(list)
    for r in rs: groups[r["sha256"]].append(r)
    kept=[]; dups=[]; conflicts=[]
    for digest,g in sorted(groups.items()):
        g=sorted(g,key=lambda r:r["identity"]); classes=sorted({r["class"] for r in g})
        if len(g)==1: kept.append(g[0])
        elif len(classes)==1:
            kept.append(g[0]); dups.append({"sha256":digest,"class":classes[0],"kept":g[0]["identity"],"removed":[r["identity"] for r in g[1:]]})
        else:
            conflicts.append({"sha256":digest,"classes":classes,"quarantined":[r["identity"] for r in g]})
    return sorted(kept,key=lambda r:r["identity"]),dups,conflicts
def _assign(clean:list[dict],folds:int,seed:int,val_ratio:float):
    if folds<3: raise ValueError("folds minimal 3")
    if not 0<val_ratio<1/folds: raise ValueError("validation_ratio harus < 1/folds")
    by=defaultdict(list)
    for r in clean: by[r["class"]].append(r["identity"])
    if any(len(v)<folds for v in by.values()): raise ValueError("Setiap kelas perlu >= folds")
    out={f"fold_{i}":{"train":[],"val":[],"test":[]} for i in range(1,folds+1)}; seen=set()
    for cls,ids in sorted(by.items()):
        ids=sorted(ids); random.Random(f"{seed}:{cls}:outer").shuffle(ids); bins=[ids[i::folds] for i in range(folds)]
        for fi in range(folds):
            test=bins[fi]; remain=[x for x in ids if x not in set(test)]
            random.Random(f"{seed}:{cls}:{fi}:validation").shuffle(remain)
            n=max(1,round(len(ids)*val_ratio))
            cur={"train":remain[n:],"val":remain[:n],"test":test}
            for s,vals in cur.items(): out[f"fold_{fi+1}"][s].extend(vals)
            for x in test:
                if x in seen: raise RuntimeError(f"test duplicate {x}")
                seen.add(x)
    expected={r["identity"] for r in clean}
    if seen!=expected: raise RuntimeError("Outer-test union tidak sama dengan clean population")
    for fold,splits in out.items():
        for s in splits: splits[s]=sorted(splits[s])
        sets={s:set(v) for s,v in splits.items()}
        if sets["train"]&sets["val"] or sets["train"]&sets["test"] or sets["val"]&sets["test"]: raise RuntimeError(f"{fold}: overlap")
        for cls in by:
            for s in ("train","val","test"):
                if not any(x.startswith(cls+"/") for x in splits[s]): raise RuntimeError(f"{fold}/{s} kehilangan kelas {cls}")
    return out

def prepare_preprocessing_folds(canonical_root:Path,provenance_path:Path,output_dir:Path,*,folds:int=5,seed:int=42,validation_ratio:float=.10)->dict:
    prov=_load(provenance_path)
    if prov.get("decision")!="PASS" or prov.get("split_created") is not False: raise RuntimeError("Provenance tidak valid")
    raw=_records(Path(canonical_root))
    if _content_sha(raw)!=prov.get("raw_content_sha256"): raise RuntimeError("Canonical content != provenance")
    counts=dict(sorted(Counter(r["class"] for r in raw).items()))
    if counts!=prov.get("class_counts"): raise RuntimeError("Class counts berubah")
    clean,dups,conflicts=_clean(raw); clean_sha=_content_sha(clean); assignments=_assign(clean,folds,seed,validation_ratio)
    cm={"format":"bilinear_lmmd.coffee17.clean_manifest.v1","raw_content_sha256":prov["raw_content_sha256"],"clean_content_sha256":clean_sha,"raw_count":len(raw),"clean_count":len(clean),"removed_same_class_count":sum(len(x["removed"]) for x in dups),"quarantined_conflict_count":sum(len(x["quarantined"]) for x in conflicts),"clean_class_counts":dict(sorted(Counter(r["class"] for r in clean).items())),"same_class_duplicate_groups":dups,"label_conflict_groups":conflicts,"policy":"Keep one lexicographic canonical file for exact same-class hashes; quarantine every identity in an exact cross-class hash group.","images":clean}
    fm={"format":FORMAT,"decision":"PASS","raw_content_sha256":prov["raw_content_sha256"],"clean_content_sha256":clean_sha,"clean_manifest_sha256":_json_sha(cm),"folds":folds,"seed":seed,"validation_ratio":validation_ratio,"assignments":assignments,"test_materialized":False,"model_accessed":False,"training_executed":False}
    summary={"format":"bilinear_lmmd.coffee17.preprocessing_folds.summary.v1","decision":"PASS_COFFEE17_PREPROCESSING_DATA_GATE","raw_count":len(raw),"clean_count":len(clean),"class_count":len(cm["clean_class_counts"]),"raw_content_sha256":prov["raw_content_sha256"],"clean_content_sha256":clean_sha,"clean_manifest_sha256":_json_sha(cm),"fold_manifest_sha256":_json_sha(fm),"fold_counts":{f:{s:len(v) for s,v in x.items()} for f,x in assignments.items()},"test_materialized":False,"model_accessed":False,"training_executed":False}
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):
        old=out/"fold_summary.json"
        if old.is_file():
            v=_load(old)
            if v.get("fold_manifest_sha256")==summary["fold_manifest_sha256"] and v.get("clean_manifest_sha256")==summary["clean_manifest_sha256"]: return v
        raise FileExistsError(f"Output berbeda/parsial: {out}")
    out.mkdir(parents=True,exist_ok=True)
    (out/"clean_manifest.json").write_text(json.dumps(cm,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (out/"fold_manifest.json").write_text(json.dumps(fm,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (out/"fold_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--canonical-root",required=True,type=Path); ap.add_argument("--provenance",required=True,type=Path); ap.add_argument("--output-dir",required=True,type=Path); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--validation-ratio",type=float,default=.10)
    a=ap.parse_args(); prepare_preprocessing_folds(a.canonical_root,a.provenance,a.output_dir,folds=a.folds,seed=a.seed,validation_ratio=a.validation_ratio)
if __name__=="__main__": main()
