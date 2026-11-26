"""Audited B0/C2 recovery: inference-only internal OOF, never training."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import shutil
import statistics
import tempfile
import platform
from pathlib import Path
from importlib.metadata import version
import torch
import yaml

from bilinear_lmmd.core.reproducibility import canonical_json_sha256, sha256_file, current_git_commit
from bilinear_lmmd.data.preparation.prepare_coffee17 import discover_directory_samples
from bilinear_lmmd.data.preprocessing.study_data import build_preprocessing_evaluation_loader
from bilinear_lmmd.engine.train import classification_metrics
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.experiments.run_frequency_screening_v1 import _evaluate, METRICS, validate_config
from bilinear_lmmd.experiments.run_c2_recovery_v1 import verify_manifest, EXPECTED_PARAMS, EXPECTED_CORE
from bilinear_lmmd.modeling.frequency_screening import build_frequency_screening_model

PROTOCOL="coffee17-b0-c2-internal-oof-v1"
RECOVERY_PIN="12cc8e614c63968cd2df69db42490f6f3da19450"
AUTH_SHA="40029c4143e13e192eeb697772722106039703e4b5f801a41d4e23dcb95685da"
RAW_SHA="500f0343da4ed7a7e0c908d64302b0d989c64d2974297e8f6c7e4ee913de055b"
MARKER_SHA={
1:{"B0":"559f82b232a138f0444d77b586c997c4474f0bb62a2f7d35a1ee4134975eb52b","C2":"233d79b648634de49849de4b8f4f4084585e5437bac20e9398f416b419a9ed33"},
2:{"B0":"6c9e527e43a1a0e0ca4072b1428ae4a21bcd66843ba7b2a79147efb7a280986b","C2":"0043360d874137ba4634402a6ba57d58f1dad09b5d884aef9bc8082907fa88dd"},
3:{"B0":"6f73fcf03a29dce0f04b14e842683a9e157aeb4e403ec52e5c25a02c719e4728","C2":"1c033457f916d2d19f81aa0990625d14d69d575f202b319ba79f62f75d8c40b8"},
4:{"B0":"89368eec95561aeacdccef56f061e3d02a4cf90352853e1bd21a74973231cfac","C2":"79950e6840cba108c8133cbef0a50fdfe8db7d848d372730d4828be45e00090e"},
5:{"B0":"29317cd347b41174ff709e7597f27867257220340220b547232552c869dd11ca","C2":"385f99927ce0bb9895496a8d32325925a6f865e4a00f364cfd757965a294302e"}}
ARMS=("B0","C2")
CLASSES=['Broken','Cut','Dry Cherry','Fade','Floater','Full Black','Full Sour','Fungus Damage','Husk','Immature','Parchment','Partial Black','Partial Sour','Severe Insect Damage','Shell','Slight Insect Damage','Withered']
GROUPS={"sour_black":["Partial Black","Partial Sour","Full Sour"],"shape_withered":["Withered","Immature","Cut"],"insect_damage":["Slight Insect Damage","Severe Insect Damage"]}
PAIRS=[("Withered","Immature"),("Severe Insect Damage","Slight Insect Damage"),("Cut","Slight Insect Damage"),("Partial Sour","Full Sour"),("Slight Insect Damage","Fade"),("Full Black","Partial Black")]

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def read(p): return json.loads(p.read_text(encoding="utf-8"))

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(p)

def checked(p,sha):
    require(p.is_file(),f"Missing file: {p}")
    require(sha256_file(p)==sha,f"SHA mismatch: {p}")

def model_for(cfg,arm,ck):
    cfg=copy.deepcopy(cfg)
    cfg["model"]["pretrained"]=False # weights exclusively from audited checkpoint
    model=build_frequency_screening_model(arm,cfg)
    require(sum(p.numel() for p in model.parameters())==EXPECTED_PARAMS[arm],"Parameter drift")
    model.load_state_dict(ck["model"],strict=True)
    return model.eval()

def audit(recovery,dataset,device):
    checked(recovery/"recovery_authority.json",AUTH_SHA)
    authority=read(recovery/"recovery_authority.json")
    m=recovery/"manifests"
    checked(m/"coffee17_raw_manifest.json",RAW_SHA)
    raw=read(m/"coffee17_raw_manifest.json")
    clean=read(m/"clean_manifest.json")
    folds=read(m/"fold_manifest.json")
    verify_manifest(clean,folds)
    require(canonical_json_sha256(clean)==authority["clean_manifest_sha256"],"Clean manifest drift")
    require(canonical_json_sha256(folds)==authority["fold_manifest_sha256"],"Fold manifest drift")
    raw_rows={r["identity"]:r for r in raw["images"]}
    clean_rows={r["identity"]:r for r in clean["images"]}
    require(len(raw_rows)==979 and len(clean_rows)==965 and len({r["sha256"] for r in clean_rows.values()})==965,"Population drift")
    require(all(raw_rows[k]["sha256"]==r["sha256"] for k,r in clean_rows.items()),"Clean/raw mismatch")
    # Read source bytes for provenance only, no image inference before full gate.
    paths={}
    for cls,files in discover_directory_samples(dataset).items():
        for p in files:
            ident=f"{cls}/{p.name}"
            require(ident not in paths,"Duplicate source identity")
            paths[ident]=p
    require(paths.keys()==raw_rows.keys(),"Raw source identity mismatch")
    for ident,p in paths.items():
        checked(p,raw_rows[ident]["sha256"])
    print("SOURCE BYTES: 979/979 MATCH RECOVERY MANIFEST",flush=True)
    records={}
    required={"best.pt","last.pt","history.json","run_config.yaml","run_contract.json","validation/metrics.json","validation/predictions.csv","validation/confusion_matrix.csv"}
    for f in range(1,6):
        for arm in ARMS:
            d=recovery/"experiments"/f"fold_{f}"/"seed42"/arm
            checked(d/"completed.json",MARKER_SHA[f][arm])
            marker=read(d/"completed.json")
            require(set(marker["files"])==required,"Completion file list mismatch")
            for name,h in marker["files"].items(): checked(d/name,h)
            c=read(d/"run_contract.json")
            cfg=yaml.safe_load((d/"run_config.yaml").read_text())
            validate_config(cfg)
            h=read(d/"history.json")
            selected=max(h,key=lambda x:x["source"]["macro_f1"])
            contract=canonical_json_sha256({k:v for k,v in c.items() if k!="run_contract_sha256"})
            require(c["run_contract_sha256"]==contract==marker["run_contract_sha256"],"Contract mismatch")
            require(c["protocol"]=="coffee17-b0-c2-checkpoint-recovery-v1" and c["git_commit"]==RECOVERY_PIN and c["fold"]==f and c["candidate"]==arm and c["seed"]==42 and c["outer_test_accessed"] is False,"Wrong recovery run")
            require(c["manifest_sha256"]==authority["fold_manifest_sha256"] and c["preflight"]["shared_core_sha256"]==EXPECTED_CORE,"Provenance drift")
            require(canonical_json_sha256(cfg)==c["resolved_config_sha256"] and cfg["evaluation"]["hard_groups"]==GROUPS,"Config mismatch")
            require([x["epoch"] for x in h]==list(range(1,51)),"Incomplete history")
            ck=torch.load(d/"best.pt",map_location="cpu",weights_only=True)
            require(ck["run_contract_sha256"]==contract and ck["candidate"]==arm and ck["epoch"]==selected["epoch"] and ck["best_f1"]==selected["source"]["macro_f1"] and ck["classes"]==CLASSES and ck["config"]==cfg,"Selected checkpoint metadata mismatch")
            model=model_for(cfg,arm,ck).to(device)
            with torch.inference_mode():
                logits=model(torch.zeros(1,3,224,224,device=device)).logits
            require(tuple(logits.shape)==(1,17) and torch.isfinite(logits).all().item(),"Synthetic forward failed")
            del model,ck,logits
            torch.cuda.empty_cache()
            records[(f,arm)]={"dir":d,"config":cfg,"sha256":marker["files"]["best.pt"],"selected_epoch":selected["epoch"]}
            print(f"VERIFIED {f}/{arm} best+last and metadata",flush=True)
    return authority,folds,clean_rows,paths,records

def metric_subset(m): return {k:float(m[k]) for k in METRICS}

def paired(left,right):
    require(left.keys()==right.keys(),"Paired identity mismatch")
    out={"count":len(left),"rescue":0,"damage":0,"both_correct":0,"both_wrong":0}
    for i in left:
        a,b=left[i],right[i]
        require(a["actual"]==b["actual"],"Label mismatch")
        x,y=a["predicted"]==a["actual"],b["predicted"]==b["actual"]
        out["both_correct" if x and y else "both_wrong" if not x and not y else "rescue" if y else "damage"]+=1
    out["net_correct"]=out["rescue"]-out["damage"]
    return out

def pooled_metrics(rows):
    ordered=[rows[k] for k in sorted(rows)]
    y=[CLASSES.index(r["actual"]) for r in ordered]
    pred=[CLASSES.index(r["predicted"]) for r in ordered]
    return classification_metrics(y,pred,CLASSES,GROUPS)

def run(args):
    require(args.authorize_internal_oof,"Explicit internal OOF execution flag required")
    require(torch.cuda.is_available(),"CUDA required")
    commit=current_git_commit(Path(__file__).resolve().parents[3])
    require(commit==args.required_commit,"Scientific pin mismatch")
    require(not args.output.exists(),"Existing evaluation output preserved. Do not rerun over it.")
    device=torch.device("cuda:0")
    env={"torch":torch.__version__,"torchvision":version("torchvision"),"timm":version("timm"),"python":platform.python_version(),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),"cudnn":torch.backends.cudnn.version(),"tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32}
    torch.set_grad_enabled(False)
    authority,folds,clean,paths,records=audit(args.recovery,args.dataset,device)
    print("ALL 10 CHECKPOINTS AND MANIFESTS VERIFIED. INTERNAL OOF ONLY.",flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    evaluation_authority={"protocol":PROTOCOL,"git_commit":commit,"recovery_authority_sha256":AUTH_SHA,"fold_manifest_sha256":authority["fold_manifest_sha256"],"environment":env,"training_performed":False,"independent_confirmation":False,"checkpoint_hashes":{f"fold_{f}/{a}":r["sha256"] for (f,a),r in records.items()},"source_bytes_verified":979,"test_access_authorized":True}
    write(args.output/"evaluation_authority.json",evaluation_authority)
    pooled={a:{} for a in ARMS}
    fold_results=[]
    for f in range(1,6):
        identities=set(folds["assignments"][f"fold_{f}"]["test"])
        with tempfile.TemporaryDirectory(prefix=f"coffee17-oof-fold{f}-") as tmp:
            root=Path(tmp)
            for ident in sorted(identities):
                dest=root/"source"/"test"/ident
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(paths[ident],dest)
                checked(dest,clean[ident]["sha256"])
            result={"fold":f,"n":len(identities),"results":{}}
            fold_rows={}
            for arm in ARMS:
                rec=records[(f,arm)]
                cp=rec["dir"]/"best.pt"
                checked(cp,rec["sha256"])
                ck=torch.load(cp,map_location="cpu",weights_only=True)
                cfg=copy.deepcopy(rec["config"])
                cfg["data"]["root"]=str(root)
                loader,classes=build_preprocessing_evaluation_loader(cfg["data"],split="test",seed=42)
                require(classes==CLASSES,"Test class mapping mismatch")
                model=model_for(cfg,arm,ck).to(device)
                metrics,y,pred,probs,filenames=_evaluate(model,loader,device,CLASSES,GROUPS)
                rows={}
                for path,actual,p,prob in zip(filenames,y,pred,probs):
                    name=Path(path)
                    ident=f"{name.parent.name}/{name.name}"
                    require(ident not in rows and CLASSES[actual]==name.parent.name,"Invalid predicted identity/label")
                    require(all(math.isfinite(x) for x in prob) and abs(sum(prob)-1)<1e-4,"Invalid probability")
                    rows[ident]={"identity":ident,"actual":CLASSES[actual],"predicted":CLASSES[p],"probabilities":prob,"fold":f}
                require(rows.keys()==identities and not (rows.keys() & pooled[arm].keys()),"Incorrect or duplicate test coverage")
                pooled[arm].update(rows)
                fold_rows[arm]=rows
                dest=args.output/"experiments"/f"fold_{f}"/arm
                # Save canonical identity paths, not deleted temporary paths.
                ids=[f"{Path(p).parent.name}/{Path(p).name}" for p in filenames]
                _write_evaluation(dest,metrics=metrics,labels=y,predictions=pred,probabilities=probs,paths=ids,classes=CLASSES,checkpoint=cp,split="test")
                result["results"][arm]={"metrics":metric_subset(metrics),"checkpoint_sha256":rec["sha256"],"selected_epoch":rec["selected_epoch"]}
                del model,ck,loader
                torch.cuda.empty_cache()
            result["delta"]={k:result["results"]["C2"]["metrics"][k]-result["results"]["B0"]["metrics"][k] for k in METRICS}
            result["paired_outcomes"]=paired(fold_rows["B0"],fold_rows["C2"])
            write(args.output/"experiments"/f"fold_{f}"/"fold_result.json",result)
            fold_results.append(result)
            print("FOLD",f,result["delta"],flush=True)
    metrics={}
    for arm,rows in pooled.items():
        require(rows.keys()==clean.keys(),"Incomplete OOF coverage")
        ordered=[rows[k] for k in sorted(rows)]
        y=[CLASSES.index(r["actual"]) for r in ordered]
        pred=[CLASSES.index(r["predicted"]) for r in ordered]
        metrics[arm]=pooled_metrics(rows)
        _write_evaluation(args.output/"pooled"/arm,metrics=metrics[arm],labels=y,predictions=pred,probabilities=[r["probabilities"] for r in ordered],paths=[r["identity"] for r in ordered],classes=CLASSES,checkpoint=Path("fold_specific_checkpoints"),split="internal_oof")
        write(args.output/"pooled"/arm/"identity_fold_map.json",{r["identity"]:r["fold"] for r in ordered})
    delta={k:float(metrics["C2"][k]-metrics["B0"][k]) for k in METRICS}
    positive=sum(r["delta"]["macro_f1"]>0 for r in fold_results)
    gate=delta["macro_f1"]>0 and positive>=3 and delta["hard_class_f1"]>=0 and delta["worst_class_f1"]>=0
    pairs={}
    for a,b in PAIRS:
        pairs[f"{a} <-> {b}"]={arm:{"a_to_b":sum(r["actual"]==a and r["predicted"]==b for r in rows.values()),"b_to_a":sum(r["actual"]==b and r["predicted"]==a for r in rows.values())} for arm,rows in pooled.items()}
    summary={"protocol":PROTOCOL,"training_performed":False,"outer_test_accessed":True,"independent_confirmation":False,"n_unique_per_arm":len(clean),"fold_results":fold_results,"pooled_metrics":metrics,"pooled_delta":delta,"positive_macro_folds":positive,"paired_outcomes":paired(pooled["B0"],pooled["C2"]),"hard_pair_confusion":pairs,"mean_fold_metrics":{a:{k:{"mean":statistics.mean(r["results"][a]["metrics"][k] for r in fold_results),"std":statistics.stdev(r["results"][a]["metrics"][k] for r in fold_results)} for k in METRICS} for a in ARMS},"decision":"INTERNAL_SUPPORT" if gate else "NO_INTERNAL_SUPPORT"}
    write(args.output/"analysis/internal_oof_v1.json",summary)
    print("OOF COMPLETE:",summary["decision"],"DELTA:",delta,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("recovery","dataset","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--required-commit",required=True)
    p.add_argument("--authorize-internal-oof",action="store_true")
    run(p.parse_args())

if __name__=="__main__": main()
