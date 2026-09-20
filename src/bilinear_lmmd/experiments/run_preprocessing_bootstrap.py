from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS=("R0","C0","F0","W0"); COMPARISONS=("C0","F0","W0")

def _read_master(path:Path):
    with Path(path).open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    if not rows: raise ValueError("OOF master table kosong.")
    identities=[row["identity"] for row in rows]
    if len(identities)!=len(set(identities)): raise ValueError("OOF master table memiliki identity duplikat.")
    actual=[row["actual"] for row in rows]; predictions={arm:[row[f"{arm}_predicted"] for row in rows] for arm in ARMS}; return identities,actual,predictions

def _confusion(actual,predicted,indices,classes): return np.bincount(actual[indices]*classes+predicted[indices],minlength=classes*classes).reshape(classes,classes)
def _metrics(actual,predicted,indices,classes):
    matrix=_confusion(actual,predicted,indices,classes).astype(np.float64); tp=np.diag(matrix); pred_support=matrix.sum(0); true_support=matrix.sum(1); denom=pred_support+true_support
    f1=np.divide(2.0*tp,denom,out=np.zeros_like(tp),where=denom>0); recall=np.divide(tp,true_support,out=np.zeros_like(tp),where=true_support>0); accuracy=tp.sum()/max(matrix.sum(),1.0)
    return np.asarray([float(f1.mean()),float(recall.mean()),float(accuracy),float(f1.min())],dtype=np.float64)
def _interval(values): return {"mean":float(values.mean()),"std":float(values.std(ddof=1)),"median":float(np.median(values)),"ci95_low":float(np.quantile(values,.025)),"ci95_high":float(np.quantile(values,.975)),"probability_positive":float(np.mean(values>0.0))}

def run_bootstrap(master_table:Path,output:Path,*,iterations:int=10000,random_seed:int=20260919)->dict:
    if iterations<100: raise ValueError("Bootstrap minimal 100 iterasi.")
    identities,actual_names,predicted_names=_read_master(master_table); classes=sorted(set(actual_names)); class_to_index={name:index for index,name in enumerate(classes)}
    actual=np.asarray([class_to_index[x] for x in actual_names],dtype=np.int64); predictions={arm:np.asarray([class_to_index[x] for x in predicted_names[arm]],dtype=np.int64) for arm in ARMS}; indices=np.arange(len(identities),dtype=np.int64); groups=[indices[actual==i] for i in range(len(classes))]
    if any(len(g)==0 for g in groups): raise RuntimeError("Setiap class harus hadir untuk stratified bootstrap.")
    metric_names=("macro_f1","balanced_accuracy","accuracy","worst_class_f1")
    point={arm:{m:float(v) for m,v in zip(metric_names,_metrics(actual,predictions[arm],indices,len(classes)))} for arm in ARMS}; point_delta={arm:{m:point[arm][m]-point["R0"][m] for m in metric_names} for arm in COMPARISONS}
    draws={arm:np.empty((iterations,len(metric_names)),dtype=np.float64) for arm in COMPARISONS}; rng=np.random.default_rng(random_seed)
    for iteration in range(iterations):
        sampled=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups]); raw=_metrics(actual,predictions["R0"],sampled,len(classes))
        for arm in COMPARISONS: draws[arm][iteration]=_metrics(actual,predictions[arm],sampled,len(classes))-raw
        if (iteration+1)%1000==0: print(f"PAIRED STRATIFIED BOOTSTRAP {iteration+1}/{iterations}",flush=True)
    result={"format":"bilinear_lmmd.preprocessing.paired_bootstrap.v1","estimand":"preprocessing arm minus R0 Raw on the same OOF identities","samples":len(identities),"classes":classes,"iterations":iterations,"random_seed":random_seed,"stratification":"resample_with_replacement_within_each_actual_class","scope":"Conditional on the frozen seed-42 trained models; this quantifies OOF sample uncertainty, not full optimization-seed uncertainty.","point_estimate":point,"point_delta_vs_R0":point_delta,"paired_bootstrap_delta_vs_R0":{arm:{m:_interval(draws[arm][:,i]) for i,m in enumerate(metric_names)} for arm in COMPARISONS}}
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result["point_delta_vs_R0"],indent=2),flush=True); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--master-table",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--iterations",type=int,default=10000); p.add_argument("--random-seed",type=int,default=20260919); a=p.parse_args(); run_bootstrap(a.master_table,a.output,iterations=a.iterations,random_seed=a.random_seed)
if __name__=="__main__": main()
