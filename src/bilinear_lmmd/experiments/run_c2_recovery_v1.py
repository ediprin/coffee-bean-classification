"""Authorized B0/C2 recovery only; no outer evaluation. Training loop copied from
focused scientific pin 6653a50e296f7b38591100b395f1a09dbaf68c56.
Only resume loading uses CPU to preserve CPU RNG tensors.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256, capture_rng_state, current_git_commit,
    restore_rng_state, seed_everything, sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import build_preprocessing_study_loaders, set_study_epoch
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.experiments.run_frequency_screening_v1 import (
    METRICS, _evaluate, _paired_outcomes, _prediction_rows, preflight_candidate, validate_config,
)
from bilinear_lmmd.modeling.frequency_screening import build_frequency_screening_model
from bilinear_lmmd.data.preparation.materialize_preprocessing_development import materialize_preprocessing_development

PROTOCOL = "coffee17-b0-c2-checkpoint-recovery-v1"
ARMS = ("B0", "C2")
EXPECTED_CORE = "e288e9d23781017c7afd8e97fc5a522eaef84ada4751ca8034d133017a70b995"
EXPECTED_VAL = {
    1: "37c2e04d069200cd8c1768016d1fa7e9a6fb19511988f2f32397ad8f3aaf00cd",
    2: "853da93b314ce3d2e992c1c67718aea27780cfcf36f5e045e1533b34bb7788b4",
    3: "783cb11115dad436c4f0ea126df8bf39d1c98638d668fb0dc5b9e09ffb668d0f",
    4: "b1c181a9620352555a1c4d962a4bc2cdfcd02edb9feb80d7e04c402e5a80b03e",
    5: "4a7340263bc5416bcde64de6b20312af55a6fe74b80a6db06890a07ac8429a97",
}
EXPECTED_PARAMS = {"B0": 2988289, "C2": 2993089}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise RuntimeError(f"Existing artifact differs: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_manifest(clean, folds):
    if clean["raw_count"] != 979 or clean["clean_count"] != 965 or len(clean["clean_class_counts"]) != 17:
        raise RuntimeError("Unexpected population")
    if folds["decision"] != "PASS" or folds["seed"] != 42 or folds["folds"] != 5 or folds["validation_ratio"] != .1:
        raise RuntimeError("Unexpected split contract")
    if folds["clean_content_sha256"] != clean["clean_content_sha256"] or folds["clean_manifest_sha256"] != canonical_json_sha256(clean):
        raise RuntimeError("Manifest fingerprint mismatch")
    ids = {x["identity"] for x in clean["images"]}
    seen = set()
    for fold, expected in EXPECTED_VAL.items():
        splits = folds["assignments"][f"fold_{fold}"]
        a,b,c = (set(splits[k]) for k in ("train","val","test"))
        if a&b or a&c or b&c or a|b|c != ids or c&seen:
            raise RuntimeError(f"Invalid split fold {fold}")
        seen |= c
        actual = hashlib.sha256("".join(f"{i}\t{i.rsplit('/',1)[0]}\n" for i in sorted(b)).encode()).hexdigest()
        if len(b)!=97 or actual!=expected:
            raise RuntimeError(f"Historical validation identity mismatch fold {fold}")
    if seen!=ids:
        raise RuntimeError("Incomplete OOF coverage")


def reconcile_resume(run_dir, contract_sha):
    """Recover interrupted last->best save only from same-run last.pt."""
    last, best = run_dir/"last.pt", run_dir/"best.pt"
    if not last.exists():
        if best.exists() or (run_dir/"history.json").exists() or (run_dir/"validation").exists():
            raise RuntimeError(f"Partial run missing last.pt: {run_dir}")
        return 0
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck["run_contract_sha256"] != contract_sha:
        raise RuntimeError("Last checkpoint contract mismatch")
    hist = ck["history"]
    epoch = int(ck["epoch"])
    if epoch != len(hist) or not 1<=epoch<=50 or [x["epoch"] for x in hist] != list(range(1,epoch+1)):
        raise RuntimeError("Checkpoint history mismatch")
    selected = max(hist,key=lambda x: x["source"]["macro_f1"])
    if float(ck["best_f1"]) != float(selected["source"]["macro_f1"]):
        raise RuntimeError("Best score mismatch")
    old = torch.load(best,map_location="cpu",weights_only=False) if best.exists() else None
    if old is not None and old["run_contract_sha256"] != contract_sha:
        raise RuntimeError("Best checkpoint contract mismatch")
    if old is None or old["epoch"] != selected["epoch"]:
        if selected["epoch"] != epoch or (old is not None and old["epoch"] > epoch):
            raise RuntimeError("Cannot reconstruct missing selected best safely")
        # last was saved immediately before best in the historical trainer.
        # Preserve an earlier best before repairing the interrupted transaction.
        if best.exists():
            import shutil
            backup = run_dir / f"best.interrupted.{sha256_file(best)}.pt"
            if not backup.exists():
                shutil.copy2(best,backup)
        atomic_torch_save({k:ck[k] for k in ("model","classes","config","candidate","epoch","best_f1","run_contract_sha256")},best)
        old = torch.load(best,map_location="cpu",weights_only=False)
    if float(old["best_f1"]) != float(ck["best_f1"]):
        raise RuntimeError("Best checkpoint score mismatch")
    if epoch == 50:
        write(run_dir/"history.json",hist,immutable=True) if not (run_dir/"history.json").exists() else None
        if read(run_dir/"history.json") != hist:
            write(run_dir/"history.json",hist)
    return epoch


def train_candidate(
    cfg: dict,
    *,
    candidate: str,
    run_dir: Path,
    run_contract_sha256: str,
    resume: bool,
) -> dict:
    validate_config(cfg)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    model = build_frequency_screening_model(candidate, cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
    )
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError(f"Checkpoint {candidate} berasal dari kontrak berbeda.")
        required = {"model", "optimizer", "scheduler", "history", "best_f1", "rng_state"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint {candidate} tidak lengkap: {missing}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME {candidate}: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        model.train()
        running_loss = 0.0
        batches = 0
        progress = tqdm(loaders.train, desc=f"{candidate} epoch {epoch + 1}/{epochs}")
        for raw_images, labels in progress:
            raw_images = raw_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            normalized = imagenet_normalize(raw_images)
            optimizer.zero_grad(set_to_none=True)
            output = model(normalized, labels=labels)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics, _, _, _, _ = _evaluate(
            model, loaders.val, device, loaders.classes, hard_groups
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(1, batches),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps({
            "candidate": candidate,
            "epoch": epoch + 1,
            "macro_f1": metrics["macro_f1"],
            "hard_class_f1": metrics["hard_class_f1"],
            "worst_class_f1": metrics["worst_class_f1"],
        }), flush=True)

        is_best = float(metrics["macro_f1"]) > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "candidate": candidate,
            "epoch": epoch + 1,
            "history": history,
            "best_f1": best_f1,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
        }
        atomic_torch_save(checkpoint, last_path)
        if is_best:
            atomic_torch_save({
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "candidate": candidate,
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "run_contract_sha256": run_contract_sha256,
            }, best_path)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    eval_model = build_frequency_screening_model(candidate, cfg)
    eval_model.load_state_dict(best["model"])
    eval_model = eval_model.to(device)
    metrics, labels, predictions, probabilities, paths = _evaluate(
        eval_model, loaders.val, device, loaders.classes, hard_groups
    )
    _write_evaluation(
        run_dir / "validation",
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        paths=paths,
        classes=loaders.classes,
        checkpoint=best_path,
        split="val",
    )
    return metrics



def run(args):
    if not args.authorize_recovery:
        raise RuntimeError("Explicit recovery authorization required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root=Path(__file__).resolve().parents[3]
    commit=current_git_commit(root)
    if commit!=args.required_commit:
        raise RuntimeError("Git pin mismatch")
    cfg0=load_config(args.config)
    validate_config(cfg0)
    if not cfg0["model"]["pretrained"]:
        raise RuntimeError("Historical pretrained initialization required")
    clean=read(args.manifests/"clean_manifest.json")
    folds=read(args.manifests/"fold_manifest.json")
    verify_manifest(clean,folds)
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0")
    torch.use_deterministic_algorithms(False)
    env={"torch":torch.__version__,"torchvision":importlib.metadata.version("torchvision"),
         "timm":importlib.metadata.version("timm"),"numpy":importlib.metadata.version("numpy"),
         "python":platform.python_version(),"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
         "gpu":torch.cuda.get_device_name(device),"gpu_count":torch.cuda.device_count(),
         "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,
         "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32}
    preflight={}
    for arm in ARMS:
        preflight[arm]=preflight_candidate(cfg0,arm)
        if preflight[arm]["shared_core_sha256"]!=EXPECTED_CORE:
            raise RuntimeError(f"{arm}: historical pretrained core differs")
        seed_everything(42)
        model=build_frequency_screening_model(arm,cfg0).to(device)
        if sum(p.numel() for p in model.parameters())!=EXPECTED_PARAMS[arm]:
            raise RuntimeError("Parameter count drift")
        model.train()
        x=torch.randn(2,3,224,224,device=device)
        loss=nn.CrossEntropyLoss()(model(x).logits,torch.tensor([0,1],device=device))
        loss.backward()
        if not torch.isfinite(loss).item() or any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise RuntimeError("Non-finite GPU preflight")
        del model,x,loss
        torch.cuda.empty_cache()
    write(args.output/"recovery_authority.json",{
        "protocol":PROTOCOL,"git_commit":commit,"environment":env,
        "clean_manifest_sha256":canonical_json_sha256(clean),"fold_manifest_sha256":canonical_json_sha256(folds),
        "historical_validation_hashes":{str(k):v for k,v in EXPECTED_VAL.items()},
        "historical_core_verified":True,"outer_test_accessed":False,
        "historical_content_hash_available":False,"weights_are_recovered_not_original":True,
        "config_sha256":sha256_file(args.config),
    },immutable=True)
    # Every fold's identities were checked before any optimization.
    for fold in range(1,6):
        dev=args.scratch/f"fold_{fold}"
        materialize_preprocessing_development(args.canonical,args.manifests/"clean_manifest.json",args.manifests/"fold_manifest.json",dev,fold=fold)
        count, val_sha=validation_identity_label_sha256(dev)
        if (count,val_sha)!=(97,EXPECTED_VAL[fold]):
            raise RuntimeError("Actual validation tree mismatch")
        cfg=copy.deepcopy(cfg0)
        cfg["device"]="cuda:0"
        cfg["data"]["root"]=str(dev.resolve())
        folder=args.output/"experiments"/f"fold_{fold}"/"seed42"
        results={}
        for arm in ARMS:
            d=folder/arm
            d.mkdir(parents=True,exist_ok=True)
            contract={"protocol":PROTOCOL,"fold":fold,"seed":42,"candidate":arm,"anchor":"B0",
                      "git_commit":commit,"validation_count":count,"validation_identity_label_sha256":val_sha,
                      "preflight":preflight[arm],"parameter_count":EXPECTED_PARAMS[arm],
                      "manifest_sha256":canonical_json_sha256(folds),"environment":env,
                      "outer_test_accessed":False,"resolved_config_sha256":canonical_json_sha256(cfg)}
            sha=canonical_json_sha256(contract)
            contract["run_contract_sha256"]=sha
            write(d/"run_contract.json",contract,immutable=True)
            cp=d/"run_config.yaml"
            if cp.exists() and yaml.safe_load(cp.read_text())!=cfg:
                raise RuntimeError("Run config differs")
            if not cp.exists():
                cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
            epoch=reconcile_resume(d,sha)
            done=d/"completed.json"
            if done.exists():
                marker=read(done)
                if epoch!=50 or marker["run_contract_sha256"]!=sha or any(sha256_file(d/name)!=digest for name,digest in marker["files"].items()):
                    raise RuntimeError("Completed artifact drift; refuse retraining")
                print(f"SKIP completed {fold}/{arm}",flush=True)
            else:
                train_candidate(cfg,candidate=arm,run_dir=d,run_contract_sha256=sha,resume=epoch>0)
                if reconcile_resume(d,sha)!=50:
                    raise RuntimeError("Incomplete recovery")
                names=["best.pt","last.pt","history.json","run_config.yaml","run_contract.json","validation/metrics.json","validation/predictions.csv","validation/confusion_matrix.csv"]
                write(done,{"run_contract_sha256":sha,"files":{name:sha256_file(d/name) for name in names}},immutable=True)
            metrics=read(d/"validation/metrics.json")
            results[arm]={"metrics":{k:float(metrics[k]) for k in METRICS},"best_checkpoint_sha256":sha256_file(d/"best.pt"),"parameter_count":EXPECTED_PARAMS[arm]}
        results["C2"]["delta_vs_anchor"]={k:results["C2"]["metrics"][k]-results["B0"]["metrics"][k] for k in METRICS}
        results["C2"]["paired_outcomes"]=_paired_outcomes(_prediction_rows(folder/"B0/validation/predictions.csv"),_prediction_rows(folder/"C2/validation/predictions.csv"))
        write(folder/"fold_result.json",{"protocol":PROTOCOL,"fold":fold,"results":results,"outer_test_accessed":False},immutable=True)
    rows=[read(args.output/"experiments"/f"fold_{f}"/"seed42"/"fold_result.json") for f in range(1,6)]
    summary={"protocol":PROTOCOL,"weights_are_recovered_not_original":True,"outer_test_accessed":False,
             "training_runs":10,"folds":rows,"aggregate":{}}
    for arm in ARMS:
        summary["aggregate"][arm]={k:{"mean":statistics.mean([r["results"][arm]["metrics"][k] for r in rows]),
                                    "std":statistics.stdev([r["results"][arm]["metrics"][k] for r in rows])} for k in METRICS}
    summary["delta_vs_anchor"]={k:{"mean":statistics.mean([r["results"]["C2"]["delta_vs_anchor"][k] for r in rows]),
                                    "positive_folds":sum(r["results"]["C2"]["delta_vs_anchor"][k]>0 for r in rows)} for k in METRICS}
    write(args.output/"analysis/recovery_v1.json",summary,immutable=True)
    print("RECOVERY COMPLETE; OUTER TEST LOCKED",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("config","canonical","manifests","output","scratch"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--required-commit",required=True)
    p.add_argument("--authorize-recovery",action="store_true")
    a=p.parse_args()
    with exclusive_training_lock(a.output,lock_name="RECOVERY_V1.training.lock",stale_seconds=900):
        run(a)

if __name__=="__main__":
    main()
