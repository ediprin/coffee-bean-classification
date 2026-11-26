"""SAM-AdamW reference, B0 architecture, development only. No baseline retraining."""
from __future__ import annotations
import argparse, copy, json, math, platform, statistics, shutil, time
from pathlib import Path
from importlib.metadata import version
import torch
import yaml
from torch import nn
from tqdm import tqdm

from bilinear_lmmd.core.reproducibility import canonical_json_sha256, capture_rng_state, restore_rng_state, seed_everything, sha256_file, current_git_commit
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import build_preprocessing_study_loaders, set_study_epoch
from bilinear_lmmd.data.preparation.prepare_coffee17 import discover_directory_samples
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.experiments.run_frequency_screening_v1 import METRICS, _evaluate, _paired_outcomes, _prediction_rows, preflight_candidate, validate_config
from bilinear_lmmd.experiments.run_c2_recovery_v1 import verify_manifest, reconcile_resume, EXPECTED_CORE, EXPECTED_VAL, read, write
from bilinear_lmmd.experiments.run_c2_internal_oof_v1 import AUTH_SHA, RAW_SHA, MARKER_SHA, RECOVERY_PIN, GROUPS
from bilinear_lmmd.modeling.frequency_screening import build_frequency_screening_model

PROTOCOL = "coffee17-b0-sam-reference-v1"
ARM = "B0_SAM"
RHO = .05
PARAMS = 2988289

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def checked(path, digest):
    require(path.is_file() and sha256_file(path)==digest, f"Missing or changed audited artifact: {path}")

def rng():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore(state):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state_all(state[1])

def grad_norm(params):
    grads=[p.grad for p in params if p.grad is not None]
    require(bool(grads), "No gradients")
    n=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads]))
    require(torch.isfinite(n).item(), "Non-finite gradients")
    return n

def sam_step(model, optimizer, closure, *, rho=RHO, debug=False):
    """Two gradients, one AdamW step. Replay dropout RNG; persist BN buffers once."""
    require(math.isfinite(rho) and rho>=0, "Invalid rho")
    require(model.training, "SAM requires training mode")
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    before_rng=rng()
    loss=closure()
    require(torch.isfinite(loss).item(), "Non-finite base loss")
    loss.backward()
    norm=grad_norm(params)
    after_rng=rng()
    # Save the state AFTER the ordinary pass. The second pass uses training-mode
    # batch statistics, but its running stats/counters must not persist.
    bn=[(b,b.detach().clone()) for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)
        for b in (m.running_mean,m.running_var,m.num_batches_tracked) if b is not None]
    saved=[]
    try:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    saved.append((p,p.detach().clone()))
                    p.add_(p.grad * (rho / norm.clamp_min(1e-12)))
            if debug:
                radius=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm((p-old).float()) for p,old in saved]))
                expected=rho if float(norm)>1e-12 else rho*float(norm)/1e-12
                require(math.isclose(float(radius),expected,rel_tol=2e-4,abs_tol=1e-6), "Perturbation norm drift")
        optimizer.zero_grad(set_to_none=True)
        restore(before_rng)
        loss2=closure()
        require(torch.isfinite(loss2).item(), "Non-finite perturbed loss")
        loss2.backward()
        grad_norm(params)
    finally:
        with torch.no_grad():
            for p,old in saved: p.copy_(old)
            for b,old in bn: b.copy_(old)
        restore(after_rng)
    if debug:
        require(all(torch.equal(p,old) for p,old in saved),"Weights not restored before optimizer")
    optimizer.step()
    return float(loss.detach()),float(loss2.detach())

def numerical_tests(device):
    # rho=0 must match an ordinary AdamW step, including dropout and BN state.
    seed_everything(42)
    a=nn.Sequential(nn.Linear(4,8),nn.BatchNorm1d(8),nn.ReLU(),nn.Dropout(.2),nn.Linear(8,3)).to(device)
    b=copy.deepcopy(a)
    x=torch.randn(16,4,device=device); y=torch.arange(16,device=device)%3
    opt_a=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=.0001)
    opt_b=torch.optim.AdamW(b.parameters(),lr=.0003,weight_decay=.0001)
    state=rng()
    opt_a.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(a(x),y).backward(); opt_a.step()
    final_rng=rng()
    restore(state)
    sam_step(b,opt_b,lambda: nn.functional.cross_entropy(b(x),y),rho=0.,debug=True)
    require(all(torch.allclose(v,b.state_dict()[k],atol=1e-7,rtol=1e-6) for k,v in a.state_dict().items()),"rho=0 not equivalent to AdamW")
    require(torch.equal(torch.get_rng_state(),final_rng[0]),"CPU RNG drift")
    if final_rng[1] is not None:
        require(all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),final_rng[1])),"CUDA RNG drift")
    # Nonzero-radius analytical quadratic step with SGD (tests gradient at perturbed weights).
    p=nn.Linear(1,1,bias=False).to(device)
    with torch.no_grad(): p.weight.fill_(2.)
    o=torch.optim.SGD(p.parameters(),lr=.1)
    sam_step(p,o,lambda: .5*p.weight.square().sum(),rho=.05,debug=True)
    require(torch.allclose(p.weight,torch.full_like(p.weight,1.795),atol=1e-6),"Incorrect nonzero-radius update")
    # Nonzero rho and BN single update.
    n=int(b[1].num_batches_tracked)
    sam_step(b,opt_b,lambda: nn.functional.cross_entropy(b(x),y),rho=.05,debug=True)
    require(int(b[1].num_batches_tracked)==n+1,"BN counted twice")
    # Restoration on failing second pass.
    snapshot={k:v.clone() for k,v in p.state_dict().items()}
    count=0
    def fail_second():
        nonlocal count
        count+=1
        if count==2: raise ValueError("intentional")
        return .5*p.weight.square().sum()
    try:
        sam_step(p,o,fail_second,rho=.05)
    except ValueError as exc:
        require(str(exc)=="intentional","Wrong exception")
    else: raise RuntimeError("Exception test did not raise")
    require(all(torch.equal(v,p.state_dict()[k]) for k,v in snapshot.items()),"Exception restoration failed")
    print("SAM NUMERICAL TESTS PASS:",device,flush=True)

def environment(device):
    return {"torch":torch.__version__,"torchvision":version("torchvision"),"timm":version("timm"),"numpy":version("numpy"),
         "python":platform.python_version(),"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
         "gpu":torch.cuda.get_device_name(device),"gpu_count":torch.cuda.device_count(),
         "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,
         "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32}

def baseline_audit(recovery,dataset):
    checked(recovery/"recovery_authority.json",AUTH_SHA)
    authority=read(recovery/"recovery_authority.json")
    m=recovery/"manifests"
    checked(m/"coffee17_raw_manifest.json",RAW_SHA)
    clean,folds=read(m/"clean_manifest.json"),read(m/"fold_manifest.json")
    verify_manifest(clean,folds)
    require(canonical_json_sha256(clean)==authority["clean_manifest_sha256"],"Clean manifest drift")
    require(canonical_json_sha256(folds)==authority["fold_manifest_sha256"],"Fold manifest drift")
    raw={r["identity"]:r for r in read(m/"coffee17_raw_manifest.json")["images"]}
    clean_rows={r["identity"]:r for r in clean["images"]}
    require(len(raw)==979 and len(clean_rows)==965 and all(raw[k]["sha256"]==v["sha256"] for k,v in clean_rows.items()),"Source manifest drift")
    paths={}
    for cls,files in discover_directory_samples(dataset).items():
        for p in files:
            ident=f"{cls}/{p.name}"
            require(ident not in paths,"Duplicate identity")
            paths[ident]=p
    require(paths.keys()==raw.keys(),"Raw dataset mismatch")
    for ident,p in paths.items(): checked(p,raw[ident]["sha256"])
    records={}
    for f in range(1,6):
        d=recovery/"experiments"/f"fold_{f}"/"seed42"/"B0"
        checked(d/"completed.json",MARKER_SHA[f]["B0"])
        marker=read(d/"completed.json")
        for name,digest in marker["files"].items():
            if name not in ("best.pt","last.pt"): checked(d/name,digest)
        c=read(d/"run_contract.json")
        cfg=yaml.safe_load((d/"run_config.yaml").read_text())
        validate_config(cfg)
        require(canonical_json_sha256({k:v for k,v in c.items() if k!="run_contract_sha256"})==marker["run_contract_sha256"]==c["run_contract_sha256"],"B0 contract mismatch")
        require(c["git_commit"]==RECOVERY_PIN and c["candidate"]=="B0" and c["fold"]==f and c["seed"]==42 and c["outer_test_accessed"] is False,"Wrong baseline")
        require(c["manifest_sha256"]==authority["fold_manifest_sha256"] and c["preflight"]["shared_core_sha256"]==EXPECTED_CORE and c["environment"]==authority["environment"],"Baseline provenance mismatch")
        require(canonical_json_sha256(cfg)==c["resolved_config_sha256"] and cfg["evaluation"]["hard_groups"]==GROUPS and cfg["model"]["pretrained"],"Baseline config mismatch")
        hist=read(d/"history.json")
        require([x["epoch"] for x in hist]==list(range(1,51)),"Baseline history incomplete")
        metrics=read(d/"validation/metrics.json")
        rows=_prediction_rows(d/"validation/predictions.csv")
        ids={f"{Path(x['path']).parent.name}/{Path(x['path']).name}" for x in rows}
        require(len(rows)==len(ids)==97 and ids==set(folds["assignments"][f"fold_{f}"]["val"]),"Baseline validation mismatch")
        records[f]=(d,cfg,metrics,rows)
    print("AUDITED BASELINE: 5/5 B0 MARKERS AND 979 SOURCE BYTE HASHES MATCH; NO BASELINE TRAINING",flush=True)
    return authority,clean,folds,clean_rows,paths,records

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
    model = build_frequency_screening_model("B0", cfg).to(device)

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
        running_perturbed_loss = 0.0
        epoch_start = time.perf_counter()
        batches = 0
        progress = tqdm(loaders.train, desc=f"{candidate} epoch {epoch + 1}/{epochs}")
        for raw_images, labels in progress:
            raw_images = raw_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            normalized = imagenet_normalize(raw_images)
            loss, perturbed_loss = sam_step(model, optimizer, lambda: loss_fn(model(normalized, labels=labels).logits, labels), rho=RHO)
            running_loss += loss
            running_perturbed_loss += perturbed_loss
            batches += 1
            progress.set_postfix(loss=f"{loss:.4f}", perturbed=f"{perturbed_loss:.4f}")

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_start
        scheduler.step()
        metrics, _, _, _, _ = _evaluate(
            model, loaders.val, device, loaders.classes, hard_groups
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(1, batches),
            "perturbed_loss": running_perturbed_loss / max(1, batches),
            "training_seconds": train_seconds,
            "optimizer_steps": batches,
            "gradient_evaluations": 2*batches,
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
    eval_model = build_frequency_screening_model("B0", cfg)
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
    require(args.authorize_sam_reference,"Explicit authorization flag required")
    require(torch.cuda.is_available(),"CUDA required")
    commit=current_git_commit(Path(__file__).resolve().parents[3])
    require(commit==args.required_commit,"Scientific pin mismatch")
    device=torch.device("cuda:0")
    torch.use_deterministic_algorithms(False)
    authority,clean,folds,clean_rows,paths,records=baseline_audit(args.recovery,args.dataset)
    env=environment(device)
    historical=authority["environment"]
    differences={k:{"historical":historical.get(k),"current":v} for k,v in env.items() if historical.get(k)!=v}
    # Device count and Python version can differ; log them. Refuse core numerical-stack drift.
    critical={"torch","torchvision","timm","cuda","cudnn","gpu","cudnn_benchmark","cudnn_deterministic","tf32_matmul","tf32_cudnn"}
    require(not (set(differences)&critical), f"Historical numerical stack differs: {differences}")
    numerical_tests(torch.device("cpu"))
    numerical_tests(device)
    cfg0=records[1][1]
    preflight=preflight_candidate(cfg0,"B0")
    require(preflight["shared_core_sha256"]==EXPECTED_CORE,"Pretrained core mismatch")
    seed_everything(42)
    model=build_frequency_screening_model("B0",cfg0).to(device).train()
    require(sum(p.numel() for p in model.parameters())==PARAMS,"Parameter count drift")
    x=torch.randn(2,3,224,224,device=device); y=torch.tensor([0,1],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    sam_step(model,opt,lambda: nn.functional.cross_entropy(model(x,labels=y).logits,y,label_smoothing=.1),debug=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),"GPU preflight nonfinite")
    del model,opt,x,y
    torch.cuda.empty_cache()
    print("REAL B0 GPU SAM FORWARD/BACKWARD PREFLIGHT PASS",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/"sam_authority.json",{"protocol":PROTOCOL,"git_commit":commit,"rho":RHO,
       "baseline_authority_sha256":AUTH_SHA,"baseline_marker_sha256":{str(f):MARKER_SHA[f]["B0"] for f in range(1,6)},
       "clean_manifest_sha256":authority["clean_manifest_sha256"],"fold_manifest_sha256":authority["fold_manifest_sha256"],
       "environment":env,"environment_differences":differences,"preflight":preflight,
       "outer_test_accessed":False,"baseline_retrained":False,"parameter_count":PARAMS,
       "dropout_rng":"replay_first_pass","bn_buffers":"restore_after_first_pass","numerical_preflight_passed":True},immutable=True)
    for name in ("clean_manifest.json","fold_manifest.json","coffee17_raw_manifest.json"):
        src=args.recovery/"manifests"/name
        dest=args.output/"manifests"/name
        dest.parent.mkdir(exist_ok=True)
        if dest.exists(): checked(dest,sha256_file(src))
        else: shutil.copy2(src,dest)
    for f in range(1,6):
        baseline,cfg,base_metrics,base_rows=records[f]
        cfg=copy.deepcopy(cfg)
        dev=args.scratch/f"fold_{f}"
        require(not (dev/"source/test").exists(),"Test directory forbidden")
        for split in ("train","val"):
            expected=set(folds["assignments"][f"fold_{f}"][split])
            root=dev/"source"/split
            for ident in sorted(expected):
                dest=root/ident; dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists(): shutil.copy2(paths[ident],dest)
                checked(dest,clean_rows[ident]["sha256"])
            actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
            require(actual==expected,"Unexpected materialized files")
        require(validation_identity_label_sha256(dev)==(97,EXPECTED_VAL[f]),"Validation identity mismatch")
        cfg["device"]="cuda:0"; cfg["data"]["root"]=str(dev.resolve())
        cfg["sam_reference"]={"rho":RHO,"base_optimizer":"AdamW","replay_dropout_rng":True,"bn_running_buffers_once":True}
        folder=args.output/"experiments"/f"fold_{f}"/"seed42"
        d=folder/ARM
        d.mkdir(parents=True,exist_ok=True)
        contract={"protocol":PROTOCOL,"fold":f,"seed":42,"candidate":ARM,"model_builder":"B0","git_commit":commit,
                  "rho":RHO,"baseline_marker_sha256":MARKER_SHA[f]["B0"],"preflight":preflight,"parameter_count":PARAMS,
                  "validation_count":97,"validation_identity_label_sha256":EXPECTED_VAL[f],
                  "manifest_sha256":authority["fold_manifest_sha256"],"environment":env,"outer_test_accessed":False,
                  "resolved_config_sha256":canonical_json_sha256(cfg)}
        sha=canonical_json_sha256(contract); contract["run_contract_sha256"]=sha
        write(d/"run_contract.json",contract,immutable=True)
        cp=d/"run_config.yaml"
        if cp.exists(): require(yaml.safe_load(cp.read_text())==cfg,"Config drift")
        else: cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
        epoch=reconcile_resume(d,sha)
        done=d/"completed.json"
        names=["best.pt","last.pt","history.json","run_config.yaml","run_contract.json","validation/metrics.json","validation/predictions.csv","validation/confusion_matrix.csv"]
        if done.exists():
            marker=read(done)
            require(epoch==50 and marker["run_contract_sha256"]==sha and set(marker["files"])==set(names),"Completion mismatch")
            for name,digest in marker["files"].items(): checked(d/name,digest)
            print(f"SKIP COMPLETE FOLD {f}",flush=True)
        else:
            train_candidate(cfg,candidate=ARM,run_dir=d,run_contract_sha256=sha,resume=epoch>0)
            require(reconcile_resume(d,sha)==50,"Incomplete SAM training")
            write(done,{"run_contract_sha256":sha,"files":{name:sha256_file(d/name) for name in names}},immutable=True)
        metrics=read(d/"validation/metrics.json")
        rows=_prediction_rows(d/"validation/predictions.csv")
        def canonical_rows(items):
            return sorted([dict(x,path=f"{Path(x['path']).parent.name}/{Path(x['path']).name}") for x in items],key=lambda x:x["path"])
        old,new=canonical_rows(base_rows),canonical_rows(rows)
        require(len(old)==len(new)==97 and len({x["path"] for x in new})==97,"Duplicate or incomplete predictions")
        paired=_paired_outcomes(old,new)
        hist=read(d/"history.json")
        result={"fold":f,"B0":{k:float(base_metrics[k]) for k in METRICS},"B0_SAM":{k:float(metrics[k]) for k in METRICS},
                "delta":{k:float(metrics[k])-float(base_metrics[k]) for k in METRICS},"paired_outcomes":paired,
                "best_sha256":sha256_file(d/"best.pt"),"training_seconds":sum(h["training_seconds"] for h in hist),
                "optimizer_steps":sum(h["optimizer_steps"] for h in hist),"gradient_evaluations":sum(h["gradient_evaluations"] for h in hist)}
        write(folder/"fold_result.json",result,immutable=True)
        print("FOLD DELTA",f,result["delta"],flush=True)
    results=[read(args.output/"experiments"/f"fold_{f}"/"seed42"/"fold_result.json") for f in range(1,6)]
    means={a:{k:statistics.mean(r[a][k] for r in results) for k in METRICS} for a in ("B0","B0_SAM")}
    delta={k:statistics.mean(r["delta"][k] for r in results) for k in METRICS}
    positive=sum(r["delta"]["macro_f1"]>0 for r in results)
    passed=delta["macro_f1"]>0 and positive>=3 and delta["hard_class_f1"]>=0 and delta["worst_class_f1"]>=0
    report={"protocol":PROTOCOL,"training_runs":5,"baseline_retrained":False,"outer_test_accessed":False,
            "independent_confirmation":False,"means":means,"delta":delta,"positive_macro_folds":positive,"folds":results,
            "decision":"EXPLORATORY_SAM_SIGNAL" if passed else "FAIL_STOP"}
    write(args.output/"analysis/sam_reference_v1.json",report,immutable=True)
    print("SAM REFERENCE COMPLETE",report["decision"],json.dumps(delta),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("recovery","dataset","output","scratch"): p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--required-commit",required=True)
    p.add_argument("--authorize-sam-reference",action="store_true")
    a=p.parse_args()
    with exclusive_training_lock(a.output,lock_name="SAM_REFERENCE_V1.training.lock",stale_seconds=900):
        run(a)

if __name__=="__main__": main()
