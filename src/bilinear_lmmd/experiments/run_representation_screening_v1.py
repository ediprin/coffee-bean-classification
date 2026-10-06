from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    capture_rng_state,
    current_git_commit,
    restore_rng_state,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.engine.preprocessing_study import _write_evaluation
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.train import (
    atomic_torch_save,
    classification_metrics,
    resolve_device,
)
from bilinear_lmmd.experiments.run_frequency_screening_v1 import (
    METRICS,
    _configure_strict_determinism,
    _paired_outcomes,
    _prediction_rows,
    _run_complete,
    validate_config,
)
from bilinear_lmmd.modeling.frequency_screening import assert_shared_core_equal
from bilinear_lmmd.modeling.representation_screening import (
    REPRESENTATION_CANDIDATES,
    LowRankBilinearClassifier,
    build_representation_screening_model,
    focal_loss,
    lrbp_hinge_loss,
)


PROTOCOL = "coffee17-representation-screening-v1"


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("encoder.")
    }


def _assert_encoder_equal(anchor: nn.Module, candidate: nn.Module) -> None:
    left = _encoder_state(anchor)
    right = _encoder_state(candidate)
    if left.keys() != right.keys():
        raise RuntimeError("Encoder keys candidate/anchor berbeda.")
    for key in left:
        if not torch.equal(left[key], right[key]):
            raise RuntimeError(f"Encoder initialization berbeda pada {key}.")


@torch.no_grad()
def preflight_candidate(cfg: dict, candidate: str) -> dict:
    validate_config(cfg)
    spec = REPRESENTATION_CANDIDATES[candidate]
    anchor_name = spec["anchor"]

    seed_everything(42)
    anchor = build_representation_screening_model(anchor_name, cfg)
    seed_everything(42)
    model = build_representation_screening_model(candidate, cfg)

    if candidate in {"R1", "R2"}:
        assert_shared_core_equal(anchor, model)
        equality = "encoder_pool_classifier"
    elif candidate == "R5":
        left = anchor.state_dict()
        right = model.state_dict()
        if left.keys() != right.keys():
            raise RuntimeError("R5/B1 state keys berbeda.")
        for key in left:
            if not torch.equal(left[key], right[key]):
                raise RuntimeError(f"R5/B1 initialization berbeda pada {key}.")
        equality = "entire_model"
    elif candidate == anchor_name:
        equality = "self"
    else:
        _assert_encoder_equal(anchor, model)
        equality = "encoder_only"

    return {
        "candidate": candidate,
        "anchor": anchor_name,
        "matched_initialization_scope": equality,
        "anchor_encoder_sha256": _state_sha(_encoder_state(anchor)),
        "candidate_encoder_sha256": _state_sha(_encoder_state(model)),
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    classes: list[str],
    hard_groups: dict,
) -> tuple[dict, list[int], list[int], list[list[float]], list[str]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for raw_images, targets in loader:
        raw_images = raw_images.to(device, non_blocking=True)
        normalized = imagenet_normalize(raw_images)
        probs = model(normalized).logits.softmax(1).cpu()
        labels.extend(targets.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())
    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah validation path berbeda dari prediksi.")
    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


def _candidate_loss(
    cfg: dict,
    candidate: str,
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    loss_kind = REPRESENTATION_CANDIDATES[candidate]["loss"]
    if loss_kind == "ce":
        return F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(cfg["training"]["label_smoothing"]),
        )
    if loss_kind == "ce_focal":
        rep = cfg["representation"]
        ce = F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(cfg["training"]["label_smoothing"]),
        )
        focal = focal_loss(
            logits,
            labels,
            gamma=float(rep.get("fusion_loss_gamma", 2.0)),
        )
        return (
            float(rep.get("fusion_loss_ce_weight", 0.7)) * ce
            + float(rep.get("fusion_loss_focal_weight", 0.3)) * focal
        )
    if loss_kind == "lrbp_hinge":
        if not isinstance(model, LowRankBilinearClassifier):
            raise TypeError("R3 harus LowRankBilinearClassifier.")
        return lrbp_hinge_loss(
            model,
            logits,
            labels,
            regularization_weight=float(
                cfg["representation"].get("lrbp_regularization", 5e-4)
            ),
        )
    raise RuntimeError(f"Loss tidak dikenal: {loss_kind}")


def train_candidate(
    cfg: dict,
    *,
    candidate: str,
    run_dir: Path,
    run_contract_sha256: str,
    resume: bool,
) -> dict:
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)
    model = build_representation_screening_model(candidate, cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
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
            loss = _candidate_loss(cfg, candidate, model, output.logits, labels)
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
    eval_model = build_representation_screening_model(candidate, cfg)
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


def run_fold(
    *,
    config_path: Path,
    fold: int,
    data_root: Path,
    output_root: Path,
    candidates: list[str],
    device: str,
    resume: bool,
    authorize_training: bool,
    required_commit: str | None,
    strict_determinism: bool,
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")
    unknown = sorted(set(candidates).difference(REPRESENTATION_CANDIDATES))
    if unknown:
        raise ValueError(f"Candidate tidak dikenal: {unknown}")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if required_commit and actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda: {actual_commit} != {required_commit}"
        )

    cfg = copy.deepcopy(load_config(config_path))
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    validate_config(cfg)
    determinism = (
        _configure_strict_determinism(int(cfg["seed"]))
        if strict_determinism
        else None
    )
    val_count, val_sha = validation_identity_label_sha256(data_root)

    required_anchors = {
        REPRESENTATION_CANDIDATES[candidate]["anchor"] for candidate in candidates
    }
    ordered = []
    for name in ("B0", "B1"):
        if name in required_anchors or name in candidates:
            ordered.append(name)
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)

    fold_root = (
        Path(output_root).expanduser().resolve()
        / f"fold_{fold}"
        / "seed42"
    )
    fold_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    with exclusive_training_lock(
        Path(output_root).expanduser().resolve(),
        lock_name=f"REP_SCREEN_V1_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        for candidate in ordered:
            preflight = preflight_candidate(cfg, candidate)
            run_dir = fold_root / candidate
            run_dir.mkdir(parents=True, exist_ok=True)
            parameter_count = sum(
                parameter.numel()
                for parameter in build_representation_screening_model(
                    candidate, cfg
                ).parameters()
            )
            contract = {
                "format": "bilinear_lmmd.representation_screening.arm_contract.v1",
                "protocol": PROTOCOL,
                "fold": fold,
                "seed": 42,
                "candidate": candidate,
                "anchor": REPRESENTATION_CANDIDATES[candidate]["anchor"],
                "loss": REPRESENTATION_CANDIDATES[candidate]["loss"],
                "git_commit": actual_commit,
                "validation_identity_label_sha256": val_sha,
                "validation_count": val_count,
                "preflight": preflight,
                "strict_determinism": determinism,
                "parameter_count": parameter_count,
                "outer_test_accessed": False,
                "resolved_config_sha256": canonical_json_sha256(cfg),
            }
            contract["run_contract_sha256"] = canonical_json_sha256(contract)
            contract_path = run_dir / "run_contract.json"
            if contract_path.is_file():
                existing = json.loads(contract_path.read_text(encoding="utf-8"))
                if existing != contract:
                    raise RuntimeError(f"Run contract berbeda: {contract_path}")
            else:
                contract_path.write_text(
                    json.dumps(contract, indent=2) + "\n", encoding="utf-8"
                )
                (run_dir / "run_config.yaml").write_text(
                    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )

            if _run_complete(run_dir, int(cfg["training"]["epochs"])):
                print(f"SKIP {candidate}: completed.", flush=True)
            else:
                train_candidate(
                    cfg,
                    candidate=candidate,
                    run_dir=run_dir,
                    run_contract_sha256=contract["run_contract_sha256"],
                    resume=resume or (run_dir / "last.pt").is_file(),
                )

            metrics = json.loads(
                (run_dir / "validation" / "metrics.json").read_text(encoding="utf-8")
            )
            results[candidate] = {
                "metrics": {key: float(metrics[key]) for key in METRICS},
                "parameter_count": parameter_count,
                "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                "anchor": REPRESENTATION_CANDIDATES[candidate]["anchor"],
            }

    for candidate in ordered:
        anchor = results[candidate]["anchor"]
        if candidate == anchor:
            results[candidate]["delta_vs_anchor"] = {key: 0.0 for key in METRICS}
            continue
        results[candidate]["delta_vs_anchor"] = {
            key: results[candidate]["metrics"][key] - results[anchor]["metrics"][key]
            for key in METRICS
        }
        control_rows = _prediction_rows(
            fold_root / anchor / "validation" / "predictions.csv"
        )
        candidate_rows = _prediction_rows(
            fold_root / candidate / "validation" / "predictions.csv"
        )
        results[candidate]["paired_outcomes"] = _paired_outcomes(
            control_rows, candidate_rows
        )

    payload = {
        "format": "bilinear_lmmd.representation_screening.fold_result.v1",
        "protocol": PROTOCOL,
        "fold": fold,
        "seed": 42,
        "git_commit": actual_commit,
        "validation_identity_label_sha256": val_sha,
        "validation_count": val_count,
        "candidates": ordered,
        "results": results,
        "outer_test_accessed": False,
    }
    (fold_root / "fold_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coffee17 representation/spectral architecture screening V1"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(REPRESENTATION_CANDIDATES),
        default=["R1", "R2", "R3", "R4", "R5", "W7", "F1", "F2"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--required-commit")
    parser.add_argument("--strict-determinism", action="store_true")
    args = parser.parse_args()
    run_fold(
        config_path=args.config,
        fold=args.fold,
        data_root=args.data_root,
        output_root=args.output_root,
        candidates=args.candidates,
        device=args.device,
        resume=args.resume,
        authorize_training=args.authorize_training,
        required_commit=args.required_commit,
        strict_determinism=args.strict_determinism,
    )


if __name__ == "__main__":
    main()
