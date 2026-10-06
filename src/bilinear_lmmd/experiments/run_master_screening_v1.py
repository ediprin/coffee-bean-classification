from __future__ import annotations

import argparse
import copy
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
from bilinear_lmmd.engine.wavelet_residual_hbp import (
    build_candidate as build_wr_hbp,
    preflight_matched_initialization as preflight_wr_hbp,
)
from bilinear_lmmd.experiments.run_frequency_screening_v1 import (
    METRICS,
    _configure_strict_determinism,
    _paired_outcomes,
    _prediction_rows,
    _run_complete,
    preflight_candidate as preflight_frequency,
    validate_config,
)
from bilinear_lmmd.experiments.run_representation_screening_v1 import (
    _candidate_loss as representation_loss,
    preflight_candidate as preflight_representation,
)
from bilinear_lmmd.modeling.frequency_screening import (
    FREQUENCY_CANDIDATES,
    build_frequency_screening_model,
)
from bilinear_lmmd.modeling.representation_screening import (
    REPRESENTATION_CANDIDATES,
    build_representation_screening_model,
)


PROTOCOL = "coffee17-master-representation-screening-v1"

MASTER_CANDIDATES = {
    "B0": {"anchor": "B0", "family": "anchor"},
    "B1": {"anchor": "B1", "family": "anchor"},
    "B2": {"anchor": "B1", "family": "wr_hbp"},
    "R1": {"anchor": "B0", "family": "representation"},
    "R2": {"anchor": "B1", "family": "representation"},
    "R3": {"anchor": "B1", "family": "representation"},
    "R4": {"anchor": "B1", "family": "representation"},
    "R5": {"anchor": "B1", "family": "representation"},
    "W1": {"anchor": "B0", "family": "frequency"},
    "W2": {"anchor": "B1", "family": "frequency"},
    "W3": {"anchor": "B0", "family": "frequency"},
    "W4": {"anchor": "B1", "family": "frequency"},
    "W5": {"anchor": "B0", "family": "frequency"},
    "W6": {"anchor": "B1", "family": "frequency"},
    "W7": {"anchor": "B0", "family": "representation"},
    "F1": {"anchor": "B0", "family": "representation"},
    "F2": {"anchor": "B0", "family": "representation"},
}


def build_master_model(candidate: str, cfg: dict) -> nn.Module:
    if candidate == "B2":
        return build_wr_hbp(cfg)
    if candidate in FREQUENCY_CANDIDATES:
        return build_frequency_screening_model(candidate, cfg)
    if candidate in REPRESENTATION_CANDIDATES:
        return build_representation_screening_model(candidate, cfg)
    raise ValueError(f"Candidate tidak dikenal: {candidate}")


@torch.no_grad()
def preflight_master(cfg: dict, candidate: str) -> dict:
    if candidate == "B2":
        payload = dict(preflight_wr_hbp(cfg))
        payload["candidate"] = "B2"
        payload["anchor"] = "B1"
        payload["matched_initialization_scope"] = "rgb_hbp_shared_core"
        return payload
    if candidate in FREQUENCY_CANDIDATES:
        return preflight_frequency(cfg, candidate)
    if candidate in REPRESENTATION_CANDIDATES:
        return preflight_representation(cfg, candidate)
    raise ValueError(candidate)


def _forward(
    candidate: str,
    model: nn.Module,
    raw_images: torch.Tensor,
    labels: torch.Tensor | None = None,
):
    normalized = imagenet_normalize(raw_images)
    if candidate == "B2":
        return model(normalized, raw_rgb=raw_images, labels=labels)
    return model(normalized, labels=labels)


def _loss(
    cfg: dict,
    candidate: str,
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if candidate in REPRESENTATION_CANDIDATES and candidate not in {"B0", "B1"}:
        return representation_loss(cfg, candidate, model, logits, labels)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=float(cfg["training"]["label_smoothing"]),
    )


@torch.no_grad()
def _evaluate(
    candidate: str,
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
        probs = _forward(candidate, model, raw_images).logits.softmax(1).cpu()
        labels.extend(targets.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())
    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError("Jumlah validation path berbeda dari prediksi.")
    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    return metrics, labels, predictions, probabilities, paths


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
    model = build_master_model(candidate, cfg).to(device)

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
            optimizer.zero_grad(set_to_none=True)
            output = _forward(candidate, model, raw_images, labels)
            loss = _loss(cfg, candidate, model, output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        metrics, _, _, _, _ = _evaluate(
            candidate, model, loaders.val, device, loaders.classes, hard_groups
        )
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(1, batches),
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if candidate == "B2":
            record["wavelet_gate"] = float(model.gate_value().detach().cpu().item())
        history.append(record)

        message = {
            "candidate": candidate,
            "epoch": epoch + 1,
            "macro_f1": metrics["macro_f1"],
            "hard_class_f1": metrics["hard_class_f1"],
            "worst_class_f1": metrics["worst_class_f1"],
        }
        if candidate == "B2":
            message["wavelet_gate"] = record["wavelet_gate"]
        print(json.dumps(message), flush=True)

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
            best_payload = {
                "model": model.state_dict(),
                "classes": loaders.classes,
                "config": cfg,
                "candidate": candidate,
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "run_contract_sha256": run_contract_sha256,
            }
            if candidate == "B2":
                best_payload["wavelet_gate"] = float(
                    model.gate_value().detach().cpu().item()
                )
            atomic_torch_save(best_payload, best_path)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    eval_model = build_master_model(candidate, cfg)
    eval_model.load_state_dict(best["model"])
    eval_model = eval_model.to(device)
    metrics, labels, predictions, probabilities, paths = _evaluate(
        candidate, eval_model, loaders.val, device, loaders.classes, hard_groups
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
    unknown = sorted(set(candidates).difference(MASTER_CANDIDATES))
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

    required_anchors = {MASTER_CANDIDATES[c]["anchor"] for c in candidates}
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
        lock_name=f"MASTER_SCREEN_V1_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        for candidate in ordered:
            preflight = preflight_master(cfg, candidate)
            run_dir = fold_root / candidate
            run_dir.mkdir(parents=True, exist_ok=True)
            probe_model = build_master_model(candidate, cfg)
            parameter_count = sum(p.numel() for p in probe_model.parameters())
            del probe_model

            contract = {
                "format": "bilinear_lmmd.master_screening.arm_contract.v1",
                "protocol": PROTOCOL,
                "fold": fold,
                "seed": 42,
                "candidate": candidate,
                "family": MASTER_CANDIDATES[candidate]["family"],
                "anchor": MASTER_CANDIDATES[candidate]["anchor"],
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
            result = {
                "metrics": {key: float(metrics[key]) for key in METRICS},
                "parameter_count": parameter_count,
                "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                "anchor": MASTER_CANDIDATES[candidate]["anchor"],
                "family": MASTER_CANDIDATES[candidate]["family"],
            }
            if candidate == "B2":
                best = torch.load(
                    run_dir / "best.pt", map_location="cpu", weights_only=False
                )
                result["wavelet_gate_at_best"] = float(best["wavelet_gate"])
            results[candidate] = result

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
        "format": "bilinear_lmmd.master_screening.fold_result.v1",
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
        description="Unified Coffee17 paper-grounded representation screening V1"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(MASTER_CANDIDATES),
        default=[
            "B2",
            "R1", "R2", "R3", "R4", "R5",
            "W1", "W2", "W3", "W4", "W5", "W6", "W7",
            "F1", "F2",
        ],
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
