from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from bilinear_lmmd.analysis.nuisance import (
    NUISANCE_KINDS,
    apply_acquisition_nuisance,
    cosine_stability,
    fisher_separability,
    normalized_l2_shift,
)
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_evaluation_loader,
)
from bilinear_lmmd.engine.at_sbn import AUX_VIEWS, VIEWS, AuxiliaryTrainingModel
from bilinear_lmmd.engine.shared_multiview_sbn import auxiliary_batchnorm_batch_stats
from bilinear_lmmd.engine.train import classification_metrics, resolve_device


PROTOCOL = "coffee17-preprocessing-nuisance-analysis-v1"


def _load_model(checkpoint_path: Path, data_root: Path, device: torch.device):
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    cfg["model"]["pretrained"] = False

    model = AuxiliaryTrainingModel(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader, classes = build_preprocessing_evaluation_loader(
        cfg["data"],
        split="val",
        seed=int(cfg["seed"]),
    )
    if checkpoint.get("classes") != classes:
        raise RuntimeError("Urutan kelas checkpoint dan validation berbeda.")

    runtimes = {
        arm: PreprocessingRuntime.from_config(
            cfg["feature_distillation"]["frontends"][arm],
            device,
        )
        for arm in VIEWS
    }
    return checkpoint, cfg, model, loader, classes, runtimes


@torch.no_grad()
def _collect(
    *,
    model: AuxiliaryTrainingModel,
    loader,
    runtime: PreprocessingRuntime,
    arm: str,
    device: torch.device,
    nuisance: str | None,
    nuisance_seed: int,
):
    all_embeddings = []
    labels = []
    predictions = []

    for batch_index, (images, targets) in enumerate(loader):
        if nuisance is not None:
            images = apply_acquisition_nuisance(
                images,
                nuisance,
                seed=int(nuisance_seed) + batch_index,
            )
        view_input = runtime(images)

        if arm == "R0":
            output = model.forward_primary(view_input)
        else:
            with auxiliary_batchnorm_batch_stats(model.base):
                output = model.forward_auxiliary(arm, view_input)

        all_embeddings.append(output.embedding.detach().cpu().float())
        predictions.extend(output.logits.argmax(1).detach().cpu().tolist())
        labels.extend(targets.tolist())

    return (
        torch.cat(all_embeddings, dim=0),
        torch.tensor(labels, dtype=torch.long),
        predictions,
    )


def run_analysis(
    *,
    checkpoint: Path,
    data_root: Path,
    output: Path,
    fold: int,
    device_name: str,
) -> dict:
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    device = resolve_device(device_name)
    _, cfg, model, loader, classes, runtimes = _load_model(
        checkpoint, data_root, device
    )
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    clean = {}
    for arm in VIEWS:
        embedding, labels, predictions = _collect(
            model=model,
            loader=loader,
            runtime=runtimes[arm],
            arm=arm,
            device=device,
            nuisance=None,
            nuisance_seed=42000 + fold * 1000,
        )
        metrics = classification_metrics(
            labels.tolist(), predictions, classes, hard_groups
        )
        clean[arm] = {
            "embedding": embedding,
            "labels": labels,
            "predictions": predictions,
            "metrics": metrics,
            "fisher_separability": fisher_separability(embedding, labels),
        }

    nuisances = {}
    for nuisance_index, nuisance in enumerate(NUISANCE_KINDS):
        by_view = {}
        for arm in VIEWS:
            embedding, labels, predictions = _collect(
                model=model,
                loader=loader,
                runtime=runtimes[arm],
                arm=arm,
                device=device,
                nuisance=nuisance,
                nuisance_seed=(
                    42000 + fold * 1000 + nuisance_index * 100
                ),
            )
            if not torch.equal(labels, clean[arm]["labels"]):
                raise RuntimeError("Urutan label nuisance berubah.")

            metrics = classification_metrics(
                labels.tolist(), predictions, classes, hard_groups
            )
            fisher = fisher_separability(embedding, labels)
            clean_metrics = clean[arm]["metrics"]
            clean_fisher = clean[arm]["fisher_separability"]

            by_view[arm] = {
                "cosine_stability": cosine_stability(
                    clean[arm]["embedding"], embedding
                ),
                "normalized_l2_shift": normalized_l2_shift(
                    clean[arm]["embedding"], embedding
                ),
                "fisher_separability": fisher,
                "fisher_retention": (
                    fisher / clean_fisher if clean_fisher > 0.0 else None
                ),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_drop": (
                    metrics["macro_f1"] - clean_metrics["macro_f1"]
                ),
                "balanced_accuracy_drop": (
                    metrics["balanced_accuracy"]
                    - clean_metrics["balanced_accuracy"]
                ),
                "per_class": metrics["per_class"],
            }
        nuisances[nuisance] = by_view

    clean_payload = {
        arm: {
            "accuracy": clean[arm]["metrics"]["accuracy"],
            "balanced_accuracy": clean[arm]["metrics"]["balanced_accuracy"],
            "macro_f1": clean[arm]["metrics"]["macro_f1"],
            "fisher_separability": clean[arm]["fisher_separability"],
            "per_class": clean[arm]["metrics"]["per_class"],
        }
        for arm in VIEWS
    }

    payload = {
        "format": "coffee17.preprocessing_nuisance_analysis.fold.v1",
        "protocol": PROTOCOL,
        "scope": (
            "development-validation diagnostic only; synthetic acquisition-like "
            "perturbations; frozen MVFD-SBN checkpoint; no retraining"
        ),
        "fold": int(fold),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "num_validation_images": len(loader.dataset),
        "classes": classes,
        "views": list(VIEWS),
        "nuisance_kinds": list(NUISANCE_KINDS),
        "clean": clean_payload,
        "nuisances": nuisances,
        "test_images_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_analysis(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        output=args.output,
        fold=args.fold,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
