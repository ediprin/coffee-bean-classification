from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_evaluation_loader,
)
from bilinear_lmmd.engine.preprocessing_kd import (
    ARMS,
    load_frozen_teachers,
)
from bilinear_lmmd.engine.train import resolve_device


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1.0e-12)
    return -(p * p.log()).sum(dim=1)


def _js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1.0e-12)
    q = q.clamp_min(1.0e-12)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - m.log())).sum(dim=1)
        + (q * (q.log() - m.log())).sum(dim=1)
    )


@torch.no_grad()
def audit_split(
    *,
    loader,
    teachers,
    device: torch.device,
    classes: list[str],
) -> dict:
    all_probs = {arm: [] for arm in ARMS}
    all_labels = []

    for images, labels in loader:
        all_labels.append(labels.cpu())
        for arm in ARMS:
            teacher = teachers[arm]
            model_input = teacher.runtime(images)
            probs = teacher.model(model_input).logits.softmax(1).cpu()
            all_probs[arm].append(probs)

    labels = torch.cat(all_labels)
    probs = {arm: torch.cat(chunks) for arm, chunks in all_probs.items()}
    preds = {arm: value.argmax(dim=1) for arm, value in probs.items()}

    pairwise = {}
    for left, right in itertools.combinations(ARMS, 2):
        key = f"{left}_vs_{right}"
        pairwise[key] = {
            "top1_disagreement": float((preds[left] != preds[right]).float().mean()),
            "js_divergence": float(_js(probs[left], probs[right]).mean()),
        }

    stacked_preds = torch.stack([preds[arm] for arm in ARMS], dim=0)
    any_disagreement = (stacked_preds != stacked_preds[0:1]).any(dim=0)
    any_correct = torch.stack(
        [preds[arm].eq(labels) for arm in ARMS],
        dim=0,
    ).any(dim=0)

    ensemble = torch.stack([probs[arm] for arm in ARMS], dim=0).mean(dim=0)
    ensemble_pred = ensemble.argmax(dim=1)

    class_rows = {}
    for class_index, name in enumerate(classes):
        mask = labels.eq(class_index)
        if not mask.any():
            continue
        class_rows[name] = {
            "support": int(mask.sum()),
            "any_teacher_top1_disagreement": float(any_disagreement[mask].float().mean()),
            "oracle_any_teacher_correct": float(any_correct[mask].float().mean()),
            "all4_accuracy": float(ensemble_pred[mask].eq(labels[mask]).float().mean()),
        }

    return {
        "samples": int(labels.numel()),
        "pairwise": pairwise,
        "any_teacher_top1_disagreement": float(any_disagreement.float().mean()),
        "teacher_entropy": {
            arm: float(_entropy(probs[arm]).mean()) for arm in ARMS
        },
        "all4_entropy": float(_entropy(ensemble).mean()),
        "oracle_any_teacher_correct": float(any_correct.float().mean()),
        "all4_accuracy": float(ensemble_pred.eq(labels).float().mean()),
        "per_class": class_rows,
    }


def run_audit(
    *,
    data_root: Path,
    authority_path: Path,
    experiments_root: Path,
    fold: int,
    output: Path,
    device_name: str = "auto",
) -> dict:
    device = resolve_device(device_name)
    teachers, classes, teacher_metadata = load_frozen_teachers(
        authority_path=authority_path,
        experiments_root=experiments_root,
        fold=fold,
        device=device,
    )

    data_cfg = {
        "root": str(Path(data_root).expanduser().resolve()),
        "source": "source",
        "image_size": 224,
        "batch_size": 32,
        "workers": 4,
    }

    split_reports = {}
    for split in ("train", "val"):
        loader, split_classes = build_preprocessing_evaluation_loader(
            data_cfg,
            split=split,
            seed=42,
        )
        if split_classes != classes:
            raise RuntimeError(f"Urutan kelas {split} berbeda dari teacher.")
        split_reports[split] = audit_split(
            loader=loader,
            teachers=teachers,
            device=device,
            classes=classes,
        )

    payload = {
        "format": "bilinear_lmmd.preprocessing.kd_teacher_audit.v1",
        "protocol": "coffee17-preprocessing-kd-exploratory-v1",
        "scope": "diagnostic-only; must not choose teacher weights or views",
        "fold": int(fold),
        "seed": 42,
        "teachers": teacher_metadata,
        "splits": split_reports,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_audit(
        data_root=args.data_root,
        authority_path=args.authority,
        experiments_root=args.experiments_root,
        fold=args.fold,
        output=args.output,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
