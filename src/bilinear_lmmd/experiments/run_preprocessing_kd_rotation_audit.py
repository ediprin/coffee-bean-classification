from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bilinear_lmmd.data.preprocessing.study_data import (
    PreprocessingStudyImageFolder,
    deterministic_rotation_angle,
    identity_from_path,
)
from bilinear_lmmd.engine.preprocessing_kd import ARMS, load_frozen_teachers
from bilinear_lmmd.engine.train import resolve_device


ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0)
EPOCHS = 50
SEED = 42


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1.0e-12)
    return -(p * p.log()).sum(dim=1)


def _pairwise_js_tensor(probabilities: torch.Tensor) -> torch.Tensor:
    """Mean pairwise JS for [N, V, C] probabilities."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities harus [sample, view, class].")
    pair_values = []
    for left, right in itertools.combinations(range(probabilities.shape[1]), 2):
        p = probabilities[:, left].clamp_min(1.0e-12)
        q = probabilities[:, right].clamp_min(1.0e-12)
        m = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.log() - m.log())).sum(dim=1)
            + (q * (q.log() - m.log())).sum(dim=1)
        )
        pair_values.append(js)
    return torch.stack(pair_values, dim=1).mean(dim=1)


def _fixed_angle_loader(
    *,
    train_root: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    angle: float,
) -> tuple[DataLoader, list[str], list[str], torch.Tensor]:
    dataset = PreprocessingStudyImageFolder(
        train_root,
        image_size=image_size,
        train=True,
        seed=SEED,
        rotation_angles=[float(angle)],
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
    )
    identities = [identity_from_path(path) for path, _ in dataset.samples]
    paths = [str(path) for path, _ in dataset.samples]
    labels = torch.tensor([target for _, target in dataset.samples], dtype=torch.long)
    return loader, identities, paths, labels


@torch.no_grad()
def _predict_angle(
    *,
    loader: DataLoader,
    teachers,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    chunks = {arm: [] for arm in ARMS}
    labels = []
    for images, targets in loader:
        labels.append(targets.cpu())
        for arm in ARMS:
            teacher = teachers[arm]
            teacher.model.eval()
            model_input = teacher.runtime(images)
            probs = teacher.model(model_input).logits.softmax(1).cpu()
            chunks[arm].append(probs)
    return {
        "labels": torch.cat(labels),
        **{arm: torch.cat(value) for arm, value in chunks.items()},
    }


def _angle_summary(
    *,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    # probabilities: [N, 4, 17]
    predictions = probabilities.argmax(dim=2)
    any_disagreement = (predictions != predictions[:, 0:1]).any(dim=1)
    pairwise_js = _pairwise_js_tensor(probabilities)
    any_correct = predictions.eq(labels[:, None]).any(dim=1)
    ensemble = probabilities.mean(dim=1)
    ensemble_pred = ensemble.argmax(dim=1)

    pairwise = {}
    for left, right in itertools.combinations(range(len(ARMS)), 2):
        p = probabilities[:, left]
        q = probabilities[:, right]
        m = 0.5 * (p + q)
        js = 0.5 * (
            (p.clamp_min(1.0e-12) * (
                p.clamp_min(1.0e-12).log() - m.clamp_min(1.0e-12).log()
            )).sum(dim=1)
            + (q.clamp_min(1.0e-12) * (
                q.clamp_min(1.0e-12).log() - m.clamp_min(1.0e-12).log()
            )).sum(dim=1)
        )
        pairwise[f"{ARMS[left]}_vs_{ARMS[right]}"] = {
            "top1_disagreement": float(
                predictions[:, left].ne(predictions[:, right]).float().mean()
            ),
            "js_divergence": float(js.mean()),
        }

    summary = {
        "samples": int(labels.numel()),
        "any_teacher_top1_disagreement": float(any_disagreement.float().mean()),
        "mean_pairwise_js": float(pairwise_js.mean()),
        "oracle_any_teacher_correct": float(any_correct.float().mean()),
        "all4_accuracy": float(ensemble_pred.eq(labels).float().mean()),
        "teacher_entropy": {
            arm: float(_entropy(probabilities[:, index]).mean())
            for index, arm in enumerate(ARMS)
        },
        "all4_entropy": float(_entropy(ensemble).mean()),
        "pairwise": pairwise,
    }
    return summary, any_disagreement, pairwise_js


def run_rotation_audit(
    *,
    data_root: Path,
    authority_path: Path,
    experiments_root: Path,
    fold: int,
    output_dir: Path,
    device_name: str = "auto",
    image_size: int = 224,
    batch_size: int = 32,
    workers: int = 4,
) -> dict:
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    data_root = Path(data_root).expanduser().resolve()
    train_root = data_root / "source" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Train split tidak ditemukan: {train_root}")
    if (data_root / "source" / "test").exists():
        raise RuntimeError("Rotation audit tidak boleh mengekspos source/test.")

    device = resolve_device(device_name)
    teachers, classes, teacher_metadata = load_frozen_teachers(
        authority_path=authority_path,
        experiments_root=experiments_root,
        fold=fold,
        device=device,
        arms=ARMS,
    )

    angle_probabilities = []
    angle_disagreement = []
    angle_js = []
    per_angle = {}
    reference_identities = None
    reference_paths = None
    reference_labels = None

    for angle in ANGLES:
        loader, identities, paths, dataset_labels = _fixed_angle_loader(
            train_root=train_root,
            image_size=image_size,
            batch_size=batch_size,
            workers=workers,
            angle=angle,
        )
        if loader.dataset.classes != classes:
            raise RuntimeError("Urutan kelas train berbeda dari teacher.")
        if reference_identities is None:
            reference_identities = identities
            reference_paths = paths
            reference_labels = dataset_labels
        else:
            if identities != reference_identities:
                raise RuntimeError("Urutan identity berubah antar-angle.")
            if not torch.equal(dataset_labels, reference_labels):
                raise RuntimeError("Label berubah antar-angle.")

        result = _predict_angle(loader=loader, teachers=teachers, device=device)
        if not torch.equal(result["labels"], dataset_labels):
            raise RuntimeError("Label loader berbeda dari dataset labels.")
        stacked = torch.stack([result[arm] for arm in ARMS], dim=1)
        summary, disagreement, mean_js = _angle_summary(
            labels=dataset_labels,
            probabilities=stacked,
        )
        key = str(int(angle))
        per_angle[key] = summary
        angle_probabilities.append(stacked)
        angle_disagreement.append(disagreement)
        angle_js.append(mean_js)
        print(
            f"fold={fold} angle={angle:g} "
            f"disagree={summary['any_teacher_top1_disagreement']:.4%} "
            f"meanJS={summary['mean_pairwise_js']:.6f} "
            f"ALL4acc={summary['all4_accuracy']:.4%}",
            flush=True,
        )

    assert reference_identities is not None
    assert reference_paths is not None
    assert reference_labels is not None

    # [N, A, V, C], [N, A]
    all_probs = torch.stack(angle_probabilities, dim=1)
    all_disagreement = torch.stack(angle_disagreement, dim=1)
    all_js = torch.stack(angle_js, dim=1)

    any_angle_disagreement = all_disagreement.any(dim=1)
    disagree_angle_count = all_disagreement.sum(dim=1)
    max_js, max_js_index = all_js.max(dim=1)

    # Reconstruct the exact 50-epoch angle schedule used by the training loader.
    angle_to_index = {float(angle): index for index, angle in enumerate(ANGLES)}
    scheduled_indices = torch.empty(
        (len(reference_identities), EPOCHS),
        dtype=torch.long,
    )
    for sample_index, identity in enumerate(reference_identities):
        for epoch in range(EPOCHS):
            angle = deterministic_rotation_angle(
                seed=SEED,
                epoch=epoch,
                identity=identity,
                angles=list(ANGLES),
            )
            scheduled_indices[sample_index, epoch] = angle_to_index[float(angle)]

    sample_indices = torch.arange(len(reference_identities))[:, None]
    scheduled_disagreement = all_disagreement[
        sample_indices, scheduled_indices
    ]
    scheduled_js = all_js[sample_indices, scheduled_indices]
    scheduled_disagreement_epochs = scheduled_disagreement.sum(dim=1)
    scheduled_mean_js = scheduled_js.mean(dim=1)

    flat_schedule = scheduled_indices.flatten()
    schedule_counts = {
        str(int(angle)): int(flat_schedule.eq(index).sum())
        for index, angle in enumerate(ANGLES)
    }

    per_class = {}
    for class_index, class_name in enumerate(classes):
        mask = reference_labels.eq(class_index)
        if not mask.any():
            continue
        per_class[class_name] = {
            "support": int(mask.sum()),
            "any_disagreement_in_7_angles": float(
                any_angle_disagreement[mask].float().mean()
            ),
            "mean_disagreement_angle_count": float(
                disagree_angle_count[mask].float().mean()
            ),
            "mean_max_pairwise_js": float(max_js[mask].mean()),
            "scheduled_disagreement_exposure": float(
                scheduled_disagreement[mask].float().mean()
            ),
            "scheduled_mean_pairwise_js": float(
                scheduled_js[mask].mean()
            ),
        }

    zero = per_angle["0"]
    across = {
        "samples": len(reference_identities),
        "angles": [int(angle) for angle in ANGLES],
        "any_disagreement_in_7_angles": float(
            any_angle_disagreement.float().mean()
        ),
        "mean_disagreement_angle_count": float(
            disagree_angle_count.float().mean()
        ),
        "median_disagreement_angle_count": float(
            disagree_angle_count.float().median()
        ),
        "mean_max_pairwise_js": float(max_js.mean()),
        "zero_angle_disagreement": float(
            zero["any_teacher_top1_disagreement"]
        ),
        "zero_angle_mean_pairwise_js": float(zero["mean_pairwise_js"]),
        "disagreement_lift_any_angle_vs_zero": float(
            any_angle_disagreement.float().mean()
            - zero["any_teacher_top1_disagreement"]
        ),
        "scheduled_sample_epoch_exposures": int(
            scheduled_disagreement.numel()
        ),
        "scheduled_disagreement_exposure": float(
            scheduled_disagreement.float().mean()
        ),
        "scheduled_mean_pairwise_js": float(scheduled_js.mean()),
        "scheduled_images_with_at_least_one_disagreement_epoch": float(
            scheduled_disagreement.any(dim=1).float().mean()
        ),
        "mean_disagreement_epochs_per_image": float(
            scheduled_disagreement_epochs.float().mean()
        ),
        "schedule_angle_counts": schedule_counts,
    }

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = output_dir / "rotation_audit_per_image.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "identity",
                "path",
                "actual",
                "actual_index",
                "disagreement_angle_count",
                "any_disagreement_in_7_angles",
                "max_mean_pairwise_js",
                "max_js_angle",
                "scheduled_disagreement_epochs",
                "scheduled_disagreement_fraction",
                "scheduled_mean_pairwise_js",
            ]
        )
        for index, identity in enumerate(reference_identities):
            writer.writerow(
                [
                    identity,
                    reference_paths[index],
                    classes[int(reference_labels[index])],
                    int(reference_labels[index]),
                    int(disagree_angle_count[index]),
                    bool(any_angle_disagreement[index]),
                    float(max_js[index]),
                    int(ANGLES[int(max_js_index[index])]),
                    int(scheduled_disagreement_epochs[index]),
                    float(scheduled_disagreement[index].float().mean()),
                    float(scheduled_mean_js[index]),
                ]
            )

    payload = {
        "format": "bilinear_lmmd.preprocessing.kd_rotation_audit.v1",
        "protocol": "coffee17-preprocessing-kd-exploratory-v1",
        "scope": (
            "diagnostic-only; fixed seven-angle teacher-diversity audit on the "
            "training split; must not choose teacher weights or tune KD hyperparameters"
        ),
        "fold": int(fold),
        "seed": SEED,
        "epochs_reconstructed": EPOCHS,
        "teachers": teacher_metadata,
        "per_angle": per_angle,
        "across_angles_and_schedule": across,
        "per_class": per_class,
        "per_image_csv": str(sample_csv),
        "test_images_accessed": False,
    }
    output = output_dir / "rotation_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["across_angles_and_schedule"], indent=2), flush=True)
    print(f"SAVED: {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_rotation_audit(
        data_root=args.data_root,
        authority_path=args.authority,
        experiments_root=args.experiments_root,
        fold=args.fold,
        output_dir=args.output_dir,
        device_name=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
