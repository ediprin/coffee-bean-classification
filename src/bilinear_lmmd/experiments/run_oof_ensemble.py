from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from bilinear_lmmd.engine.evaluate_checkpoint import (
    CheckpointPredictions,
    collect_checkpoint_predictions,
)
from bilinear_lmmd.engine.train import classification_metrics


@dataclass(frozen=True)
class AlignedPredictions:
    identities: list[str]
    paths: list[str]
    labels: list[int]
    gap_probabilities: torch.Tensor
    hbp_probabilities: torch.Tensor
    classes: list[str]
    hard_groups: dict[str, list[str]]


def _identity(path: str) -> str:
    image_path = Path(path)
    return f"{image_path.parent.name}/{image_path.name}"


def align_predictions(
    gap: CheckpointPredictions, hbp: CheckpointPredictions
) -> AlignedPredictions:
    if gap.classes != hbp.classes:
        raise ValueError("Urutan kelas checkpoint GAP dan HBP berbeda.")
    if gap.hard_groups != hbp.hard_groups:
        raise ValueError("Definisi hard-group checkpoint GAP dan HBP berbeda.")

    def index_bundle(bundle: CheckpointPredictions) -> dict[str, int]:
        result: dict[str, int] = {}
        for index, path in enumerate(bundle.paths):
            identity = _identity(path)
            if identity in result:
                raise ValueError(f"Identitas prediksi duplikat: {identity}")
            result[identity] = index
        return result

    gap_index = index_bundle(gap)
    hbp_index = index_bundle(hbp)
    if gap_index.keys() != hbp_index.keys():
        gap_only = sorted(gap_index.keys() - hbp_index.keys())
        hbp_only = sorted(hbp_index.keys() - gap_index.keys())
        raise ValueError(
            "Sampel GAP dan HBP berbeda. "
            f"GAP-only={gap_only[:3]}, HBP-only={hbp_only[:3]}"
        )

    identities = sorted(gap_index)
    gap_indices = [gap_index[identity] for identity in identities]
    hbp_indices = [hbp_index[identity] for identity in identities]
    gap_labels = [gap.labels[index] for index in gap_indices]
    hbp_labels = [hbp.labels[index] for index in hbp_indices]
    if gap_labels != hbp_labels:
        raise ValueError("Label aktual GAP dan HBP berbeda.")

    return AlignedPredictions(
        identities=identities,
        paths=[gap.paths[index] for index in gap_indices],
        labels=gap_labels,
        gap_probabilities=gap.probabilities[gap_indices],
        hbp_probabilities=hbp.probabilities[hbp_indices],
        classes=gap.classes,
        hard_groups=gap.hard_groups,
    )


def _alpha_grid(step: float) -> list[float]:
    if not 0 < step <= 1:
        raise ValueError("alpha-step harus berada pada interval (0, 1].")
    intervals = round(1.0 / step)
    if not math.isclose(intervals * step, 1.0, abs_tol=1e-9):
        raise ValueError("alpha-step harus membagi interval 0 sampai 1 secara tepat.")
    return [index / intervals for index in range(intervals + 1)]


def ensemble_predictions(
    gap_probabilities: torch.Tensor,
    hbp_probabilities: torch.Tensor,
    alpha: float,
) -> list[int]:
    probabilities = (1.0 - alpha) * gap_probabilities + alpha * hbp_probabilities
    return probabilities.argmax(1).tolist()


def select_alpha(
    aligned: AlignedPredictions,
    step: float = 0.05,
    objective: str = "macro_f1",
) -> tuple[float, list[dict[str, float]]]:
    curve: list[dict[str, float]] = []
    best_score = -1.0
    best_alphas: list[float] = []
    for alpha in _alpha_grid(step):
        predictions = ensemble_predictions(
            aligned.gap_probabilities, aligned.hbp_probabilities, alpha
        )
        metrics = classification_metrics(
            aligned.labels, predictions, aligned.classes, aligned.hard_groups
        )
        score = metrics[objective]
        if score is None:
            raise ValueError(f"Metrik objective {objective} tidak tersedia.")
        score = float(score)
        curve.append({"alpha": alpha, "score": score})
        if score > best_score + 1e-12:
            best_score = score
            best_alphas = [alpha]
        elif math.isclose(score, best_score, abs_tol=1e-12):
            best_alphas.append(alpha)

    # Validation predictions are discrete, so adjacent alpha values often tie.
    # Prefer the least extreme mixture instead of arbitrarily selecting an endpoint.
    selected = min(best_alphas, key=lambda alpha: (abs(alpha - 0.5), -alpha))
    return selected, curve


def _compact_metrics(metrics: dict) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "hard_class_f1": metrics["hard_class_f1"],
        "worst_class_f1": metrics["worst_class_f1"],
        "hard_groups": metrics["hard_groups"],
    }


def run_oof_ensemble(
    data_root: Path,
    output_root: Path,
    folds: int = 5,
    seed: int = 42,
    alpha_step: float = 0.05,
    objective: str = "macro_f1",
    expected_count: int = 979,
) -> dict:
    all_labels: list[int] = []
    gap_predictions: list[int] = []
    hbp_predictions: list[int] = []
    fused_predictions: list[int] = []
    prediction_rows: list[dict] = []
    fold_results: list[dict] = []
    seen_identities: set[str] = set()
    classes: list[str] | None = None
    hard_groups: dict[str, list[str]] | None = None

    for fold in range(1, folds + 1):
        print(f"\n=== FOLD {fold}/{folds} ===", flush=True)
        fold_root = data_root / f"fold_{fold}"
        gap_checkpoint = (
            output_root / "outputs" / f"M0_fold{fold}_seed{seed}" / "best.pt"
        )
        hbp_checkpoint = (
            output_root / "outputs" / f"M1_fold{fold}_seed{seed}" / "best.pt"
        )
        missing = [
            str(path)
            for path in (gap_checkpoint, hbp_checkpoint)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Checkpoint belum tersedia:\n- " + "\n- ".join(missing))

        validation = align_predictions(
            collect_checkpoint_predictions(
                gap_checkpoint,
                "source",
                "val",
                data_root=fold_root,
                progress_desc=f"fold {fold} GAP val",
            ),
            collect_checkpoint_predictions(
                hbp_checkpoint,
                "source",
                "val",
                data_root=fold_root,
                progress_desc=f"fold {fold} HBP val",
            ),
        )
        alpha, alpha_curve = select_alpha(validation, alpha_step, objective)
        print(f"alpha HBP terpilih dari validation: {alpha:.2f}", flush=True)

        test = align_predictions(
            collect_checkpoint_predictions(
                gap_checkpoint,
                "source",
                "test",
                data_root=fold_root,
                progress_desc=f"fold {fold} GAP test",
            ),
            collect_checkpoint_predictions(
                hbp_checkpoint,
                "source",
                "test",
                data_root=fold_root,
                progress_desc=f"fold {fold} HBP test",
            ),
        )
        if classes is None:
            classes = test.classes
            hard_groups = test.hard_groups
        elif classes != test.classes or hard_groups != test.hard_groups:
            raise ValueError("Kelas atau hard-group antar-fold berbeda.")

        overlap = seen_identities.intersection(test.identities)
        if overlap:
            raise ValueError(f"Identitas test OOF duplikat: {sorted(overlap)[:3]}")
        seen_identities.update(test.identities)

        gap_fold_predictions = test.gap_probabilities.argmax(1).tolist()
        hbp_fold_predictions = test.hbp_probabilities.argmax(1).tolist()
        fused_fold_probabilities = (
            (1.0 - alpha) * test.gap_probabilities
            + alpha * test.hbp_probabilities
        )
        fused_fold_predictions = fused_fold_probabilities.argmax(1).tolist()

        all_labels.extend(test.labels)
        gap_predictions.extend(gap_fold_predictions)
        hbp_predictions.extend(hbp_fold_predictions)
        fused_predictions.extend(fused_fold_predictions)
        fold_metrics = classification_metrics(
            test.labels,
            fused_fold_predictions,
            test.classes,
            test.hard_groups,
        )
        fold_results.append(
            {
                "fold": fold,
                "alpha_hbp": alpha,
                "validation_curve": alpha_curve,
                "test_metrics": _compact_metrics(fold_metrics),
            }
        )

        for index, identity in enumerate(test.identities):
            actual = test.labels[index]
            gap_predicted = gap_fold_predictions[index]
            hbp_predicted = hbp_fold_predictions[index]
            fused_predicted = fused_fold_predictions[index]
            row = {
                "identity": identity,
                "fold": fold,
                "alpha_hbp": alpha,
                "actual": test.classes[actual],
                "predicted": test.classes[fused_predicted],
                "correct": int(actual == fused_predicted),
                "gap_predicted": test.classes[gap_predicted],
                "hbp_predicted": test.classes[hbp_predicted],
            }
            row.update(
                {
                    f"prob::{name}": probability
                    for name, probability in zip(
                        test.classes, fused_fold_probabilities[index].tolist()
                    )
                }
            )
            prediction_rows.append(row)

    if len(seen_identities) != expected_count:
        raise ValueError(
            f"OOF harus mencakup {expected_count} identitas, "
            f"ditemukan {len(seen_identities)}."
        )
    if classes is None or hard_groups is None:
        raise RuntimeError("Tidak ada fold yang dievaluasi.")

    gap_metrics = classification_metrics(
        all_labels, gap_predictions, classes, hard_groups
    )
    hbp_metrics = classification_metrics(
        all_labels, hbp_predictions, classes, hard_groups
    )
    ensemble_metrics = classification_metrics(
        all_labels, fused_predictions, classes, hard_groups
    )
    comparison = {
        "method": "validation-tuned probability ensemble",
        "alpha_definition": "p = (1-alpha) * GAP + alpha * HBP",
        "alpha_objective": objective,
        "alpha_step": alpha_step,
        "sample_count": len(all_labels),
        "classes": classes,
        "folds": fold_results,
        "gap": gap_metrics,
        "hbp": hbp_metrics,
        "ensemble": ensemble_metrics,
        "delta_ensemble_vs_hbp": {
            key: ensemble_metrics[key] - hbp_metrics[key]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "hard_class_f1", "worst_class_f1")
        },
    }

    output_dir = output_root / "oof" / f"M0_M1_ensemble_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    metrics_payload = dict(ensemble_metrics)
    metrics_payload.update(
        {
            "method": comparison["method"],
            "alpha_definition": comparison["alpha_definition"],
            "alpha_objective": objective,
            "fold_alphas": [result["alpha_hbp"] for result in fold_results],
            "sample_count": len(all_labels),
            "classes": classes,
            "baselines": {
                "gap": _compact_metrics(gap_metrics),
                "hbp": _compact_metrics(hbp_metrics),
            },
            "delta_ensemble_vs_hbp": comparison["delta_ensemble_vs_hbp"],
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "identity",
        "fold",
        "alpha_hbp",
        "actual",
        "predicted",
        "correct",
        "gap_predicted",
        "hbp_predicted",
        *(f"prob::{name}" for name in classes),
    ]
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(prediction_rows, key=lambda row: row["identity"]))

    print("\n=== HASIL OOF ENSEMBLE ===")
    print("Alpha HBP per fold:", [result["alpha_hbp"] for result in fold_results])
    for label, metrics in (
        ("GAP", gap_metrics),
        ("HBP", hbp_metrics),
        ("ENSEMBLE", ensemble_metrics),
    ):
        print(
            f"{label:8s} Macro-F1={metrics['macro_f1']:.2%} "
            f"Hard-F1={metrics['hard_class_f1']:.2%} "
            f"Worst-F1={metrics['worst_class_f1']:.2%}"
        )
    print(f"SAVED: {output_dir}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble GAP-HBP OOF dengan alpha dipilih pada validation tiap fold"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument(
        "--objective",
        choices=("macro_f1", "hard_class_f1", "balanced_accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--expected-count", type=int, default=979)
    args = parser.parse_args()
    run_oof_ensemble(
        data_root=args.data_root,
        output_root=args.output_root,
        folds=args.folds,
        seed=args.seed,
        alpha_step=args.alpha_step,
        objective=args.objective,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()
