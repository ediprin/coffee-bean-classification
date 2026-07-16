from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .data import build_loaders
from .models import build_model
from .train import classification_metrics, resolve_device


@dataclass(frozen=True)
class CheckpointPredictions:
    classes: list[str]
    labels: list[int]
    probabilities: torch.Tensor
    paths: list[str]
    hard_groups: dict[str, list[str]]
    gate_weights: torch.Tensor | None = None
    gate_names: tuple[str, ...] | None = None
    prediction_head: str = "fused"

    @property
    def predictions(self) -> list[int]:
        return self.probabilities.argmax(1).tolist()


@torch.no_grad()
def collect_checkpoint_predictions(
    checkpoint_path: Path,
    domain: str,
    split: str,
    data_root: Path | None = None,
    progress_desc: str | None = None,
    prediction_head: str = "fused",
) -> CheckpointPredictions:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["model"]["pretrained"] = False
    cfg["data"]["val_split"] = split
    if data_root is not None:
        cfg["data"]["root"] = str(data_root)
    device = resolve_device(cfg["device"])
    loaders = build_loaders(cfg["data"], require_target=domain == "target")
    loader = loaders.source_val if domain == "source" else loaders.target_val
    if loader is None:
        raise FileNotFoundError(f"Loader {domain}/{split} tidak tersedia.")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    labels: list[int] = []
    probabilities: list[torch.Tensor] = []
    gate_weights: list[torch.Tensor] = []
    gate_names: tuple[str, ...] | None = None
    batches = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    for images, targets in batches:
        output = model(images.to(device))
        if prediction_head == "fused":
            logits = output.logits
        else:
            if output.expert_logits is None or prediction_head not in output.expert_logits:
                available = tuple(output.expert_logits or {})
                raise ValueError(
                    f"prediction_head={prediction_head!r} tidak tersedia; "
                    f"expert={available}."
                )
            logits = output.expert_logits[prediction_head]
        labels.extend(targets.tolist())
        probabilities.append(logits.softmax(1).cpu())
        if output.gate_weights is not None:
            gate_weights.append(output.gate_weights.cpu())
            current_names = tuple(output.expert_logits or {})
            if gate_names is None:
                gate_names = current_names
            elif gate_names != current_names:
                raise RuntimeError("Urutan expert berubah antar-batch.")

    paths = [sample[0] for sample in loader.dataset.samples]
    if len(paths) != len(labels):
        raise RuntimeError(
            f"Jumlah path ({len(paths)}) berbeda dari prediksi ({len(labels)})."
        )
    return CheckpointPredictions(
        classes=loaders.classes,
        labels=labels,
        probabilities=torch.cat(probabilities),
        paths=paths,
        hard_groups=cfg.get("evaluation", {}).get("hard_groups", {}),
        gate_weights=(torch.cat(gate_weights) if gate_weights else None),
        gate_names=gate_names,
        prediction_head=prediction_head,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    domain: str,
    split: str,
    output_dir: Path,
    data_root: Path | None = None,
    prediction_head: str = "fused",
) -> None:
    bundle = collect_checkpoint_predictions(
        checkpoint_path,
        domain,
        split,
        data_root=data_root,
        progress_desc=f"evaluate {domain}/{split}",
        prediction_head=prediction_head,
    )
    predictions = bundle.predictions

    metrics = classification_metrics(
        bundle.labels, predictions, bundle.classes, bundle.hard_groups
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "domain": domain,
            "split": split,
            "classes": bundle.classes,
            "prediction_head": bundle.prediction_head,
        }
    )
    if bundle.gate_weights is not None:
        weights = bundle.gate_weights
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        selected = weights.argmax(dim=1)
        names = bundle.gate_names or tuple(f"expert_{i}" for i in range(weights.shape[1]))
        metrics["gate"] = {
            **{
                f"mean_{name}": float(weights[:, index].mean())
                for index, name in enumerate(names)
            },
            **{
                f"{name}_selected_fraction": float((selected == index).float().mean())
                for index, name in enumerate(names)
            },
            "mean_entropy": float(entropy.mean()),
        }
    matrix = confusion_matrix(
        bundle.labels, predictions, labels=list(range(len(bundle.classes)))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *bundle.classes])
        for name, row in zip(bundle.classes, matrix.tolist()):
            writer.writerow([name, *row])
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        probability_columns = [f"prob::{name}" for name in bundle.classes]
        gate_columns = (
            [f"gate::{name}" for name in (bundle.gate_names or ())]
            if bundle.gate_weights is not None
            else []
        )
        writer.writerow([
            "path", "actual", "predicted", "correct",
            *probability_columns, *gate_columns,
        ])
        gates = (
            bundle.gate_weights.tolist()
            if bundle.gate_weights is not None
            else [[] for _ in bundle.labels]
        )
        for path, actual, predicted, probabilities, gate in zip(
            bundle.paths,
            bundle.labels,
            predictions,
            bundle.probabilities.tolist(),
            gates,
        ):
            writer.writerow(
                [
                    path,
                    bundle.classes[actual],
                    bundle.classes[predicted],
                    int(actual == predicted),
                    *probabilities,
                    *gate,
                ]
            )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi checkpoint per kelas")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--domain", choices=("source", "target"), default="source")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override data.root checkpoint, berguna jika folder dipindahkan.",
    )
    parser.add_argument(
        "--prediction-head",
        default="fused",
        help="Gunakan fused atau nama expert seperti gap/hbp.",
    )
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.domain,
        args.split,
        args.output_dir,
        data_root=args.data_root,
        prediction_head=args.prediction_head,
    )


if __name__ == "__main__":
    main()
