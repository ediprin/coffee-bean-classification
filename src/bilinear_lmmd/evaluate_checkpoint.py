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
    batches = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    for images, targets in batches:
        logits = model(images.to(device)).logits
        labels.extend(targets.tolist())
        probabilities.append(logits.softmax(1).cpu())

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
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    domain: str,
    split: str,
    output_dir: Path,
    data_root: Path | None = None,
) -> None:
    bundle = collect_checkpoint_predictions(
        checkpoint_path,
        domain,
        split,
        data_root=data_root,
        progress_desc=f"evaluate {domain}/{split}",
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
        }
    )
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
        writer.writerow(
            ["path", "actual", "predicted", "correct", *probability_columns]
        )
        for path, actual, predicted, probabilities in zip(
            bundle.paths, bundle.labels, predictions, bundle.probabilities.tolist()
        ):
            writer.writerow(
                [
                    path,
                    bundle.classes[actual],
                    bundle.classes[predicted],
                    int(actual == predicted),
                    *probabilities,
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
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.domain,
        args.split,
        args.output_dir,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
