from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix

from .data import build_loaders
from .models import build_model
from .train import classification_metrics, resolve_device


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    domain: str,
    split: str,
    output_dir: Path,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    cfg["model"]["pretrained"] = False
    cfg["data"]["val_split"] = split
    device = resolve_device(cfg["device"])
    loaders = build_loaders(cfg["data"], require_target=domain == "target")
    loader = loaders.source_val if domain == "source" else loaders.target_val
    if loader is None:
        raise FileNotFoundError(f"Loader {domain}/{split} tidak tersedia.")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    for images, targets in loader:
        logits = model(images.to(device)).logits
        labels.extend(targets.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())

    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})
    metrics = classification_metrics(labels, predictions, loaders.classes, hard_groups)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "domain": domain,
            "split": split,
            "classes": loaders.classes,
        }
    )
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(loaders.classes)))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *loaders.classes])
        for name, row in zip(loaders.classes, matrix.tolist()):
            writer.writerow([name, *row])
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "actual", "predicted", "correct"])
        paths = [sample[0] for sample in loader.dataset.samples]
        for path, actual, predicted in zip(paths, labels, predictions):
            writer.writerow(
                [
                    path,
                    loaders.classes[actual],
                    loaders.classes[predicted],
                    int(actual == predicted),
                ]
            )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi checkpoint per kelas")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--domain", choices=("source", "target"), default="source")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.domain, args.split, args.output_dir)


if __name__ == "__main__":
    main()
