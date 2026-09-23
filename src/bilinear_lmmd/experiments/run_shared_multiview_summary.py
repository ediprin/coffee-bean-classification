from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "worst_class_f1",
    "hard_class_f1",
)


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def run_summary(*, output_root: Path, output: Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    rows = {}
    for fold in FOLDS:
        result = _json(
            output_root / "MVCE_ALL4" / f"fold_{fold}" / "seed42" / "result.json"
        )
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")
        rows[f"fold_{fold}"] = {
            "R0_BASELINE": result["r0_baseline"],
            "MVCE_ALL4": result["metrics"],
            "DELTA": result["delta_vs_r0"],
        }

    aggregate = {
        "R0_BASELINE": {
            metric: _stats(
                [rows[f"fold_{fold}"]["R0_BASELINE"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "MVCE_ALL4": {
            metric: _stats(
                [rows[f"fold_{fold}"]["MVCE_ALL4"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
        "DELTA": {
            metric: _stats(
                [rows[f"fold_{fold}"]["DELTA"][metric] for fold in FOLDS]
            )
            for metric in METRICS
        },
    }
    payload = {
        "format": "bilinear_lmmd.shared_multiview.summary.v1",
        "protocol": "coffee17-shared-multiview-mvce-v1",
        "scope": (
            "descriptive five-fold validation comparison against the frozen "
            "historical R0 validation baseline; no outer-test inference"
        ),
        "folds": rows,
        "aggregate": aggregate,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"SAVED: {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(output_root=args.output_root, output=args.output)


if __name__ == "__main__":
    main()
