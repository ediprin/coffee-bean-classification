from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)
KEYS = (
    "zero_angle_disagreement",
    "zero_angle_mean_pairwise_js",
    "any_disagreement_in_7_angles",
    "mean_disagreement_angle_count",
    "mean_max_pairwise_js",
    "scheduled_disagreement_exposure",
    "scheduled_mean_pairwise_js",
    "scheduled_images_with_at_least_one_disagreement_epoch",
    "mean_disagreement_epochs_per_image",
)


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def run_summary(*, audit_root: Path, output: Path) -> dict:
    audit_root = Path(audit_root).expanduser().resolve()
    rows = {}
    class_values: dict[str, dict[str, list[float]]] = {}

    for fold in FOLDS:
        path = audit_root / f"fold_{fold}" / "rotation_audit.json"
        payload = _json(path)
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} audit menyentuh test.")
        if int(payload.get("fold", -1)) != fold:
            raise RuntimeError(f"Fold mismatch pada {path}.")

        across = payload["across_angles_and_schedule"]
        rows[f"fold_{fold}"] = {key: float(across[key]) for key in KEYS}
        rows[f"fold_{fold}"]["samples"] = int(across["samples"])

        for class_name, class_row in payload.get("per_class", {}).items():
            bucket = class_values.setdefault(
                class_name,
                {
                    "any_disagreement_in_7_angles": [],
                    "mean_disagreement_angle_count": [],
                    "mean_max_pairwise_js": [],
                    "scheduled_disagreement_exposure": [],
                    "scheduled_mean_pairwise_js": [],
                },
            )
            for key in bucket:
                bucket[key].append(float(class_row[key]))

    fold_mean = {
        key: _stats([rows[f"fold_{fold}"][key] for fold in FOLDS])
        for key in KEYS
    }
    per_class_fold_mean = {
        class_name: {key: _stats(values) for key, values in metrics.items()}
        for class_name, metrics in sorted(class_values.items())
    }

    payload = {
        "format": "bilinear_lmmd.preprocessing.kd_rotation_audit_summary.v1",
        "protocol": "coffee17-preprocessing-kd-exploratory-v1",
        "scope": (
            "descriptive fold summary; training populations overlap across folds "
            "and these fold means are not independent replicates"
        ),
        "folds": rows,
        "fold_mean": fold_mean,
        "per_class_fold_mean": per_class_fold_mean,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(fold_mean, indent=2), flush=True)
    print(f"SAVED: {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(audit_root=args.audit_root, output=args.output)


if __name__ == "__main__":
    main()
