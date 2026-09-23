from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)
MODES = ("R0", "ALL4")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "worst_class_f1",
    "hard_class_f1",
)


def _json(path: Path, label: str) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pick_metrics(payload: dict) -> dict:
    return {name: float(payload[name]) for name in METRICS}


def run_summary(
    *,
    experiments_root: Path,
    kd_root: Path,
    output: Path,
) -> dict:
    experiments_root = Path(experiments_root).expanduser().resolve()
    kd_root = Path(kd_root).expanduser().resolve()

    folds = {}
    for fold in FOLDS:
        baseline = _pick_metrics(
            _json(
                experiments_root
                / "primary"
                / "R0"
                / f"fold_{fold}"
                / "seed42"
                / "validation"
                / "metrics.json",
                f"R0 baseline fold {fold}",
            )
        )
        kd = {}
        for mode in MODES:
            kd[mode] = _pick_metrics(
                _json(
                    kd_root
                    / mode
                    / f"fold_{fold}"
                    / "seed42"
                    / "validation"
                    / "metrics.json",
                    f"KD {mode} fold {fold}",
                )
            )
        folds[f"fold_{fold}"] = {
            "R0_BASELINE": baseline,
            "KD_R0": kd["R0"],
            "KD_ALL4": kd["ALL4"],
            "delta_KD_R0_vs_R0": {
                name: kd["R0"][name] - baseline[name] for name in METRICS
            },
            "delta_KD_ALL4_vs_R0": {
                name: kd["ALL4"][name] - baseline[name] for name in METRICS
            },
            "delta_KD_ALL4_vs_KD_R0": {
                name: kd["ALL4"][name] - kd["R0"][name] for name in METRICS
            },
        }

    def mean_for(label: str, metric: str) -> float:
        return statistics.mean(
            folds[f"fold_{fold}"][label][metric] for fold in FOLDS
        )

    aggregate = {
        label: {metric: mean_for(label, metric) for metric in METRICS}
        for label in ("R0_BASELINE", "KD_R0", "KD_ALL4")
    }
    aggregate["delta_KD_R0_vs_R0"] = {
        metric: aggregate["KD_R0"][metric] - aggregate["R0_BASELINE"][metric]
        for metric in METRICS
    }
    aggregate["delta_KD_ALL4_vs_R0"] = {
        metric: aggregate["KD_ALL4"][metric] - aggregate["R0_BASELINE"][metric]
        for metric in METRICS
    }
    aggregate["delta_KD_ALL4_vs_KD_R0"] = {
        metric: aggregate["KD_ALL4"][metric] - aggregate["KD_R0"][metric]
        for metric in METRICS
    }

    payload = {
        "format": "bilinear_lmmd.preprocessing.kd_validation_summary.v1",
        "protocol": "coffee17-preprocessing-kd-exploratory-v1",
        "scope": (
            "descriptive fold-validation comparison only; validation folds are "
            "not an independent confirmatory test population"
        ),
        "folds": folds,
        "fold_mean": aggregate,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--kd-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(
        experiments_root=args.experiments_root,
        kd_root=args.kd_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
