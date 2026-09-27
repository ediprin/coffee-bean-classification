from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from bilinear_lmmd.analysis.nuisance import NUISANCE_KINDS
from bilinear_lmmd.engine.at_sbn import VIEWS


FOLDS = (1, 2, 3, 4, 5)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values):
    values = [float(v) for v in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def run_summary(*, input_dir: Path, output: Path) -> dict:
    input_dir = Path(input_dir).expanduser().resolve()
    folds = {
        fold: _load(input_dir / f"fold_{fold}.json")
        for fold in FOLDS
    }
    for fold, payload in folds.items():
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold} menyentuh test.")

    clean = {
        arm: {
            metric: _stats([
                folds[f]["clean"][arm][metric] for f in FOLDS
            ])
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "fisher_separability",
            )
        }
        for arm in VIEWS
    }

    nuisances = {}
    for nuisance in NUISANCE_KINDS:
        nuisances[nuisance] = {}
        for arm in VIEWS:
            nuisances[nuisance][arm] = {
                metric: _stats([
                    folds[f]["nuisances"][nuisance][arm][metric]
                    for f in FOLDS
                ])
                for metric in (
                    "cosine_stability",
                    "normalized_l2_shift",
                    "fisher_separability",
                    "fisher_retention",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "macro_f1_drop",
                    "balanced_accuracy_drop",
                )
            }

    vs_r0 = {}
    for nuisance in NUISANCE_KINDS:
        vs_r0[nuisance] = {}
        for arm in ("C0", "F0", "W0"):
            per_fold = []
            for fold in FOLDS:
                a = folds[fold]["nuisances"][nuisance][arm]
                r = folds[fold]["nuisances"][nuisance]["R0"]
                per_fold.append({
                    "cosine_stability_delta": (
                        a["cosine_stability"] - r["cosine_stability"]
                    ),
                    "normalized_l2_shift_delta": (
                        a["normalized_l2_shift"] - r["normalized_l2_shift"]
                    ),
                    "fisher_retention_delta": (
                        a["fisher_retention"] - r["fisher_retention"]
                    ),
                    "macro_f1_drop_delta": (
                        a["macro_f1_drop"] - r["macro_f1_drop"]
                    ),
                })
            vs_r0[nuisance][arm] = {
                key: _stats([row[key] for row in per_fold])
                for key in per_fold[0]
            }

    robustness_counts = {}
    for arm in ("C0", "F0", "W0"):
        stronger = 0
        details = {}
        for nuisance in NUISANCE_KINDS:
            d = vs_r0[nuisance][arm]
            stable = d["cosine_stability_delta"]["mean"] > 0.0
            separation = d["fisher_retention_delta"]["mean"] >= 0.0
            classifier = d["macro_f1_drop_delta"]["mean"] >= 0.0
            qualifies = stable and (separation or classifier)
            details[nuisance] = {
                "more_stable_than_r0": stable,
                "separability_retention_not_worse": separation,
                "macro_f1_retention_not_worse": classifier,
                "qualifies_descriptively": qualifies,
            }
            stronger += int(qualifies)
        robustness_counts[arm] = {
            "qualified_nuisances": stronger,
            "out_of": len(NUISANCE_KINDS),
            "details": details,
        }

    payload = {
        "format": "coffee17.preprocessing_nuisance_analysis.summary.v1",
        "protocol": "coffee17-preprocessing-nuisance-analysis-v1",
        "scope": (
            "five-fold overlapping development-validation observations; "
            "descriptive robustness diagnostic only"
        ),
        "clean": clean,
        "nuisances": nuisances,
        "delta_vs_r0": vs_r0,
        "descriptive_robustness_count": robustness_counts,
        "uniform_claim_prohibited": (
            "Synthetic nuisance robustness does not establish physical camera "
            "or illumination invariance."
        ),
        "test_images_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(input_dir=args.input_dir, output=args.output)


if __name__ == "__main__":
    main()
