from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FOLDS = (1, 2, 3, 4, 5)
SEED = 42
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(
    *,
    treatment_root: Path,
    control_root: Path,
    output: Path,
) -> dict:
    treatment_root = Path(treatment_root).expanduser().resolve()
    control_root = Path(control_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    rows = []
    reference_clean = None
    reference_fold_manifest = None
    reference_initial_model = None

    for fold in FOLDS:
        treatment_path = (
            treatment_root
            / "treatment"
            / f"fold_{fold}"
            / f"seed{SEED}"
            / "result.json"
        )
        control_path = (
            control_root
            / "primary"
            / "R0"
            / f"fold_{fold}"
            / f"seed{SEED}"
            / "result.json"
        )

        treatment = _json(treatment_path, f"W0-RAS fold {fold}")
        control = _json(control_path, f"R0 control fold {fold}")

        if treatment.get("protocol") != "coffee17-w0-ras-v1":
            raise RuntimeError(f"Fold {fold}: protocol treatment tidak cocok.")
        if control.get("arm") != "R0":
            raise RuntimeError(f"Fold {fold}: control bukan R0.")
        if treatment.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: treatment mengakses outer test.")
        if control.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: control mengakses outer test.")

        tc = treatment["run_contract"]
        cc = control["run_contract"]

        if tc["clean_content_sha256"] != cc["clean_content_sha256"]:
            raise RuntimeError(f"Fold {fold}: clean population mismatch.")
        if tc["fold_manifest_sha256"] != cc["fold_manifest_sha256"]:
            raise RuntimeError(f"Fold {fold}: fold manifest mismatch.")
        if (
            tc["common_initialized_model_state_sha256"]
            != cc["common_initialized_model_state_sha256"]
        ):
            raise RuntimeError(f"Fold {fold}: initial model mismatch.")

        if reference_clean is None:
            reference_clean = tc["clean_content_sha256"]
            reference_fold_manifest = tc["fold_manifest_sha256"]
            reference_initial_model = tc[
                "common_initialized_model_state_sha256"
            ]
        elif (
            reference_clean != tc["clean_content_sha256"]
            or reference_fold_manifest != tc["fold_manifest_sha256"]
            or reference_initial_model
            != tc["common_initialized_model_state_sha256"]
        ):
            raise RuntimeError("Kontrak lintas-fold treatment tidak konsisten.")

        row = {
            "fold": fold,
            "epoch1_weighted_aux_to_ce_ratio": float(
                treatment["epoch1_weighted_aux_to_ce_ratio"]
            ),
        }
        for metric in METRICS:
            t = float(treatment["metrics"][metric])
            c = float(control["metrics"][metric])
            row[f"treatment_{metric}"] = t
            row[f"control_{metric}"] = c
            row[f"delta_{metric}"] = t - c
        rows.append(row)

    means = {}
    for metric in METRICS:
        means[f"treatment_{metric}"] = _mean(
            [row[f"treatment_{metric}"] for row in rows]
        )
        means[f"control_{metric}"] = _mean(
            [row[f"control_{metric}"] for row in rows]
        )
        means[f"delta_{metric}"] = _mean(
            [row[f"delta_{metric}"] for row in rows]
        )

    macro_positive_folds = sum(
        row["delta_macro_f1"] > 0.0 for row in rows
    )
    loss_scale_pass = all(
        row["epoch1_weighted_aux_to_ce_ratio"] <= 1.0 for row in rows
    )

    criteria = {
        "mean_macro_delta_gt_0": means["delta_macro_f1"] > 0.0,
        "macro_positive_at_least_3_of_5": macro_positive_folds >= 3,
        "mean_hard_delta_ge_0": means["delta_hard_class_f1"] >= 0.0,
        "mean_worst_delta_ge_0": means["delta_worst_class_f1"] >= 0.0,
        "loss_scale_gate_pass_all_folds": loss_scale_pass,
    }
    decision = (
        "PROMOTE"
        if all(criteria.values())
        else "STOP_W0_RAS_V1"
    )

    result = {
        "format": "bilinear_lmmd.w0_ras.aggregate.v1",
        "protocol": "coffee17-w0-ras-v1",
        "population": "five_development_folds_seed42",
        "outer_test_oof_accessed": False,
        "rows": rows,
        "means": means,
        "macro_positive_folds": macro_positive_folds,
        "criteria": criteria,
        "decision": decision,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate(
        treatment_root=args.treatment_root,
        control_root=args.control_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
