from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.engine.wavelet_clahe_residual_hbp import (
    build_wr_control,
    build_wrc_candidate,
)


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


def _parameter_counts(config_path: Path) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg["model"]["pretrained"] = False
    control = build_wr_control(cfg)
    candidate = build_wrc_candidate(cfg)
    control_n = sum(p.numel() for p in control.parameters())
    candidate_n = sum(p.numel() for p in candidate.parameters())
    return {
        "WR_HBP": control_n,
        "WRC_HBP": candidate_n,
        "candidate_overhead": candidate_n - control_n,
        "candidate_overhead_percent": 100.0 * (candidate_n - control_n) / control_n,
    }


def run_summary(*, output_root: Path, config_path: Path, output: Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    rows: dict[str, dict] = {}
    shared_hashes = set()
    validation_hashes = set()
    classes: list[str] | None = None

    paired_totals = {
        "count": 0,
        "rescue": 0,
        "damage": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net_correct": 0,
    }
    per_class_deltas: dict[str, list[float]] = {}
    hard_group_deltas: dict[str, list[float]] = {}

    for fold in FOLDS:
        pair_dir = output_root / f"fold_{fold}" / "seed42"
        result = _json(pair_dir / "pair_result.json")

        if result.get("protocol") != "coffee17-wavelet-clahe-residual-hbp-v1":
            raise RuntimeError(f"Fold {fold}: protocol tidak cocok.")
        if result.get("matched_shared_wr_initialization") is not True:
            raise RuntimeError(f"Fold {fold}: shared WR init tidak matched.")
        if result.get("matched_validation_rows") is not True:
            raise RuntimeError(f"Fold {fold}: validation rows tidak matched.")
        if result.get("wr_control_retrained") is not True:
            raise RuntimeError(f"Fold {fold}: WR control tidak diretrain.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")

        shared_hashes.add(result["shared_wr_initial_sha256"])
        validation_hashes.add(result["validation_identity_label_sha256"])
        rows[f"fold_{fold}"] = result

        for key in paired_totals:
            paired_totals[key] += int(result["paired_outcomes"][key])

        control_metrics = _json(pair_dir / "WR_HBP/validation/metrics.json")
        candidate_metrics = _json(pair_dir / "WRC_HBP/validation/metrics.json")
        if control_metrics["classes"] != candidate_metrics["classes"]:
            raise RuntimeError(f"Fold {fold}: class order berbeda.")

        if classes is None:
            classes = list(control_metrics["classes"])
        elif classes != list(control_metrics["classes"]):
            raise RuntimeError(f"Fold {fold}: class order lintas fold berbeda.")

        for class_name in classes:
            delta = (
                float(candidate_metrics["per_class"][class_name]["f1"])
                - float(control_metrics["per_class"][class_name]["f1"])
            )
            per_class_deltas.setdefault(class_name, []).append(delta)

        for group_name in control_metrics.get("hard_groups", {}):
            delta = (
                float(candidate_metrics["hard_groups"][group_name])
                - float(control_metrics["hard_groups"][group_name])
            )
            hard_group_deltas.setdefault(group_name, []).append(delta)

    if len(shared_hashes) != 1:
        raise RuntimeError("Shared WR fingerprint berbeda antar-fold.")
    if len(validation_hashes) != 5:
        raise RuntimeError("Validation fold hash tidak unik 5/5.")

    aggregate = {}
    for label in ("WR_HBP", "WRC_HBP", "DELTA_WRC_MINUS_WR"):
        aggregate[label] = {
            metric: _stats([
                float(rows[f"fold_{fold}"][label][metric])
                for fold in FOLDS
            ])
            for metric in METRICS
        }

    positive = {
        metric: sum(
            rows[f"fold_{fold}"]["DELTA_WRC_MINUS_WR"][metric] > 0.0
            for fold in FOLDS
        )
        for metric in METRICS
    }

    delta = aggregate["DELTA_WRC_MINUS_WR"]
    gate = {
        "macro_mean_positive": delta["macro_f1"]["mean"] > 0.0,
        "macro_positive_folds_at_least_3": positive["macro_f1"] >= 3,
        "hard_mean_nonnegative": delta["hard_class_f1"]["mean"] >= 0.0,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    wavelet_control = [
        float(rows[f"fold_{fold}"]["wavelet_gate_at_best"]["WR_HBP"])
        for fold in FOLDS
    ]
    wavelet_candidate = [
        float(rows[f"fold_{fold}"]["wavelet_gate_at_best"]["WRC_HBP"])
        for fold in FOLDS
    ]
    contrast_gates = [
        float(rows[f"fold_{fold}"]["contrast_gate_at_best"])
        for fold in FOLDS
    ]

    payload = {
        "format": "bilinear_lmmd.wavelet_clahe_residual_hbp.summary.v1",
        "protocol": "coffee17-wavelet-clahe-residual-hbp-v1",
        "scope": (
            "five-fold seed42 matched validation comparison; WR-HBP is retrained "
            "as control; WRC-HBP shares identical initial WR-HBP state and adds "
            "only the frozen zero-gated C0-derived luminance contrast residual; "
            "outer test untouched"
        ),
        "folds": rows,
        "aggregate": aggregate,
        "positive_delta_folds": positive,
        "paired_prediction_outcomes": paired_totals,
        "per_class_delta": {
            name: _stats(values)
            for name, values in per_class_deltas.items()
        },
        "hard_group_delta": {
            name: _stats(values)
            for name, values in hard_group_deltas.items()
        },
        "wavelet_gate_at_best": {
            "WR_HBP": _stats(wavelet_control),
            "WRC_HBP": _stats(wavelet_candidate),
        },
        "contrast_gate_at_best": _stats(contrast_gates),
        "parameter_counts": _parameter_counts(config_path),
        "screening_gate": gate,
        "paired_fold_comparison": True,
        "outer_test_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(
        output_root=args.output_root,
        config_path=args.config,
        output=args.output,
    )


if __name__ == "__main__":
    main()
