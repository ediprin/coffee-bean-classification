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
            output_root
            / "W0_RAS"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("protocol") != "coffee17-w0-ras-v1":
            raise RuntimeError(f"Fold {fold}: protocol W0-RAS tidak cocok.")
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: W0-RAS menyentuh test.")
        rows[f"fold_{fold}"] = result

    aggregate = {
        "R0_CONTROL": {
            metric: _stats(
                [
                    float(
                        rows[f"fold_{fold}"]["frozen_matched_r0_control"][metric]
                    )
                    for fold in FOLDS
                ]
            )
            for metric in METRICS
        },
        "W0_RAS": {
            metric: _stats(
                [
                    float(rows[f"fold_{fold}"]["metrics"][metric])
                    for fold in FOLDS
                ]
            )
            for metric in METRICS
        },
        "DELTA": {
            metric: _stats(
                [
                    float(rows[f"fold_{fold}"]["delta_vs_r0_control"][metric])
                    for fold in FOLDS
                ]
            )
            for metric in METRICS
        },
        "EPOCH1_WEIGHTED_AUX_TO_CE_RATIO": _stats(
            [
                float(
                    rows[f"fold_{fold}"]["epoch1_weighted_aux_to_ce_ratio"]
                )
                for fold in FOLDS
            ]
        ),
    }

    delta = aggregate["DELTA"]
    macro_positive = sum(
        rows[f"fold_{fold}"]["delta_vs_r0_control"]["macro_f1"] > 0.0
        for fold in FOLDS
    )
    hard_positive = sum(
        rows[f"fold_{fold}"]["delta_vs_r0_control"]["hard_class_f1"] > 0.0
        for fold in FOLDS
    )

    gate = {
        "macro_mean_positive": delta["macro_f1"]["mean"] > 0.0,
        "macro_positive_folds_at_least_3": macro_positive >= 3,
        "hard_mean_nonnegative": delta["hard_class_f1"]["mean"] >= 0.0,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
        "loss_scale_pass_all_folds": all(
            rows[f"fold_{fold}"]["epoch1_weighted_aux_to_ce_ratio"] <= 1.0
            for fold in FOLDS
        ),
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.w0_ras.summary.v1",
        "protocol": "coffee17-w0-ras-v1",
        "scope": (
            "five-fold validation-only post-OOF exploratory selective W0 "
            "retention supervision; frozen fold-matched R0 reference; "
            "no outer-test inference"
        ),
        "primary_comparison": "W0_RAS vs frozen matched R0_CONTROL",
        "folds": rows,
        "aggregate": aggregate,
        "positive_macro_folds": int(macro_positive),
        "positive_hard_folds": int(hard_positive),
        "screening_gate": gate,
        "test_images_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(output_root=args.output_root, output=args.output)


if __name__ == "__main__":
    main()
