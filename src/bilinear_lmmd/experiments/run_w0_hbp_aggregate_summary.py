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

# Historical completed HBP-R0 five-fold seed42 aggregate as previously reported.
# Only these three rounded values are available without retraining or the old
# analysis package. They are therefore used as an aggregate exploratory
# reference, not as fold-paired observations.
HBP_R0_REFERENCE = {
    "macro_f1": 0.9158,
    "hard_class_f1": 0.8623,
    "worst_class_f1": 0.6526,
}
REFERENCE_PRECISION_NOTE = (
    "HBP-R0 reference values are rounded historical aggregate values from the "
    "completed five-fold seed42 HBP-MVFD-SBN experiment. Exact per-fold control "
    "records are unavailable here; no paired-fold claims are made."
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
    initial_hashes = set()
    for fold in FOLDS:
        result = _json(
            output_root / "W0_HBP" / f"fold_{fold}" / "seed42" / "result.json"
        )
        if result.get("protocol") != "coffee17-w0-hbp-v1":
            raise RuntimeError(f"Fold {fold}: protocol W0-HBP tidak cocok.")
        if result.get("control_retrained") is not False:
            raise RuntimeError(f"Fold {fold}: R0-HBP tidak boleh retrain.")
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        initial_hashes.add(result["hbp_initial_model_state_sha256"])
        rows[f"fold_{fold}"] = result

    if len(initial_hashes) != 1:
        raise RuntimeError("Initial HBP fingerprint tidak konsisten antar-fold.")

    aggregate = {
        metric: _stats(
            [float(rows[f"fold_{fold}"]["W0_HBP"][metric]) for fold in FOLDS]
        )
        for metric in METRICS
    }

    delta_vs_historical = {
        metric: aggregate[metric]["mean"] - HBP_R0_REFERENCE[metric]
        for metric in HBP_R0_REFERENCE
    }
    gate = {
        "macro_mean_above_historical_hbp_r0": delta_vs_historical["macro_f1"] > 0.0,
        "hard_mean_not_below_historical_hbp_r0": delta_vs_historical["hard_class_f1"] >= 0.0,
        "worst_mean_not_below_historical_hbp_r0": delta_vs_historical["worst_class_f1"] >= 0.0,
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.w0_hbp.aggregate_reference_summary.v1",
        "protocol": "coffee17-w0-hbp-v1",
        "scope": (
            "five-fold seed42 W0-HBP validation-only exploratory run; "
            "R0-HBP was not retrained; aggregate historical HBP-R0 reference only"
        ),
        "folds": rows,
        "W0_HBP_AGGREGATE": aggregate,
        "HBP_R0_HISTORICAL_AGGREGATE_REFERENCE": HBP_R0_REFERENCE,
        "DELTA_W0_HBP_VS_HISTORICAL_HBP_R0": delta_vs_historical,
        "reference_precision_note": REFERENCE_PRECISION_NOTE,
        "screening_gate": gate,
        "control_retrained": False,
        "paired_fold_claim_allowed": False,
        "test_images_accessed": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
