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
            / "W0_HBP"
            / f"fold_{fold}"
            / "seed42"
            / "result.json"
        )
        if result.get("protocol") != "coffee17-w0-hbp-v1":
            raise RuntimeError(f"Fold {fold}: protocol W0-HBP tidak cocok.")
        if result.get("control_retrained") is not False:
            raise RuntimeError(f"Fold {fold}: HBP-R0 control tidak boleh retrain.")
        if result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        rows[f"fold_{fold}"] = result

    aggregate = {}
    for label in ("HBP_R0_CONTROL", "W0_HBP", "DELTA_W0_ON_HBP"):
        aggregate[label] = {
            metric: _stats(
                [
                    float(rows[f"fold_{fold}"][label][metric])
                    for fold in FOLDS
                ]
            )
            for metric in METRICS
        }

    delta = aggregate["DELTA_W0_ON_HBP"]
    positive_macro = sum(
        rows[f"fold_{fold}"]["DELTA_W0_ON_HBP"]["macro_f1"] > 0.0
        for fold in FOLDS
    )
    positive_hard = sum(
        rows[f"fold_{fold}"]["DELTA_W0_ON_HBP"]["hard_class_f1"] > 0.0
        for fold in FOLDS
    )
    positive_worst = sum(
        rows[f"fold_{fold}"]["DELTA_W0_ON_HBP"]["worst_class_f1"] > 0.0
        for fold in FOLDS
    )

    gate = {
        "macro_mean_positive": delta["macro_f1"]["mean"] > 0.0,
        "macro_positive_folds_at_least_3": positive_macro >= 3,
        "hard_mean_nonnegative": delta["hard_class_f1"]["mean"] >= 0.0,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.w0_hbp.summary.v1",
        "protocol": "coffee17-w0-hbp-v1",
        "scope": (
            "five-fold validation-only post-primary exploratory W0 preprocessing "
            "plus HBP; completed fold-matched HBP-R0 controls reused without "
            "retraining; no outer-test inference"
        ),
        "primary_comparison": "W0_HBP vs completed HBP_R0_CONTROL",
        "folds": rows,
        "aggregate": aggregate,
        "positive_macro_folds": int(positive_macro),
        "positive_hard_folds": int(positive_hard),
        "positive_worst_folds": int(positive_worst),
        "screening_gate": gate,
        "control_retrained": False,
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
