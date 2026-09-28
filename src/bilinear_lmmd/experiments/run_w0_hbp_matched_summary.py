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
    init_hashes = set()
    val_hashes = set()

    for fold in FOLDS:
        result = _json(
            output_root / f"fold_{fold}" / "seed42" / "pair_result.json"
        )
        if result.get("protocol") != "coffee17-w0-hbp-matched-v1":
            raise RuntimeError(f"Fold {fold}: protocol tidak cocok.")
        if result.get("matched_initialization") is not True:
            raise RuntimeError(f"Fold {fold}: initialization tidak matched.")
        if result.get("matched_validation_rows") is not True:
            raise RuntimeError(f"Fold {fold}: validation rows tidak matched.")
        if result.get("r0_control_retrained") is not True:
            raise RuntimeError(f"Fold {fold}: R0-HBP matched control tidak diretrain.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")

        init_hashes.add(result["initial_model_state_sha256"])
        val_hashes.add(result["validation_identity_label_sha256"])
        rows[f"fold_{fold}"] = result

    if len(init_hashes) != 1:
        raise RuntimeError("Initial model fingerprint berbeda antar-fold.")
    if len(val_hashes) != 5:
        raise RuntimeError("Validation fold hash tidak unik 5/5.")

    aggregate = {}
    for label in ("R0_HBP", "W0_HBP", "DELTA_W0_MINUS_R0"):
        aggregate[label] = {
            metric: _stats(
                [
                    float(rows[f"fold_{fold}"][label][metric])
                    for fold in FOLDS
                ]
            )
            for metric in METRICS
        }

    positive = {
        metric: sum(
            rows[f"fold_{fold}"]["DELTA_W0_MINUS_R0"][metric] > 0.0
            for fold in FOLDS
        )
        for metric in METRICS
    }

    delta = aggregate["DELTA_W0_MINUS_R0"]
    gate = {
        "macro_mean_positive": delta["macro_f1"]["mean"] > 0.0,
        "macro_positive_folds_at_least_3": positive["macro_f1"] >= 3,
        "hard_mean_nonnegative": delta["hard_class_f1"]["mean"] >= 0.0,
        "worst_mean_nonnegative": delta["worst_class_f1"]["mean"] >= 0.0,
    }
    gate["decision"] = "PASS" if all(gate.values()) else "FAIL"

    payload = {
        "format": "bilinear_lmmd.w0_hbp_matched.summary.v1",
        "protocol": "coffee17-w0-hbp-matched-v1",
        "scope": (
            "five-fold seed42 matched validation comparison; both R0-HBP and "
            "W0-HBP retrained from the same initialization and training recipe; "
            "only preprocessing differs; no outer-test inference"
        ),
        "folds": rows,
        "aggregate": aggregate,
        "positive_delta_folds": positive,
        "screening_gate": gate,
        "paired_fold_comparison": True,
        "r0_control_retrained": True,
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary(output_root=args.output_root, output=args.output)


if __name__ == "__main__":
    main()
