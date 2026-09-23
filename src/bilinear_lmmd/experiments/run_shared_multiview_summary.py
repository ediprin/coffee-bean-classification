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
        control = _json(
            output_root / "R0_CONTROL" / f"fold_{fold}" / "seed42" / "result.json"
        )
        mvce = _json(
            output_root / "MVCE_ALL4" / f"fold_{fold}" / "seed42" / "result.json"
        )
        if control.get("test_images_accessed") is not False:
            raise RuntimeError(f"R0 control fold {fold} menyentuh test.")
        if mvce.get("test_images_accessed") is not False:
            raise RuntimeError(f"MVCE fold {fold} menyentuh test.")

        control_initial = control["run_contract"]["current_initial_model_state_sha256"]
        mvce_initial = mvce["run_contract"]["current_initial_model_state_sha256"]
        if control_initial != mvce_initial:
            raise RuntimeError(
                f"Initial model R0/MVCE berbeda pada fold {fold}: "
                f"{control_initial} != {mvce_initial}"
            )

        matched_delta = {
            metric: mvce["metrics"][metric] - control["metrics"][metric]
            for metric in METRICS
        }
        historical = control["historical_r0_baseline"]
        historical_delta_control = {
            metric: control["metrics"][metric] - historical[metric]
            for metric in METRICS
        }
        historical_delta_mvce = {
            metric: mvce["metrics"][metric] - historical[metric]
            for metric in METRICS
        }

        rows[f"fold_{fold}"] = {
            "HISTORICAL_R0": historical,
            "R0_CONTROL": control["metrics"],
            "MVCE_ALL4": mvce["metrics"],
            "MVCE_MINUS_MATCHED_R0": matched_delta,
            "R0_CONTROL_MINUS_HISTORICAL_R0": historical_delta_control,
            "MVCE_MINUS_HISTORICAL_R0": historical_delta_mvce,
            "current_initial_model_state_sha256": control_initial,
            "historical_initial_match": bool(
                control["run_contract"]["historical_initial_match"]
                and mvce["run_contract"]["historical_initial_match"]
            ),
        }

    aggregate = {}
    for label in (
        "HISTORICAL_R0",
        "R0_CONTROL",
        "MVCE_ALL4",
        "MVCE_MINUS_MATCHED_R0",
        "R0_CONTROL_MINUS_HISTORICAL_R0",
        "MVCE_MINUS_HISTORICAL_R0",
    ):
        aggregate[label] = {
            metric: _stats(
                [rows[f"fold_{fold}"][label][metric] for fold in FOLDS]
            )
            for metric in METRICS
        }

    payload = {
        "format": "bilinear_lmmd.shared_multiview.summary.v2",
        "protocol": "coffee17-shared-multiview-mvce-v1",
        "scope": (
            "matched same-environment five-fold validation comparison; "
            "historical R0 retained only as a secondary reproducibility reference; "
            "no outer-test inference"
        ),
        "primary_comparison": "MVCE_ALL4 vs R0_CONTROL",
        "folds": rows,
        "aggregate": aggregate,
        "all_matched_initializations": len(
            {rows[f"fold_{fold}"]["current_initial_model_state_sha256"] for fold in FOLDS}
        ) == 1,
        "all_historical_initial_matches": all(
            rows[f"fold_{fold}"]["historical_initial_match"] for fold in FOLDS
        ),
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
