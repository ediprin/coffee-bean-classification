from __future__ import annotations

import argparse
from pathlib import Path

from bilinear_lmmd.analysis.confusion_pairs import (
    PredictionRun,
    build_confusion_audit,
    write_confusion_audit,
)


def parse_prediction_spec(value: str) -> PredictionRun:
    try:
        run, raw_path = value.split("=", 1)
        model, raw_seed = run.rsplit(":", 1)
        return PredictionRun(model=model, seed=int(raw_seed), path=Path(raw_path))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "Format --prediction harus MODEL:SEED=/path/predictions.csv"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit confusion pair lintas model/seed tanpa training"
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        type=parse_prediction_spec,
        help="Boleh diulang: MODEL:SEED=/path/predictions.csv",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    audit = build_confusion_audit(args.prediction)
    write_confusion_audit(audit, args.output_dir)
    print("=== CONFUSION PAIR AUDIT ===")
    print("Runs             :", len(audit["runs"]))
    print("Samples          :", audit["sample_count"])
    print("Unanimous wrong  :", audit["unanimous_wrong_count"])
    print("Stable pairs     :", len(audit["stable_pairs"]))
    for row in audit["stable_pairs"][:10]:
        print(
            f"- {row['class_a']} <-> {row['class_b']}: "
            f"errors={row['errors']} models={row['distinct_models']} "
            f"seeds={row['distinct_seeds']}"
        )
    print("SAVED:", args.output_dir / "confusion_audit.json")


if __name__ == "__main__":
    main()
