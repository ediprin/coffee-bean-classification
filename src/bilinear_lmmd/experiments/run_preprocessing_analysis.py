from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


ARMS = ("R0", "C0", "F0", "W0")


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_matrix(path: Path, classes: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *classes])
        for name, row in zip(classes, matrix.tolist()):
            writer.writerow([name, *row])


def run_analysis(master_table: Path, output_dir: Path) -> dict:
    rows = _read(master_table)
    if not rows:
        raise ValueError("OOF master table kosong.")
    classes = sorted({row["actual"] for row in rows})
    index = {name: i for i, name in enumerate(classes)}
    actual = np.asarray([index[row["actual"]] for row in rows], dtype=np.int64)
    predicted = {
        arm: np.asarray(
            [index[row[f"{arm}_predicted"]] for row in rows],
            dtype=np.int64,
        )
        for arm in ARMS
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    f1_by_arm: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    supports = None
    for arm in ARMS:
        _, _, f1, support = precision_recall_fscore_support(
            actual,
            predicted[arm],
            labels=list(range(len(classes))),
            zero_division=0,
        )
        f1_by_arm[arm] = f1
        supports = support
        matrices[arm] = confusion_matrix(
            actual,
            predicted[arm],
            labels=list(range(len(classes))),
        )
        _write_matrix(output_dir / f"{arm}_confusion.csv", classes, matrices[arm])

    for arm in ARMS[1:]:
        _write_matrix(
            output_dir / f"{arm}_minus_R0_confusion.csv",
            classes,
            matrices[arm] - matrices["R0"],
        )

    per_class_path = output_dir / "per_class_effects.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["class", "support"]
        for arm in ARMS:
            fields.append(f"{arm}_f1")
            if arm != "R0":
                fields.append(f"{arm}_delta_vs_R0")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for class_index, name in enumerate(classes):
            row = {
                "class": name,
                "support": int(supports[class_index]),
            }
            for arm in ARMS:
                row[f"{arm}_f1"] = float(f1_by_arm[arm][class_index])
                if arm != "R0":
                    row[f"{arm}_delta_vs_R0"] = float(
                        f1_by_arm[arm][class_index] - f1_by_arm["R0"][class_index]
                    )
            writer.writerow(row)

    summary = {
        "format": "bilinear_lmmd.preprocessing.oof_analysis.v1",
        "samples": len(rows),
        "classes": classes,
        "per_class_effects": str(per_class_path),
        "confusion_delta_definition": "candidate confusion counts minus R0 counts",
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_analysis(args.master_table, args.output_dir)


if __name__ == "__main__":
    main()
