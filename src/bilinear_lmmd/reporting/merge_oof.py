from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bilinear_lmmd.core.config import DEFAULTS
from bilinear_lmmd.engine.train import classification_metrics


def merge_oof(
    report_dirs: list[Path], output_dir: Path, expected_count: int = 979
) -> dict:
    if not report_dirs:
        raise ValueError("Minimal satu report fold diperlukan.")

    report_metrics = [
        json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
        for report_dir in report_dirs
    ]
    classes = report_metrics[0]["classes"]
    if any(report["classes"] != classes for report in report_metrics):
        raise ValueError("Urutan kelas antar-fold berbeda.")
    class_to_index = {name: index for index, name in enumerate(classes)}

    merged_rows: list[dict[str, str]] = []
    identities: set[str] = set()
    labels: list[int] = []
    predictions: list[int] = []
    for report_dir in report_dirs:
        with (report_dir / "predictions.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                path = Path(row["path"])
                identity = f"{path.parent.name}/{path.name}"
                if identity in identities:
                    raise ValueError(f"Prediksi OOF duplikat: {identity}")
                identities.add(identity)
                actual = row["actual"]
                predicted = row["predicted"]
                labels.append(class_to_index[actual])
                predictions.append(class_to_index[predicted])
                merged_rows.append(
                    {
                        "identity": identity,
                        "actual": actual,
                        "predicted": predicted,
                        "correct": str(int(actual == predicted)),
                        "fold_report": str(report_dir),
                    }
                )

    if len(identities) != expected_count:
        raise ValueError(
            f"OOF predictions harus mencakup {expected_count} identitas, "
            f"ditemukan {len(identities)}."
        )
    hard_groups = DEFAULTS["evaluation"]["hard_groups"]
    metrics = classification_metrics(labels, predictions, classes, hard_groups)
    metrics.update(
        {
            "classes": classes,
            "sample_count": len(labels),
            "fold_reports": [str(path) for path in report_dirs],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("identity", "actual", "predicted", "correct", "fold_report"),
        )
        writer.writeheader()
        writer.writerows(sorted(merged_rows, key=lambda row: row["identity"]))
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "per_class"},
            indent=2,
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Gabungkan prediksi test grouped folds")
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=979)
    args = parser.parse_args()
    merge_oof(args.reports, args.output_dir, args.expected_count)


if __name__ == "__main__":
    main()
