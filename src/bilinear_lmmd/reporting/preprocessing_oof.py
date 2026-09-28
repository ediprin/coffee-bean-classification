from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix

from bilinear_lmmd.core.reproducibility import sha256_file
from bilinear_lmmd.engine.train import classification_metrics
from bilinear_lmmd.experiments.preprocessing_contract import ARMS


FOLDS = (1, 2, 3, 4, 5)
SEED = 42


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(path_value: str) -> str:
    path = Path(path_value)
    return f"{path.parent.name}/{path.name}"


def _read_prediction_table(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_primary_oof(
    raw_reports_root: Path,
    clean_manifest_path: Path,
    fold_manifest_path: Path,
    output_root: Path,
    *,
    authority_sha256: str,
) -> dict:
    raw_reports_root = Path(raw_reports_root).resolve()
    output_root = Path(output_root).resolve()
    clean = _json(Path(clean_manifest_path).resolve())
    folds = _json(Path(fold_manifest_path).resolve())
    expected_ids = {row["identity"] for row in clean["images"]}
    expected_count = int(clean["clean_count"])
    if len(expected_ids) != expected_count:
        raise RuntimeError("Clean manifest memiliki identity count yang tidak konsisten.")

    arm_tables: dict[str, dict[str, dict]] = {}
    arm_metrics: dict[str, dict] = {}
    classes: list[str] | None = None

    for arm in ARMS:
        rows_by_id: dict[str, dict] = {}
        labels: list[int] = []
        predictions: list[int] = []
        class_to_index: dict[str, int] | None = None

        for fold in FOLDS:
            report_dir = raw_reports_root / f"{arm}_fold{fold}_seed{SEED}"
            report_meta = _json(report_dir / "test_report.json")
            if (
                report_meta.get("arm") != arm
                or int(report_meta.get("fold", -1)) != fold
                or int(report_meta.get("seed", -1)) != SEED
                or report_meta.get("training_executed") is not False
                or report_meta.get("test_images_accessed") is not True
                or report_meta.get("authority_sha256") != authority_sha256
            ):
                raise RuntimeError(f"Test report contract invalid: {report_dir}")

            metrics = _json(report_dir / "metrics.json")
            current_classes = metrics["classes"]
            if classes is None:
                classes = current_classes
            elif classes != current_classes:
                raise RuntimeError("Urutan kelas OOF berbeda antar-report.")
            if class_to_index is None:
                class_to_index = {name: index for index, name in enumerate(classes)}

            fold_rows = _read_prediction_table(report_dir / "predictions.csv")
            expected_fold_ids = set(folds["assignments"][f"fold_{fold}"]["test"])
            observed_fold_ids = {_identity(row["path"]) for row in fold_rows}
            if observed_fold_ids != expected_fold_ids:
                raise RuntimeError(
                    f"{arm}/fold{fold} identity test tidak sama dengan frozen manifest."
                )

            for row in fold_rows:
                identity = _identity(row["path"])
                if identity in rows_by_id:
                    raise RuntimeError(f"OOF identity duplikat {arm}: {identity}")
                if identity not in expected_ids:
                    raise RuntimeError(f"OOF identity bukan clean population: {identity}")
                rows_by_id[identity] = {
                    "identity": identity,
                    "actual": row["actual"],
                    "predicted": row["predicted"],
                    "correct": row["correct"],
                    **{
                        key: value
                        for key, value in row.items()
                        if key.startswith("prob::")
                    },
                }

        if set(rows_by_id) != expected_ids or len(rows_by_id) != expected_count:
            raise RuntimeError(
                f"{arm} OOF harus tepat {expected_count} clean identities."
            )
        assert classes is not None and class_to_index is not None
        for identity in sorted(rows_by_id):
            row = rows_by_id[identity]
            labels.append(class_to_index[row["actual"]])
            predictions.append(class_to_index[row["predicted"]])

        metrics = classification_metrics(
            labels,
            predictions,
            classes,
            {
                "sour_black": ["Partial Black", "Partial Sour", "Full Sour"],
                "shape_withered": ["Withered", "Immature", "Cut"],
                "insect_damage": ["Slight Insect Damage", "Severe Insect Damage"],
            },
        )
        arm_tables[arm] = rows_by_id
        arm_metrics[arm] = metrics

        arm_dir = output_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        with (arm_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(next(iter(rows_by_id.values())).keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_by_id[identity] for identity in sorted(rows_by_id))
        (arm_dir / "metrics.json").write_text(
            json.dumps({**metrics,"classes": classes,"sample_count": expected_count,"authority_sha256": authority_sha256},indent=2)+"\n",encoding="utf-8",
        )

    assert classes is not None
    reference = arm_tables["R0"]
    for arm in ARMS[1:]:
        for identity in expected_ids:
            if arm_tables[arm][identity]["actual"] != reference[identity]["actual"]:
                raise RuntimeError(f"Ground truth berbeda antar-arm: {identity}")

    master_path = output_root / "primary_oof_table.csv"
    with master_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["identity", "actual"]
        for arm in ARMS:
            fieldnames += [f"{arm}_predicted", f"{arm}_correct"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for identity in sorted(expected_ids):
            row = {"identity": identity,"actual": reference[identity]["actual"]}
            for arm in ARMS:
                row[f"{arm}_predicted"] = arm_tables[arm][identity]["predicted"]
                row[f"{arm}_correct"] = arm_tables[arm][identity]["correct"]
            writer.writerow(row)

    summary = {"format":"bilinear_lmmd.preprocessing.primary_oof.v1","protocol":"coffee17-preprocessing-primary-v1","authority_sha256":authority_sha256,"clean_content_sha256":clean["clean_content_sha256"],"fold_manifest_sha256":sha256_file(Path(fold_manifest_path).resolve()),"sample_count":expected_count,"classes":classes,"arms":arm_metrics,"paired_identity_alignment":True,"training_executed":False,"test_images_accessed":True}
    (output_root / "primary_oof_summary.json").write_text(json.dumps(summary, indent=2) + "\n",encoding="utf-8")
    return summary
