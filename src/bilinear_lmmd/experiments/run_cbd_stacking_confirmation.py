from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bilinear_lmmd.core.config import load_config


MODEL_CONFIGS = {
    "CBD0": Path("configs/cbd/CBD0_mobilenetv3_gap_source.yaml"),
    "CBD1": Path("configs/cbd/CBD1_mobilenetv3_hbp_source.yaml"),
}
MODEL_NAMES = ("GAP_RAW", "HBP_RAW", "GAP_CAL", "HBP_CAL", "STACKING")
METRICS = ("accuracy", "macro_f1", "defect_f1", "worst_f1")


@dataclass(frozen=True)
class PredictionTable:
    paths: list[str]
    classes: list[str]
    labels: np.ndarray
    probabilities: np.ndarray


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _training_complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    best = run_dir / "best.pt"
    if not history.is_file() or not best.is_file():
        return False
    try:
        return len(json.loads(history.read_text(encoding="utf-8"))) >= epochs
    except (json.JSONDecodeError, OSError):
        return False


def load_prediction_table(path: Path) -> PredictionTable:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV tanpa header: {path}")
        probability_columns = [
            name for name in reader.fieldnames if name.startswith("prob::")
        ]
        if not probability_columns:
            raise ValueError(f"Kolom probabilitas tidak ditemukan: {path}")
        rows = sorted(reader, key=lambda row: row["path"])
    classes = [name.removeprefix("prob::") for name in probability_columns]
    class_to_index = {name: index for index, name in enumerate(classes)}
    try:
        labels = np.asarray(
            [class_to_index[row["actual"]] for row in rows], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(f"Label aktual tidak ada pada kolom probabilitas: {exc}") from exc
    probabilities = np.asarray(
        [
            [float(row[column]) for column in probability_columns]
            for row in rows
        ],
        dtype=np.float64,
    )
    return PredictionTable(
        paths=[row["path"] for row in rows],
        classes=classes,
        labels=labels,
        probabilities=np.clip(probabilities, 1e-8, 1.0),
    )


def _validate_pair(first: PredictionTable, second: PredictionTable) -> None:
    if first.paths != second.paths:
        raise ValueError("Urutan/path prediksi GAP dan HBP berbeda.")
    if first.classes != second.classes:
        raise ValueError("Urutan kelas prediksi GAP dan HBP berbeda.")
    if not np.array_equal(first.labels, second.labels):
        raise ValueError("Label prediksi GAP dan HBP berbeda.")


def _metrics(labels: np.ndarray, predictions: np.ndarray, classes: list[str]) -> dict:
    _, _, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(classes)),
        zero_division=0,
    )
    defect_indices = [
        index for index, name in enumerate(classes) if name != "Good"
    ]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "defect_f1": float(f1[defect_indices].mean()),
        "worst_f1": float(f1.min()),
        "worst_class": classes[int(f1.argmin())],
    }


def fit_meta_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> object:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    )
    model.fit(train_features, train_labels)
    return model


def _meta_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    return fit_meta_model(train_features, train_labels).predict(test_features)


def stack_seed(
    gap_val_path: Path,
    hbp_val_path: Path,
    gap_test_path: Path,
    hbp_test_path: Path,
) -> tuple[dict, list[dict]]:
    gap_val = load_prediction_table(gap_val_path)
    hbp_val = load_prediction_table(hbp_val_path)
    gap_test = load_prediction_table(gap_test_path)
    hbp_test = load_prediction_table(hbp_test_path)
    _validate_pair(gap_val, hbp_val)
    _validate_pair(gap_test, hbp_test)
    if gap_val.classes != gap_test.classes:
        raise ValueError("Urutan kelas validation dan test berbeda.")

    gap_val_log = np.log(gap_val.probabilities)
    hbp_val_log = np.log(hbp_val.probabilities)
    gap_test_log = np.log(gap_test.probabilities)
    hbp_test_log = np.log(hbp_test.probabilities)
    predictions = {
        "GAP_RAW": gap_test.probabilities.argmax(axis=1),
        "HBP_RAW": hbp_test.probabilities.argmax(axis=1),
        "GAP_CAL": _meta_predict(gap_val_log, gap_val.labels, gap_test_log),
        "HBP_CAL": _meta_predict(hbp_val_log, gap_val.labels, hbp_test_log),
        "STACKING": _meta_predict(
            np.concatenate((gap_val_log, hbp_val_log), axis=1),
            gap_val.labels,
            np.concatenate((gap_test_log, hbp_test_log), axis=1),
        ),
    }
    result = {
        "classes": gap_test.classes,
        "validation_samples": len(gap_val.labels),
        "test_samples": len(gap_test.labels),
        "meta_model": {
            "features": "concatenated GAP/HBP log-probabilities",
            "scaler": "StandardScaler",
            "classifier": "LogisticRegression",
            "C": 1.0,
            "max_iter": 2000,
            "random_state": 42,
        },
        "models": {
            name: _metrics(gap_test.labels, prediction, gap_test.classes)
            for name, prediction in predictions.items()
        },
    }
    rows = []
    for index, path in enumerate(gap_test.paths):
        row = {
            "path": path,
            "actual": gap_test.classes[int(gap_test.labels[index])],
        }
        for name, prediction in predictions.items():
            row[name] = gap_test.classes[int(prediction[index])]
        rows.append(row)
    return result, rows


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_seed_results(seed_results: dict[int, dict]) -> dict:
    models = {
        model: {
            metric: _mean_std(
                [seed_results[seed]["models"][model][metric] for seed in seed_results]
            )
            for metric in METRICS
        }
        for model in MODEL_NAMES
    }
    deltas = {}
    for control in ("GAP_CAL", "HBP_CAL"):
        deltas[control] = {}
        for metric in METRICS:
            values = [
                seed_results[seed]["models"]["STACKING"][metric]
                - seed_results[seed]["models"][control][metric]
                for seed in seed_results
            ]
            deltas[control][metric] = {
                **_mean_std(values),
                "improved_seeds": sum(value > 0 for value in values),
                "total_seeds": len(values),
                "values": values,
            }

    stack_macro = models["STACKING"]["macro_f1"]["mean"]
    best_control_macro = max(
        models["GAP_CAL"]["macro_f1"]["mean"],
        models["HBP_CAL"]["macro_f1"]["mean"],
    )
    stack_worst = models["STACKING"]["worst_f1"]["mean"]
    best_control_worst = max(
        models["GAP_CAL"]["worst_f1"]["mean"],
        models["HBP_CAL"]["worst_f1"]["mean"],
    )
    macro_margin = stack_macro - best_control_macro
    worst_margin = stack_worst - best_control_worst
    consistent = all(
        deltas[control]["macro_f1"]["improved_seeds"] >= 2
        for control in ("GAP_CAL", "HBP_CAL")
    )
    full_confirmation = len(seed_results) >= 3
    passed = (
        full_confirmation
        and macro_margin >= 0.003
        and worst_margin >= 0.0
        and consistent
    )
    return {
        "seeds": list(seed_results),
        "models": models,
        "stacking_deltas": deltas,
        "pre_registered_decision": {
            "macro_margin_vs_best_calibrated": macro_margin,
            "worst_margin_vs_best_calibrated": worst_margin,
            "macro_improves_vs_each_control_on_at_least_2_of_3_seeds": consistent,
            "criteria": {
                "macro_margin_minimum": 0.003,
                "worst_margin_minimum": 0.0,
                "minimum_improved_seeds_vs_each_control": 2,
            },
            "status": (
                "PASS" if passed else ("FAIL" if full_confirmation else "SCREEN_ONLY")
            ),
        },
    }


def _write_prediction_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "actual", *MODEL_NAMES])
        writer.writeheader()
        writer.writerows(rows)


def run_confirmation(
    data_root: Path,
    output_root: Path,
    seeds: list[int],
) -> dict:
    audit_path = data_root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit CBD tidak ditemukan: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("generated_cross_split_identity_count") != 0:
        raise RuntimeError("Identity leakage ditemukan pada dataset CBD.")

    for code, config_path in MODEL_CONFIGS.items():
        epochs = int(load_config(config_path)["training"]["epochs"])
        for seed in seeds:
            run_dir = output_root / "outputs" / f"{code}_seed{seed}"
            if _training_complete(run_dir, epochs):
                print(f"SKIP training lengkap: {code} seed {seed}", flush=True)
            else:
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.engine.train",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(run_dir),
                        "--resume",
                    ]
                )
            for split in ("val", "test"):
                report_root = output_root / (
                    "reports" if split == "test" else "val_reports"
                )
                report_dir = report_root / f"{code}_seed{seed}"
                if (report_dir / "predictions.csv").is_file():
                    print(f"SKIP prediksi: {code} seed {seed} {split}", flush=True)
                    continue
                _run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "bilinear_lmmd.engine.evaluate_checkpoint",
                        "--checkpoint",
                        str(run_dir / "best.pt"),
                        "--domain",
                        "source",
                        "--split",
                        split,
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(report_dir),
                    ]
                )

    stacking_root = output_root / "stacking_reports"
    seed_results = {}
    for seed in seeds:
        result, rows = stack_seed(
            output_root / "val_reports" / f"CBD0_seed{seed}" / "predictions.csv",
            output_root / "val_reports" / f"CBD1_seed{seed}" / "predictions.csv",
            output_root / "reports" / f"CBD0_seed{seed}" / "predictions.csv",
            output_root / "reports" / f"CBD1_seed{seed}" / "predictions.csv",
        )
        result["seed"] = seed
        seed_results[seed] = result
        seed_dir = stacking_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        _write_prediction_rows(seed_dir / "predictions.csv", rows)

        print(f"\n=== STACKING SEED {seed} ===")
        for name in MODEL_NAMES:
            row = result["models"][name]
            print(
                f"{name:8s} Macro={row['macro_f1']:.2%} "
                f"Defect={row['defect_f1']:.2%} Worst={row['worst_f1']:.2%}"
            )

    aggregate = aggregate_seed_results(seed_results)
    destination = stacking_root / "stacking_confirmation.json"
    destination.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("\n=== AGREGAT STACKING ===")
    for name in MODEL_NAMES:
        row = aggregate["models"][name]
        print(
            f"{name:8s} Macro={row['macro_f1']['mean']:.2%}±"
            f"{row['macro_f1']['std']:.2%} "
            f"Worst={row['worst_f1']['mean']:.2%}±{row['worst_f1']['std']:.2%}"
        )
    decision = aggregate["pre_registered_decision"]
    print(
        "Fusion margin vs calibrated terbaik: "
        f"Macro={decision['macro_margin_vs_best_calibrated']:+.2%} "
        f"Worst={decision['worst_margin_vs_best_calibrated']:+.2%}"
    )
    print(f"KEPUTUSAN: {decision['status']}")
    print(f"SAVED: {destination}")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konfirmasi tiga seed logistic stacking GAP-HBP pada CBD"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/cbd_multiclassify_prepared")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    args = parser.parse_args()
    run_confirmation(args.data_root, args.output_root, args.seeds)


if __name__ == "__main__":
    main()
