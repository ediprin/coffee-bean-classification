from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULTS
from .train import classification_metrics


@dataclass(frozen=True)
class GateRow:
    identity: str
    fold: int
    actual: str
    hbp: str
    attribute: str


MIN_SUPPORT_VALUES = (2, 3, 5, 8, 12)
# margin=1.0 tidak pernah memilih attribute (advantage maksimum 1.0), sehingga
# inner validation selalu dapat kembali ke baseline HBP murni.
MARGIN_VALUES = (0.0, 0.05, 0.10, 0.20, 1.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions tidak ditemukan: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Predictions kosong: {path}")
    return rows


def load_gate_rows(
    attribute_predictions: Path,
    hbp_predictions: Path,
) -> list[GateRow]:
    attribute_rows = _read_csv(attribute_predictions)
    hbp_rows = _read_csv(hbp_predictions)
    required_attribute = {"identity", "fold", "actual", "predicted"}
    required_hbp = {"identity", "actual", "predicted"}
    if not required_attribute.issubset(attribute_rows[0]):
        raise ValueError(
            f"Kolom attribute wajib: {sorted(required_attribute)}"
        )
    if not required_hbp.issubset(hbp_rows[0]):
        raise ValueError(f"Kolom HBP wajib: {sorted(required_hbp)}")

    attribute = {row["identity"]: row for row in attribute_rows}
    hbp = {row["identity"]: row for row in hbp_rows}
    if len(attribute) != len(attribute_rows) or len(hbp) != len(hbp_rows):
        raise ValueError("Terdapat identity duplikat pada predictions.")
    if attribute.keys() != hbp.keys():
        raise ValueError("Identity predictions attribute dan HBP berbeda.")

    result: list[GateRow] = []
    for identity in sorted(attribute):
        left = attribute[identity]
        right = hbp[identity]
        if left["actual"] != right["actual"]:
            raise ValueError(f"Label aktual berbeda untuk {identity}.")
        result.append(
            GateRow(
                identity=identity,
                fold=int(left["fold"]),
                actual=left["actual"],
                hbp=right["predicted"],
                attribute=left["predicted"],
            )
        )
    return result


def fit_pair_rules(
    rows: list[GateRow],
    min_support: int,
    margin: float,
) -> dict[tuple[str, str], dict]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "support": 0,
            "attribute_correct": 0,
            "hbp_correct": 0,
            "both_wrong": 0,
        }
    )
    for row in rows:
        if row.hbp == row.attribute:
            continue
        pair = (row.hbp, row.attribute)
        stats = counts[pair]
        stats["support"] += 1
        attribute_correct = row.attribute == row.actual
        hbp_correct = row.hbp == row.actual
        stats["attribute_correct"] += int(attribute_correct)
        stats["hbp_correct"] += int(hbp_correct)
        stats["both_wrong"] += int(not attribute_correct and not hbp_correct)

    rules: dict[tuple[str, str], dict] = {}
    for pair, stats in counts.items():
        advantage = (
            stats["attribute_correct"] - stats["hbp_correct"]
        ) / stats["support"]
        rules[pair] = {
            **stats,
            "attribute_advantage": advantage,
            "choose": (
                "attribute"
                if stats["support"] >= min_support and advantage > margin
                else "hbp"
            ),
        }
    return rules


def apply_pair_rules(
    rows: list[GateRow], rules: dict[tuple[str, str], dict]
) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    decisions: list[str] = []
    for row in rows:
        if row.hbp == row.attribute:
            predictions.append(row.hbp)
            decisions.append("agree")
            continue
        rule = rules.get((row.hbp, row.attribute))
        if rule is not None and rule["choose"] == "attribute":
            predictions.append(row.attribute)
            decisions.append("attribute")
        else:
            predictions.append(row.hbp)
            decisions.append("hbp")
    return predictions, decisions


def _metrics(
    rows: list[GateRow],
    predictions: list[str],
    classes: list[str],
    hard_groups: dict[str, list[str]],
) -> dict:
    class_to_index = {name: index for index, name in enumerate(classes)}
    labels = [class_to_index[row.actual] for row in rows]
    predicted = [class_to_index[name] for name in predictions]
    return classification_metrics(labels, predicted, classes, hard_groups)


def _score(metrics: dict, min_support: int, margin: float) -> tuple:
    return (
        metrics["macro_f1"],
        metrics["hard_class_f1"] or 0.0,
        metrics["worst_class_f1"],
        metrics["accuracy"],
        min_support,
        margin,
    )


def select_gate_parameters(
    training_rows: list[GateRow],
    classes: list[str],
    hard_groups: dict[str, list[str]],
) -> tuple[dict, list[dict]]:
    folds = sorted({row.fold for row in training_rows})
    if len(folds) < 3:
        raise ValueError("Pemilihan gate memerlukan minimal tiga training fold.")
    candidates: list[dict] = []
    for min_support in MIN_SUPPORT_VALUES:
        for margin in MARGIN_VALUES:
            validation_rows: list[GateRow] = []
            validation_predictions: list[str] = []
            for validation_fold in folds:
                inner_train = [
                    row for row in training_rows if row.fold != validation_fold
                ]
                inner_validation = [
                    row for row in training_rows if row.fold == validation_fold
                ]
                rules = fit_pair_rules(inner_train, min_support, margin)
                predicted, _ = apply_pair_rules(inner_validation, rules)
                validation_rows.extend(inner_validation)
                validation_predictions.extend(predicted)
            metrics = _metrics(
                validation_rows,
                validation_predictions,
                classes,
                hard_groups,
            )
            candidates.append(
                {
                    "min_support": min_support,
                    "margin": margin,
                    "macro_f1": metrics["macro_f1"],
                    "hard_class_f1": metrics["hard_class_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                    "accuracy": metrics["accuracy"],
                }
            )
    selected = max(
        candidates,
        key=lambda item: _score(
            item, item["min_support"], item["margin"]
        ),
    )
    return selected, candidates


def cross_fitted_gate(
    rows: list[GateRow],
    classes: list[str],
    hard_groups: dict[str, list[str]],
) -> tuple[list[dict], list[dict]]:
    folds = sorted({row.fold for row in rows})
    if len(folds) < 4:
        raise ValueError("Cross-fitted gate memerlukan minimal empat fold.")
    prediction_rows: list[dict] = []
    fold_results: list[dict] = []
    for outer_fold in folds:
        training_rows = [row for row in rows if row.fold != outer_fold]
        test_rows = [row for row in rows if row.fold == outer_fold]
        selected, curve = select_gate_parameters(
            training_rows, classes, hard_groups
        )
        rules = fit_pair_rules(
            training_rows,
            selected["min_support"],
            selected["margin"],
        )
        predictions, decisions = apply_pair_rules(test_rows, rules)
        test_metrics = _metrics(test_rows, predictions, classes, hard_groups)
        chosen_pairs = [
            {
                "hbp": pair[0],
                "attribute": pair[1],
                **stats,
            }
            for pair, stats in sorted(rules.items())
            if stats["choose"] == "attribute"
        ]
        fold_results.append(
            {
                "fold": outer_fold,
                "selected": {
                    "min_support": selected["min_support"],
                    "margin": selected["margin"],
                    "inner_macro_f1": selected["macro_f1"],
                },
                "test_macro_f1": test_metrics["macro_f1"],
                "chosen_attribute_pairs": chosen_pairs,
                "selection_curve": curve,
            }
        )
        print(
            f"fold {outer_fold}: support={selected['min_support']} "
            f"margin={selected['margin']:.2f} "
            f"inner={selected['macro_f1']:.2%} "
            f"test={test_metrics['macro_f1']:.2%}",
            flush=True,
        )
        for row, prediction, decision in zip(test_rows, predictions, decisions):
            prediction_rows.append(
                {
                    "identity": row.identity,
                    "fold": row.fold,
                    "actual": row.actual,
                    "hbp": row.hbp,
                    "attribute": row.attribute,
                    "gate": prediction,
                    "decision": decision,
                    "correct": int(prediction == row.actual),
                }
            )
    return prediction_rows, fold_results


def _decision_audit(rows: list[dict]) -> dict:
    audit = {
        "agree": 0,
        "choose_hbp": 0,
        "choose_attribute": 0,
        "attribute_choice_correct": 0,
        "attribute_choice_hbp_was_correct": 0,
        "attribute_choice_both_wrong": 0,
    }
    for row in rows:
        decision = row["decision"]
        if decision == "agree":
            audit["agree"] += 1
        elif decision == "hbp":
            audit["choose_hbp"] += 1
        else:
            audit["choose_attribute"] += 1
            attribute_correct = row["attribute"] == row["actual"]
            hbp_correct = row["hbp"] == row["actual"]
            audit["attribute_choice_correct"] += int(attribute_correct)
            audit["attribute_choice_hbp_was_correct"] += int(hbp_correct)
            audit["attribute_choice_both_wrong"] += int(
                not attribute_correct and not hbp_correct
            )
    return audit


def run_disagreement_gate(
    attribute_predictions: Path,
    hbp_predictions: Path,
    output_dir: Path,
    expected_count: int = 979,
) -> dict:
    rows = load_gate_rows(attribute_predictions, hbp_predictions)
    if len(rows) != expected_count:
        raise ValueError(
            f"Predictions harus mencakup {expected_count} sampel, "
            f"ditemukan {len(rows)}."
        )
    classes = sorted({row.actual for row in rows})
    unknown_predictions = sorted(
        {
            predicted
            for row in rows
            for predicted in (row.hbp, row.attribute)
            if predicted not in classes
        }
    )
    if unknown_predictions:
        raise ValueError(f"Prediksi kelas tidak dikenal: {unknown_predictions}")
    hard_groups = DEFAULTS["evaluation"]["hard_groups"]
    prediction_rows, fold_results = cross_fitted_gate(
        rows, classes, hard_groups
    )
    by_identity = {row.identity: row for row in rows}
    ordered_gate_rows = [by_identity[row["identity"]] for row in prediction_rows]
    gate_predictions = [row["gate"] for row in prediction_rows]
    hbp_predictions_list = [row.hbp for row in ordered_gate_rows]
    attribute_predictions_list = [row.attribute for row in ordered_gate_rows]
    metrics = {
        "hbp": _metrics(
            ordered_gate_rows, hbp_predictions_list, classes, hard_groups
        ),
        "attribute": _metrics(
            ordered_gate_rows, attribute_predictions_list, classes, hard_groups
        ),
        "gate": _metrics(
            ordered_gate_rows, gate_predictions, classes, hard_groups
        ),
    }
    delta = {
        name: metrics["gate"][name] - metrics["hbp"][name]
        for name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "hard_class_f1",
            "worst_class_f1",
        )
    }
    result = {
        "method": "nested leave-one-fold pairwise HBP-attribute disagreement gate",
        "sample_count": len(rows),
        "classes": classes,
        "metrics": metrics,
        "delta_gate_vs_hbp": delta,
        "decision_audit": _decision_audit(prediction_rows),
        "folds": fold_results,
        "limitation": (
            "Exploratory meta-screening on existing OOF hard predictions; "
            "a final claim requires gate calibration from base-model validation "
            "predictions or an independent test set."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = (
            "identity",
            "fold",
            "actual",
            "hbp",
            "attribute",
            "gate",
            "decision",
            "correct",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(prediction_rows, key=lambda row: row["identity"]))

    print("\n=== CROSS-FITTED HBP-CST GATE ===")
    for name in ("hbp", "attribute", "gate"):
        item = metrics[name]
        print(
            f"{name.upper():9s} Accuracy={item['accuracy']:.2%} "
            f"Macro={item['macro_f1']:.2%} "
            f"Hard={item['hard_class_f1']:.2%} "
            f"Worst={item['worst_class_f1']:.2%}"
        )
    print("\nDELTA GATE vs HBP")
    for name, value in delta.items():
        print(f"{name:20s}: {value:+.2%}")
    audit = result["decision_audit"]
    print(
        f"\nGate memilih CST pada {audit['choose_attribute']} sampel: "
        f"benar={audit['attribute_choice_correct']}, "
        f"HBP-sebenarnya-benar={audit['attribute_choice_hbp_was_correct']}, "
        f"keduanya-salah={audit['attribute_choice_both_wrong']}"
    )
    print(f"SAVED: {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fitted hard-label disagreement gate untuk HBP dan CST"
    )
    parser.add_argument("--attribute-predictions", type=Path, required=True)
    parser.add_argument("--hbp-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=979)
    args = parser.parse_args()
    run_disagreement_gate(
        attribute_predictions=args.attribute_predictions,
        hbp_predictions=args.hbp_predictions,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()
