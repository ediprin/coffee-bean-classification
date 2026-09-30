from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import sha256_file
from bilinear_lmmd.engine.train import classification_metrics
from bilinear_lmmd.experiments.run_wr_hbp_final_outer_fold import validate_config


ARMS = ("R0_HBP", "WR_HBP")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _paired_stratified_bootstrap_macro_delta(
    y_true: np.ndarray,
    pred_r0: np.ndarray,
    pred_wr: np.ndarray,
    *,
    classes: int,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y_true == c) for c in range(classes)]
    values = np.empty(replicates, dtype=np.float64)
    labels = np.arange(classes)
    for r in range(replicates):
        sampled = np.concatenate(
            [rng.choice(idx, size=idx.size, replace=True) for idx in by_class]
        )
        base = f1_score(
            y_true[sampled], pred_r0[sampled],
            labels=labels, average="macro", zero_division=0,
        )
        cand = f1_score(
            y_true[sampled], pred_wr[sampled],
            labels=labels, average="macro", zero_division=0,
        )
        values[r] = cand - base
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def _targeted_confusions(
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
    pairs: list[list[str]],
) -> dict[str, int]:
    actual = [classes[i] for i in y_true]
    predicted = [classes[i] for i in y_pred]
    out = {}
    for left, right in pairs:
        key = f"{left} <-> {right}"
        out[key] = sum(
            (a == left and p == right) or (a == right and p == left)
            for a, p in zip(actual, predicted)
        )
    out["TOTAL"] = int(sum(out.values()))
    return out


def summarize(
    *,
    output_root: Path,
    authority_path: Path,
    config_path: Path,
    clean_manifest_path: Path,
    output: Path,
    bootstrap_replicates_override: int | None = None,
) -> dict:
    output_root = Path(output_root).resolve()
    authority = _json(Path(authority_path).resolve())
    cfg = load_config(config_path)
    validate_config(cfg)
    clean = _json(Path(clean_manifest_path).resolve())
    expected_ids = {row["identity"] for row in clean["images"]}
    if len(expected_ids) != int(clean["clean_count"]):
        raise RuntimeError("Clean manifest identity count tidak konsisten.")
    if authority.get("clean_content_sha256") != clean.get("clean_content_sha256"):
        raise RuntimeError("Authority/clean population mismatch.")
    authority_sha = sha256_file(Path(authority_path).resolve())

    if authority.get("decision") != "AUTHORIZE_OOF_TEST_EVALUATION":
        raise RuntimeError("Authority final test tidak valid.")
    if authority.get("scope") != "wr-hbp-final-confirmation-v1":
        raise RuntimeError("Authority scope salah.")

    all_ids: list[str] = []
    all_actual: list[str] = []
    all_pred = {arm: [] for arm in ARMS}
    fold_results = {}
    positive_macro = 0

    for fold in range(1, 6):
        fold_dir = output_root / f"fold_{fold}"
        result = _json(fold_dir / "fold_result.json")
        if result.get("protocol") != "wr-hbp-final-confirmation-v1":
            raise RuntimeError(f"Fold {fold}: protocol salah.")
        if result.get("training_executed") is not False:
            raise RuntimeError(f"Fold {fold}: training terjadi saat final test.")
        if result.get("outer_test_accessed") is not True:
            raise RuntimeError(f"Fold {fold}: outer-test flag salah.")
        if result.get("authority_sha256") != authority_sha:
            raise RuntimeError(f"Fold {fold}: authority hash berbeda.")
        if result["DELTA_WR_MINUS_R0"]["macro_f1"] > 0:
            positive_macro += 1
        fold_results[str(fold)] = result

        rows = {arm: _rows(fold_dir / arm / "predictions.csv") for arm in ARMS}
        if len(rows["R0_HBP"]) != len(rows["WR_HBP"]):
            raise RuntimeError(f"Fold {fold}: jumlah prediction berbeda.")
        for left, right in zip(rows["R0_HBP"], rows["WR_HBP"]):
            if (left["identity"], left["actual"]) != (
                right["identity"], right["actual"]
            ):
                raise RuntimeError(f"Fold {fold}: paired identity/label berbeda.")
            all_ids.append(left["identity"])
            all_actual.append(left["actual"])
            all_pred["R0_HBP"].append(left["predicted"])
            all_pred["WR_HBP"].append(right["predicted"])

    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Outer OOF identity muncul lebih dari sekali.")
    if len(all_ids) != int(authority["clean_count"]):
        raise RuntimeError(
            f"Outer OOF count {len(all_ids)} != clean_count {authority['clean_count']}."
        )
    if set(all_ids) != expected_ids:
        missing = sorted(expected_ids.difference(all_ids))
        extra = sorted(set(all_ids).difference(expected_ids))
        raise RuntimeError(
            "Outer OOF identity set berbeda dari clean population; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    classes = fold_results["1"]["classes"]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    y_true = [class_to_idx[name] for name in all_actual]
    y_pred = {
        arm: [class_to_idx[name] for name in all_pred[arm]]
        for arm in ARMS
    }

    pooled = {
        arm: classification_metrics(
            y_true,
            y_pred[arm],
            classes,
            cfg["evaluation"]["hard_groups"],
        )
        for arm in ARMS
    }
    delta = {
        metric: float(pooled["WR_HBP"][metric]) - float(pooled["R0_HBP"][metric])
        for metric in METRICS
    }

    outcomes = {
        "count": len(y_true),
        "rescue": 0,
        "damage": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net_correct": 0,
    }
    for y, a, b in zip(y_true, y_pred["R0_HBP"], y_pred["WR_HBP"]):
        ac, bc = a == y, b == y
        if not ac and bc:
            outcomes["rescue"] += 1
        elif ac and not bc:
            outcomes["damage"] += 1
        elif ac and bc:
            outcomes["both_correct"] += 1
        else:
            outcomes["both_wrong"] += 1
    outcomes["net_correct"] = outcomes["rescue"] - outcomes["damage"]

    targeted = {
        arm: _targeted_confusions(
            y_true,
            y_pred[arm],
            classes,
            cfg["evaluation"]["targeted_confusion_pairs"],
        )
        for arm in ARMS
    }

    gate_cfg = cfg["confirmation_gate"]
    criteria = {
        "pooled_macro_delta_positive": (
            delta["macro_f1"] > float(gate_cfg["pooled_macro_delta_gt"])
        ),
        "macro_positive_outer_folds_at_least_3": (
            positive_macro >= int(gate_cfg["macro_positive_folds_at_least"])
        ),
        "pooled_hard_delta_nonnegative": (
            delta["hard_class_f1"] >= float(gate_cfg["pooled_hard_delta_ge"])
        ),
        "pooled_worst_delta_nonnegative": (
            delta["worst_class_f1"] >= float(gate_cfg["pooled_worst_delta_ge"])
        ),
    }
    decision = (
        "CONFIRMED_WR_HBP"
        if all(criteria.values())
        else "WR_HBP_NOT_CONFIRMED"
    )

    reps = (
        int(bootstrap_replicates_override)
        if bootstrap_replicates_override is not None
        else int(cfg["uncertainty"]["paired_stratified_bootstrap_replicates"])
    )
    bootstrap = _paired_stratified_bootstrap_macro_delta(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred["R0_HBP"], dtype=np.int64),
        np.asarray(y_pred["WR_HBP"], dtype=np.int64),
        classes=len(classes),
        replicates=reps,
        seed=int(cfg["uncertainty"]["seed"]),
    )

    per_class_delta = {
        name: float(pooled["WR_HBP"]["per_class"][name]["f1"])
        - float(pooled["R0_HBP"]["per_class"][name]["f1"])
        for name in classes
    }
    hard_group_delta = {
        name: float(pooled["WR_HBP"]["hard_groups"][name])
        - float(pooled["R0_HBP"]["hard_groups"][name])
        for name in pooled["R0_HBP"]["hard_groups"]
    }

    payload = {
        "format": "bilinear_lmmd.wr_hbp_final_outer_summary.v1",
        "protocol": "wr-hbp-final-confirmation-v1",
        "folds": fold_results,
        "oof_identity_count": len(all_ids),
        "oof_identity_unique": True,
        "pooled": {
            arm: {metric: float(pooled[arm][metric]) for metric in METRICS}
            for arm in ARMS
        },
        "delta_wr_minus_r0": delta,
        "positive_macro_outer_folds": positive_macro,
        "paired_prediction_outcomes": outcomes,
        "targeted_confusions": targeted,
        "per_class_f1_delta": per_class_delta,
        "hard_group_f1_delta": hard_group_delta,
        "paired_stratified_bootstrap_macro_delta": bootstrap,
        "confirmation_gate": {
            "criteria": criteria,
            "decision": decision,
        },
        "training_executed": False,
        "outer_test_accessed": True,
        "further_tuning_authorized": False,
    }

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--clean-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(
        output_root=args.output_root,
        authority_path=args.authority,
        config_path=args.config,
        clean_manifest_path=args.clean_manifest,
        output=args.output,
    )


if __name__ == "__main__":
    main()
