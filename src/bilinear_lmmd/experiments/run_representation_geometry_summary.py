from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FOLDS = (1, 2, 3, 4, 5)
ENCODERS = (
    "mobilenetv3_imagenet",
    "dinov2_small",
    "convnext_tiny_imagenet",
)
DECODERS = ("knn5", "nearest_centroid", "linear_probe")
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


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _majority_correct(row: dict[str, str]) -> bool:
    actual = row["actual"]
    votes = sum(
        row[key] == actual
        for key in (
            "knn5_predicted",
            "nearest_centroid_predicted",
            "linear_probe_predicted",
        )
    )
    return votes >= 2


def run_summary(*, output_root: Path, output: Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()

    fold_results = {}
    for fold in FOLDS:
        root = output_root / f"fold_{fold}"
        result = _json(root / "audit_result.json")
        if result.get("protocol") != "coffee17-representation-geometry-audit-v1":
            raise RuntimeError(f"Fold {fold}: protocol mismatch.")
        if result.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Fold {fold}: outer test tersentuh.")
        if result.get("determinism", {}).get("deterministic_algorithms") is not True:
            raise RuntimeError(f"Fold {fold}: determinism tidak aktif.")
        if tuple(result.get("encoders", {}).keys()) != ENCODERS:
            raise RuntimeError(f"Fold {fold}: encoder set/order berubah.")
        fold_results[fold] = result

    aggregate = {}
    for encoder in ENCODERS:
        aggregate[encoder] = {}
        for decoder in DECODERS:
            aggregate[encoder][decoder] = {
                metric: _stats([
                    float(
                        fold_results[fold]["encoders"][encoder]
                        ["metrics"][decoder][metric]
                    )
                    for fold in FOLDS
                ])
                for metric in METRICS
            }

    pair_names = list(
        fold_results[1]["encoders"]["mobilenetv3_imagenet"]
        ["pair_geometry"].keys()
    )
    pair_geometry = {}
    for encoder in ENCODERS:
        pair_geometry[encoder] = {}
        for pair in pair_names:
            pair_geometry[encoder][pair] = {
                key: _stats([
                    float(
                        fold_results[fold]["encoders"][encoder]
                        ["pair_geometry"][pair][key]
                    )
                    for fold in FOLDS
                ])
                for key in (
                    "centroid_cosine_distance",
                    "left_mean_within_cosine_distance",
                    "right_mean_within_cosine_distance",
                    "inter_to_pooled_intra_ratio",
                )
            }

    pair_confusions = {}
    for encoder in ENCODERS:
        pair_confusions[encoder] = {}
        for decoder in DECODERS:
            pair_confusions[encoder][decoder] = {}
            for pair in pair_names:
                total = sum(
                    int(
                        fold_results[fold]["encoders"][encoder]
                        ["pair_confusions"][decoder][pair]["TOTAL"]
                    )
                    for fold in FOLDS
                )
                pair_confusions[encoder][decoder][pair] = total

    comparison = {
        "dinov2_minus_mobilenet": {
            decoder: {
                metric: _stats([
                    float(
                        fold_results[fold]["encoders"]["dinov2_small"]
                        ["metrics"][decoder][metric]
                    )
                    - float(
                        fold_results[fold]["encoders"]["mobilenetv3_imagenet"]
                        ["metrics"][decoder][metric]
                    )
                    for fold in FOLDS
                ])
                for metric in METRICS
            }
            for decoder in DECODERS
        },
        "convnext_minus_mobilenet": {
            decoder: {
                metric: _stats([
                    float(
                        fold_results[fold]["encoders"]["convnext_tiny_imagenet"]
                        ["metrics"][decoder][metric]
                    )
                    - float(
                        fold_results[fold]["encoders"]["mobilenetv3_imagenet"]
                        ["metrics"][decoder][metric]
                    )
                    for fold in FOLDS
                ])
                for metric in METRICS
            }
            for decoder in DECODERS
        },
    }

    geometry_delta = {}
    for pair in pair_names:
        dino = []
        convnext = []
        for fold in FOLDS:
            base = float(
                fold_results[fold]["encoders"]["mobilenetv3_imagenet"]
                ["pair_geometry"][pair]["inter_to_pooled_intra_ratio"]
            )
            dino.append(
                float(
                    fold_results[fold]["encoders"]["dinov2_small"]
                    ["pair_geometry"][pair]["inter_to_pooled_intra_ratio"]
                ) - base
            )
            convnext.append(
                float(
                    fold_results[fold]["encoders"]["convnext_tiny_imagenet"]
                    ["pair_geometry"][pair]["inter_to_pooled_intra_ratio"]
                ) - base
            )
        geometry_delta[pair] = {
            "dinov2_minus_mobilenet": _stats(dino),
            "dinov2_positive_folds": sum(v > 0.0 for v in dino),
            "convnext_minus_mobilenet": _stats(convnext),
            "convnext_positive_folds": sum(v > 0.0 for v in convnext),
        }

    sample_consensus = {
        "validation_observations": 0,
        "mobilenet_all_three_wrong": 0,
        "mobilenet_all_three_wrong_dino_majority_correct": 0,
        "mobilenet_all_three_wrong_dino_all_three_wrong": 0,
        "mobilenet_all_three_wrong_convnext_majority_correct": 0,
        "mobilenet_majority_correct_dino_all_three_wrong": 0,
    }
    persistent_rows = []

    for fold in FOLDS:
        rows = {
            encoder: {
                row["identity"]: row
                for row in _csv(
                    output_root
                    / f"fold_{fold}"
                    / encoder
                    / "sample_diagnostics.csv"
                )
            }
            for encoder in ENCODERS
        }
        identities = list(rows["mobilenetv3_imagenet"])
        if any(set(rows[encoder]) != set(identities) for encoder in ENCODERS):
            raise RuntimeError(f"Fold {fold}: sample identities antar encoder berbeda.")

        for identity in identities:
            m = rows["mobilenetv3_imagenet"][identity]
            d = rows["dinov2_small"][identity]
            c = rows["convnext_tiny_imagenet"][identity]
            sample_consensus["validation_observations"] += 1
            m_all_wrong = m["all_three_wrong"] == "1"
            d_all_wrong = d["all_three_wrong"] == "1"
            c_majority = _majority_correct(c)
            d_majority = _majority_correct(d)
            m_majority = _majority_correct(m)

            if m_all_wrong:
                sample_consensus["mobilenet_all_three_wrong"] += 1
                if d_majority:
                    sample_consensus[
                        "mobilenet_all_three_wrong_dino_majority_correct"
                    ] += 1
                if d_all_wrong:
                    sample_consensus[
                        "mobilenet_all_three_wrong_dino_all_three_wrong"
                    ] += 1
                    persistent_rows.append({
                        "fold": fold,
                        "identity": identity,
                        "actual": m["actual"],
                        "mobilenet_rival": m["nearest_rival_class"],
                        "dinov2_rival": d["nearest_rival_class"],
                        "mobilenet_knn_purity": float(m["knn_same_class_fraction"]),
                        "dinov2_knn_purity": float(d["knn_same_class_fraction"]),
                        "mobilenet_centroid_margin": float(
                            m["centroid_similarity_margin"]
                        ),
                        "dinov2_centroid_margin": float(
                            d["centroid_similarity_margin"]
                        ),
                    })
                if c_majority:
                    sample_consensus[
                        "mobilenet_all_three_wrong_convnext_majority_correct"
                    ] += 1

            if m_majority and d_all_wrong:
                sample_consensus[
                    "mobilenet_majority_correct_dino_all_three_wrong"
                ] += 1

    payload = {
        "format": "bilinear_lmmd.representation_geometry.summary.v1",
        "protocol": "coffee17-representation-geometry-audit-v1",
        "scope": (
            "five locked Coffee17 development folds; raw RGB; frozen encoders; "
            "no encoder fine-tuning; kNN/nearest-centroid/fixed-C linear probe; "
            "outer test untouched"
        ),
        "aggregate": aggregate,
        "pair_geometry": pair_geometry,
        "pair_geometry_delta": geometry_delta,
        "pair_confusion_totals": pair_confusions,
        "comparison": comparison,
        "sample_consensus": sample_consensus,
        "cross_representation_persistent_rows": persistent_rows,
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
