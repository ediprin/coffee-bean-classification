from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage import morphology, transform
from tqdm import tqdm

from bilinear_lmmd.data.attribute_features import (
    FEATURE_VERSION,
    extract_attributes,
    feature_group_indices,
    load_rgb,
    segment_bean,
)
from bilinear_lmmd.core.config import DEFAULTS
from bilinear_lmmd.engine.train import classification_metrics


COMBINATIONS: dict[str, tuple[str, ...]] = {
    "C": ("color",),
    "S": ("shape",),
    "T": ("texture",),
    "CS": ("color", "shape"),
    "CT": ("color", "texture"),
    "ST": ("shape", "texture"),
    "CST": ("color", "shape", "texture"),
}


@dataclass(frozen=True)
class FeatureCache:
    identities: list[str]
    labels: list[str]
    features: np.ndarray
    feature_names: list[str]
    mask_area_fractions: np.ndarray


def _identity(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _split_paths(data_root: Path, fold: int, split: str) -> dict[str, Path]:
    split_root = data_root / f"fold_{fold}" / "source" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split tidak ditemukan: {split_root}")
    result: dict[str, Path] = {}
    for path in sorted(split_root.glob("*/*")):
        if not path.is_file():
            continue
        identity = _identity(path)
        if identity in result:
            raise ValueError(f"Identitas duplikat pada {split_root}: {identity}")
        result[identity] = path
    return result


def _all_fold_one_paths(data_root: Path, expected_count: int) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for identity, path in _split_paths(data_root, 1, split).items():
            if identity in result:
                raise ValueError(f"Identitas duplikat di fold 1: {identity}")
            result[identity] = path
    if len(result) != expected_count:
        raise ValueError(
            f"Fold 1 harus berisi {expected_count} citra, ditemukan {len(result)}."
        )
    return result


def _save_mask_audit(paths: dict[str, Path], destination: Path) -> None:
    selected: list[tuple[str, Path]] = []
    seen_classes: set[str] = set()
    for identity, path in sorted(paths.items()):
        class_name = identity.split("/", 1)[0]
        if class_name not in seen_classes:
            selected.append((class_name, path))
            seen_classes.add(class_name)

    tile_size = 180
    caption_height = 28
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + caption_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (class_name, path) in enumerate(selected):
        rgb = load_rgb(path)
        mask = segment_bean(rgb)
        boundary = morphology.dilation(mask, morphology.disk(2)) ^ morphology.erosion(
            mask, morphology.disk(2)
        )
        overlay = rgb.copy()
        overlay[mask] = 0.82 * overlay[mask] + 0.18 * np.asarray(
            [0.0, 1.0, 0.0], dtype=np.float32
        )
        overlay[boundary] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        resized = transform.resize(
            overlay,
            (tile_size, tile_size),
            preserve_range=True,
            anti_aliasing=True,
        )
        tile = Image.fromarray(np.clip(resized * 255.0, 0, 255).astype(np.uint8))
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + caption_height)
        canvas.paste(tile, (x, y))
        draw.text((x + 4, y + tile_size + 5), class_name, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _save_cache(cache: FeatureCache, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        version=np.asarray([FEATURE_VERSION], dtype=np.int64),
        identities=np.asarray(cache.identities),
        labels=np.asarray(cache.labels),
        features=cache.features,
        feature_names=np.asarray(cache.feature_names),
        mask_area_fractions=cache.mask_area_fractions,
    )


def _load_cache(path: Path, expected_count: int) -> FeatureCache | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["version"][0]) != FEATURE_VERSION:
            return None
        identities = payload["identities"].tolist()
        if len(identities) != expected_count:
            return None
        return FeatureCache(
            identities=identities,
            labels=payload["labels"].tolist(),
            features=payload["features"],
            feature_names=payload["feature_names"].tolist(),
            mask_area_fractions=payload["mask_area_fractions"],
        )


def build_feature_cache(
    data_root: Path,
    output_root: Path,
    expected_count: int,
    workers: int = 4,
) -> FeatureCache:
    cache_path = output_root / "cache" / f"attribute_features_v{FEATURE_VERSION}.npz"
    cached = _load_cache(cache_path, expected_count)
    if cached is not None:
        print(f"SKIP ekstraksi fitur; cache ditemukan: {cache_path}", flush=True)
        audit_path = output_root / "mask_audit.png"
        areas_path = output_root / "mask_areas.csv"
        if not audit_path.is_file() or not areas_path.is_file():
            paths = _all_fold_one_paths(data_root, expected_count)
            _save_mask_audit(paths, audit_path)
            with areas_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("identity", "class", "mask_area_fraction"))
                for identity, label, fraction in zip(
                    cached.identities,
                    cached.labels,
                    cached.mask_area_fractions,
                ):
                    writer.writerow((identity, label, fraction))
        return cached

    paths = _all_fold_one_paths(data_root, expected_count)
    identities = sorted(paths)
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    areas: list[float] = []
    feature_names: list[str] | None = None

    def extract(identity: str):
        return extract_attributes(paths[identity])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        extracted_items = executor.map(extract, identities)
        iterator = tqdm(
            extracted_items,
            total=len(identities),
            desc="extract color/shape/texture",
        )
        for identity, extracted in zip(identities, iterator):
            if feature_names is None:
                feature_names = extracted.names
            elif feature_names != extracted.names:
                raise RuntimeError("Urutan fitur berubah antar-citra.")
            labels.append(identity.split("/", 1)[0])
            vectors.append(extracted.values)
            areas.append(extracted.mask_area_fraction)
    if feature_names is None:
        raise RuntimeError("Tidak ada fitur yang diekstrak.")
    cache = FeatureCache(
        identities=identities,
        labels=labels,
        features=np.vstack(vectors),
        feature_names=feature_names,
        mask_area_fractions=np.asarray(areas, dtype=np.float64),
    )
    _save_cache(cache, cache_path)
    _save_mask_audit(paths, output_root / "mask_audit.png")
    with (output_root / "mask_areas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("identity", "class", "mask_area_fraction"))
        for identity, label, fraction in zip(
            cache.identities, cache.labels, cache.mask_area_fractions
        ):
            writer.writerow((identity, label, fraction))
    print(f"SAVED mask audit: {output_root / 'mask_audit.png'}", flush=True)
    return cache


def _fit_select_svm(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
) -> tuple[Pipeline, dict, list[dict]]:
    candidates = [
        (c_value, gamma)
        for c_value in (0.1, 1.0, 10.0, 100.0)
        for gamma in ("scale", 0.001, 0.01, 0.1)
    ]
    best_model: Pipeline | None = None
    best_result: dict | None = None
    curve: list[dict] = []
    for c_value, gamma in candidates:
        model = Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "svm",
                    SVC(
                        C=c_value,
                        gamma=gamma,
                        kernel="rbf",
                        class_weight="balanced",
                        cache_size=2000,
                    ),
                ),
            )
        )
        model.fit(train_features, train_labels)
        predictions = model.predict(val_features)
        score = f1_score(
            val_labels,
            predictions,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
        result = {"C": c_value, "gamma": gamma, "validation_macro_f1": score}
        curve.append(result)
        if best_result is None or score > best_result["validation_macro_f1"] + 1e-12:
            best_model = model
            best_result = result
    if best_model is None or best_result is None:
        raise RuntimeError("Pemilihan SVM tidak menghasilkan model.")
    return best_model, best_result, curve


def _indices_for_identities(
    identities: list[str], identity_to_index: dict[str, int]
) -> np.ndarray:
    missing = [identity for identity in identities if identity not in identity_to_index]
    if missing:
        raise ValueError(f"Identitas tidak ada dalam feature cache: {missing[:3]}")
    return np.asarray([identity_to_index[identity] for identity in identities])


def _evaluate_combination(
    code: str,
    groups: tuple[str, ...],
    data_root: Path,
    output_root: Path,
    cache: FeatureCache,
    folds: int,
    classes: list[str],
) -> dict:
    combination_dir = output_root / code
    metrics_path = combination_dir / "metrics.json"
    predictions_path = combination_dir / "predictions.csv"
    if metrics_path.is_file() and predictions_path.is_file():
        completed = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            completed.get("feature_version") == FEATURE_VERSION
            and completed.get("feature_groups") == list(groups)
            and completed.get("sample_count") == len(cache.identities)
            and len(completed.get("folds", [])) == folds
        ):
            print(f"SKIP {code}; hasil lengkap ditemukan: {metrics_path}", flush=True)
            return completed

    group_indices = feature_group_indices(cache.feature_names)
    selected_indices = np.concatenate([group_indices[group] for group in groups])
    features = cache.features[:, selected_indices]
    class_to_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray([class_to_index[name] for name in cache.labels])
    identity_to_index = {
        identity: index for index, identity in enumerate(cache.identities)
    }
    hard_groups = DEFAULTS["evaluation"]["hard_groups"]
    oof_rows: list[dict] = []
    oof_labels: list[int] = []
    oof_predictions: list[int] = []
    seen: set[str] = set()
    fold_results: list[dict] = []

    print(
        f"\n=== {code}: {'+'.join(groups)} ({features.shape[1]} fitur) ===",
        flush=True,
    )
    for fold in range(1, folds + 1):
        split_identities = {
            split: sorted(_split_paths(data_root, fold, split))
            for split in ("train", "val", "test")
        }
        indices = {
            split: _indices_for_identities(items, identity_to_index)
            for split, items in split_identities.items()
        }
        model, selected, curve = _fit_select_svm(
            features[indices["train"]],
            labels[indices["train"]],
            features[indices["val"]],
            labels[indices["val"]],
            len(classes),
        )
        test_indices = indices["test"]
        predictions = model.predict(features[test_indices]).astype(int)
        test_labels = labels[test_indices].astype(int)
        test_metrics = classification_metrics(
            test_labels.tolist(), predictions.tolist(), classes, hard_groups
        )
        fold_results.append(
            {
                "fold": fold,
                "selected": selected,
                "validation_curve": curve,
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        print(
            f"fold {fold}: C={selected['C']} gamma={selected['gamma']} "
            f"val={selected['validation_macro_f1']:.2%} "
            f"test={test_metrics['macro_f1']:.2%}",
            flush=True,
        )
        for local_index, identity in enumerate(split_identities["test"]):
            if identity in seen:
                raise ValueError(f"Prediksi OOF duplikat: {identity}")
            seen.add(identity)
            actual = int(test_labels[local_index])
            predicted = int(predictions[local_index])
            oof_labels.append(actual)
            oof_predictions.append(predicted)
            oof_rows.append(
                {
                    "identity": identity,
                    "fold": fold,
                    "actual": classes[actual],
                    "predicted": classes[predicted],
                    "correct": int(actual == predicted),
                }
            )

    if len(seen) != len(cache.identities):
        raise ValueError(
            f"OOF {code} hanya memiliki {len(seen)} dari {len(cache.identities)} citra."
        )
    metrics = classification_metrics(
        oof_labels, oof_predictions, classes, hard_groups
    )
    metrics.update(
        {
            "feature_version": FEATURE_VERSION,
            "code": code,
            "feature_groups": list(groups),
            "feature_count": int(features.shape[1]),
            "sample_count": len(oof_labels),
            "classes": classes,
            "folds": fold_results,
        }
    )
    combination_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("identity", "fold", "actual", "predicted", "correct"),
        )
        writer.writeheader()
        writer.writerows(sorted(oof_rows, key=lambda row: row["identity"]))
    return metrics


def compare_with_hbp(
    attribute_predictions_path: Path,
    hbp_predictions_path: Path,
    output_path: Path,
) -> dict:
    def read(path: Path) -> dict[str, dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = {row["identity"]: row for row in csv.DictReader(handle)}
        if not rows:
            raise ValueError(f"Predictions kosong: {path}")
        return rows

    attribute = read(attribute_predictions_path)
    hbp = read(hbp_predictions_path)
    if attribute.keys() != hbp.keys():
        raise ValueError("Identitas attribute dan HBP berbeda.")
    counts = {"both_correct": 0, "attribute_only": 0, "hbp_only": 0, "both_wrong": 0}
    per_class: dict[str, dict[str, int]] = {}
    for identity in sorted(attribute):
        left = attribute[identity]
        right = hbp[identity]
        if left["actual"] != right["actual"]:
            raise ValueError(f"Label aktual berbeda untuk {identity}.")
        actual = left["actual"]
        attribute_correct = left["predicted"] == actual
        hbp_correct = right["predicted"] == actual
        if attribute_correct and hbp_correct:
            key = "both_correct"
        elif attribute_correct:
            key = "attribute_only"
        elif hbp_correct:
            key = "hbp_only"
        else:
            key = "both_wrong"
        counts[key] += 1
        class_counts = per_class.setdefault(
            actual,
            {"both_correct": 0, "attribute_only": 0, "hbp_only": 0, "both_wrong": 0},
        )
        class_counts[key] += 1
    total = len(attribute)
    result = {
        "sample_count": total,
        **counts,
        "oracle_accuracy": (
            counts["both_correct"] + counts["attribute_only"] + counts["hbp_only"]
        )
        / total,
        "per_class": per_class,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_attribute_ablation(
    data_root: Path,
    output_root: Path,
    folds: int = 5,
    expected_count: int = 979,
    hbp_predictions: Path | None = None,
    workers: int = 4,
) -> dict:
    cache = build_feature_cache(data_root, output_root, expected_count, workers)
    classes = sorted(set(cache.labels))
    results = {
        code: _evaluate_combination(
            code, groups, data_root, output_root, cache, folds, classes
        )
        for code, groups in COMBINATIONS.items()
    }
    ranking = sorted(
        (
            {
                "code": code,
                "groups": list(COMBINATIONS[code]),
                "feature_count": metrics["feature_count"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "hard_class_f1": metrics["hard_class_f1"],
                "worst_class_f1": metrics["worst_class_f1"],
            }
            for code, metrics in results.items()
        ),
        key=lambda row: row["macro_f1"],
        reverse=True,
    )
    summary = {
        "method": "grouped OOF attribute ablation with validation-tuned RBF SVM",
        "sample_count": expected_count,
        "feature_version": FEATURE_VERSION,
        "ranking": ranking,
    }
    best_code = ranking[0]["code"]
    if hbp_predictions is not None:
        complementarity = compare_with_hbp(
            output_root / best_code / "predictions.csv",
            hbp_predictions,
            output_root / f"{best_code}_vs_HBP_complementarity.json",
        )
        summary["hbp_complementarity"] = complementarity
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== ATTRIBUTE OOF RANKING ===")
    for row in ranking:
        print(
            f"{row['code']:3s} ({'+'.join(row['groups']):19s}) "
            f"Macro={row['macro_f1']:.2%} "
            f"Hard={row['hard_class_f1']:.2%} "
            f"Worst={row['worst_class_f1']:.2%}"
        )
    print(f"BEST: {best_code}; SAVED: {output_root}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped 5-fold ablation seluruh kombinasi warna/bentuk/tekstur"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--expected-count", type=int, default=979)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--hbp-predictions",
        type=Path,
        help="Optional merged HBP OOF predictions.csv untuk audit komplementaritas.",
    )
    args = parser.parse_args()
    run_attribute_ablation(
        data_root=args.data_root,
        output_root=args.output_root,
        folds=args.folds,
        expected_count=args.expected_count,
        hbp_predictions=args.hbp_predictions,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
