import csv
import json

import numpy as np
from PIL import Image, ImageDraw

from bilinear_lmmd.attribute_features import (
    extract_attributes,
    feature_group_indices,
)
from bilinear_lmmd.run_attribute_ablation import (
    COMBINATIONS,
    FeatureCache,
    _evaluate_combination,
    _fit_select_svm,
    compare_with_hbp,
)
from bilinear_lmmd.attribute_features import FEATURE_VERSION


def _synthetic_bean(path):
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 34, 106, 96), fill=(105, 140, 82))
    draw.line((35, 65, 95, 65), fill=(55, 75, 42), width=3)
    image.save(path)


def test_attribute_extraction_segments_bean_and_builds_all_groups(tmp_path):
    image_path = tmp_path / "bean.png"
    _synthetic_bean(image_path)

    extracted = extract_attributes(image_path)
    groups = feature_group_indices(extracted.names)

    assert 0.20 < extracted.mask_area_fraction < 0.30
    assert {name: len(indices) for name, indices in groups.items()} == {
        "color": 42,
        "shape": 18,
        "texture": 50,
    }
    assert extracted.values.shape == (110,)
    assert np.isfinite(extracted.values).all()


def test_attribute_ablation_contains_every_nonempty_combination():
    assert COMBINATIONS == {
        "C": ("color",),
        "S": ("shape",),
        "T": ("texture",),
        "CS": ("color", "shape"),
        "CT": ("color", "texture"),
        "ST": ("shape", "texture"),
        "CST": ("color", "shape", "texture"),
    }


def test_svm_hyperparameters_are_selected_on_validation_data():
    train_features = np.asarray(
        [[-2.0], [-1.5], [0.0], [0.2], [1.5], [2.0]], dtype=np.float64
    )
    train_labels = np.asarray([0, 0, 1, 1, 2, 2])
    val_features = np.asarray([[-1.8], [0.1], [1.8]], dtype=np.float64)
    val_labels = np.asarray([0, 1, 2])

    model, selected, curve = _fit_select_svm(
        train_features, train_labels, val_features, val_labels, num_classes=3
    )

    assert selected["validation_macro_f1"] == 1.0
    assert len(curve) == 16
    assert model.predict(val_features).tolist() == val_labels.tolist()


def _write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("identity", "actual", "predicted")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_attribute_hbp_complementarity_counts_disagreements(tmp_path):
    attribute_path = tmp_path / "attribute.csv"
    hbp_path = tmp_path / "hbp.csv"
    output_path = tmp_path / "comparison.json"
    _write_predictions(
        attribute_path,
        [
            {"identity": "A/a.jpg", "actual": "A", "predicted": "A"},
            {"identity": "A/b.jpg", "actual": "A", "predicted": "B"},
            {"identity": "B/c.jpg", "actual": "B", "predicted": "B"},
            {"identity": "B/d.jpg", "actual": "B", "predicted": "A"},
        ],
    )
    _write_predictions(
        hbp_path,
        [
            {"identity": "A/a.jpg", "actual": "A", "predicted": "A"},
            {"identity": "A/b.jpg", "actual": "A", "predicted": "A"},
            {"identity": "B/c.jpg", "actual": "B", "predicted": "A"},
            {"identity": "B/d.jpg", "actual": "B", "predicted": "A"},
        ],
    )

    result = compare_with_hbp(attribute_path, hbp_path, output_path)

    assert result["both_correct"] == 1
    assert result["attribute_only"] == 1
    assert result["hbp_only"] == 1
    assert result["both_wrong"] == 1
    assert result["oracle_accuracy"] == 0.75
    assert output_path.is_file()


def test_completed_combination_is_resumed_without_recomputation(tmp_path):
    combination_dir = tmp_path / "results" / "C"
    combination_dir.mkdir(parents=True)
    completed = {
        "feature_version": FEATURE_VERSION,
        "feature_groups": ["color"],
        "sample_count": 1,
        "folds": [{"fold": 1}],
        "macro_f1": 0.5,
    }
    (combination_dir / "metrics.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    (combination_dir / "predictions.csv").write_text(
        "identity,fold,actual,predicted,correct\nA/a.jpg,1,A,A,1\n",
        encoding="utf-8",
    )
    cache = FeatureCache(
        identities=["A/a.jpg"],
        labels=["A"],
        features=np.zeros((1, 1)),
        feature_names=["color_dummy"],
        mask_area_fractions=np.asarray([0.25]),
    )

    result = _evaluate_combination(
        "C",
        ("color",),
        tmp_path / "missing-data-root",
        tmp_path / "results",
        cache,
        folds=1,
        classes=["A"],
    )

    assert result == completed
