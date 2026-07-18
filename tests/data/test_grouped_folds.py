import json

import bilinear_lmmd.data.preparation.prepare_grouped_folds as grouped


def test_grouped_folds_keep_each_original_in_one_test_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(grouped, "EXPECTED_COUNTS", {"A": 5, "B": 5})
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        (source / split).mkdir(parents=True)
    for class_name in ("A", "B"):
        class_dir = source / "train" / class_name
        class_dir.mkdir()
        for index in range(5):
            (class_dir / f"image_{index}.jpg").write_bytes(b"image")

    output = tmp_path / "folds"
    grouped.prepare_grouped_folds(
        source, output, folds=5, seed=42, validation_ratio=0.1
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["total_originals"] == 10
    assert all(
        counts == {"train": 6, "val": 2, "test": 2}
        for counts in metadata["fold_counts"].values()
    )
    test_identities = [
        identity
        for assignments in metadata["assignments"].values()
        for identity in assignments["test"]
    ]
    assert len(test_identities) == len(set(test_identities)) == 10

