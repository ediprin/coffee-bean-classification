import csv

from bilinear_lmmd.run_disagreement_gate import (
    GateRow,
    apply_pair_rules,
    cross_fitted_gate,
    fit_pair_rules,
    load_gate_rows,
)


def test_pair_rule_chooses_attribute_only_for_supported_advantage():
    rows = [
        GateRow("1", 1, "A", "B", "A"),
        GateRow("2", 1, "A", "B", "A"),
        GateRow("3", 1, "B", "B", "A"),
        GateRow("4", 1, "C", "B", "A"),
    ]

    rules = fit_pair_rules(rows, min_support=3, margin=0.0)
    predictions, decisions = apply_pair_rules(rows, rules)

    assert rules[("B", "A")]["choose"] == "attribute"
    assert predictions == ["A", "A", "A", "A"]
    assert decisions == ["attribute"] * 4


def test_sparse_or_tied_pair_falls_back_to_hbp():
    rows = [
        GateRow("1", 1, "A", "B", "A"),
        GateRow("2", 1, "B", "B", "A"),
    ]

    rules = fit_pair_rules(rows, min_support=3, margin=0.0)
    predictions, decisions = apply_pair_rules(rows, rules)

    assert predictions == ["B", "B"]
    assert decisions == ["hbp", "hbp"]


def test_cross_fitted_prediction_for_fold_does_not_use_that_fold_labels():
    rows = []
    for fold in range(1, 6):
        rows.extend(
            [
                GateRow(f"{fold}-1", fold, "A", "B", "A"),
                GateRow(f"{fold}-2", fold, "A", "B", "A"),
                GateRow(f"{fold}-3", fold, "B", "B", "A"),
                GateRow(f"{fold}-4", fold, "C", "C", "C"),
            ]
        )
    classes = ["A", "B", "C"]

    original, _ = cross_fitted_gate(rows, classes, {})
    changed = [
        GateRow(row.identity, row.fold, "C", row.hbp, row.attribute)
        if row.fold == 5 and row.identity == "5-1"
        else row
        for row in rows
    ]
    modified, _ = cross_fitted_gate(changed, classes, {})

    original_fold = {
        row["identity"]: row["gate"] for row in original if row["fold"] == 5
    }
    modified_fold = {
        row["identity"]: row["gate"] for row in modified if row["fold"] == 5
    }
    assert original_fold == modified_fold


def _write(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_gate_rows_matches_by_identity(tmp_path):
    attribute = tmp_path / "attribute.csv"
    hbp = tmp_path / "hbp.csv"
    _write(
        attribute,
        ("identity", "fold", "actual", "predicted"),
        [
            {"identity": "B/b.jpg", "fold": 2, "actual": "B", "predicted": "A"},
            {"identity": "A/a.jpg", "fold": 1, "actual": "A", "predicted": "A"},
        ],
    )
    _write(
        hbp,
        ("identity", "actual", "predicted"),
        [
            {"identity": "A/a.jpg", "actual": "A", "predicted": "B"},
            {"identity": "B/b.jpg", "actual": "B", "predicted": "B"},
        ],
    )

    rows = load_gate_rows(attribute, hbp)

    assert [row.identity for row in rows] == ["A/a.jpg", "B/b.jpg"]
    assert rows[0] == GateRow("A/a.jpg", 1, "A", "B", "A")
