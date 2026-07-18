import torch

from bilinear_lmmd.engine.evaluate_checkpoint import CheckpointPredictions
from bilinear_lmmd.experiments.run_oof_ensemble import align_predictions, select_alpha


def _bundle(paths, labels, probabilities):
    return CheckpointPredictions(
        classes=["A", "B"],
        labels=labels,
        probabilities=torch.tensor(probabilities, dtype=torch.float32),
        paths=paths,
        hard_groups={"hard": ["A", "B"]},
    )


def test_alignment_uses_identity_instead_of_loader_order():
    gap = _bundle(
        ["/gap/A/a.jpg", "/gap/B/b.jpg"],
        [0, 1],
        [[0.9, 0.1], [0.2, 0.8]],
    )
    hbp = _bundle(
        ["/hbp/B/b.jpg", "/hbp/A/a.jpg"],
        [1, 0],
        [[0.1, 0.9], [0.8, 0.2]],
    )

    aligned = align_predictions(gap, hbp)

    assert aligned.identities == ["A/a.jpg", "B/b.jpg"]
    assert aligned.labels == [0, 1]
    assert aligned.hbp_probabilities.argmax(1).tolist() == [0, 1]


def test_select_alpha_finds_complementary_probability_mixture():
    aligned = align_predictions(
        _bundle(
            ["/gap/A/a.jpg", "/gap/A/b.jpg", "/gap/B/c.jpg", "/gap/B/d.jpg"],
            [0, 0, 1, 1],
            [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.1, 0.9]],
        ),
        _bundle(
            ["/hbp/A/a.jpg", "/hbp/A/b.jpg", "/hbp/B/c.jpg", "/hbp/B/d.jpg"],
            [0, 0, 1, 1],
            [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]],
        ),
    )

    alpha, curve = select_alpha(aligned, step=0.25)

    assert alpha == 0.5
    assert max(point["score"] for point in curve) == 1.0
