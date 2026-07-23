from pathlib import Path

import torch
from torch import nn

from bilinear_lmmd.engine.train import (
    configure_trainable_modules,
    load_initialization_checkpoint,
    set_selective_training_mode,
)


class TinyWarmStart(nn.Module):
    def __init__(self, with_residual: bool):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        self.fusion = nn.Linear(4, 4)
        self.flat_classifier = nn.Linear(4, 2)
        self.bean_pool = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.bean_residual_classifier = nn.Linear(4, 1) if with_residual else None


def test_warm_start_loads_shared_state_and_selective_mode_freezes_base(tmp_path: Path):
    source = TinyWarmStart(with_residual=False)
    checkpoint = tmp_path / "base.pt"
    torch.save(
        {"model": source.state_dict(), "classes": ["a", "b"], "epoch": 7},
        checkpoint,
    )
    candidate = TinyWarmStart(with_residual=True)
    audit = load_initialization_checkpoint(
        candidate,
        checkpoint,
        ["a", "b"],
        ("encoder.", "fusion.", "flat_classifier."),
    )
    assert audit["source_epoch"] == 7
    torch.testing.assert_close(
        candidate.flat_classifier.weight, source.flat_classifier.weight
    )

    selected = configure_trainable_modules(
        candidate, ["bean_pool", "bean_residual_classifier"]
    )
    set_selective_training_mode(candidate, selected)
    assert candidate.encoder.training is False
    assert candidate.encoder[1].training is False
    assert candidate.flat_classifier.training is False
    assert candidate.bean_pool.training is True
    assert candidate.bean_pool[1].training is True
    assert all(not parameter.requires_grad for parameter in candidate.encoder.parameters())
    assert all(parameter.requires_grad for parameter in candidate.bean_pool.parameters())
