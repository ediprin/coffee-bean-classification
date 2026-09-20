from __future__ import annotations

import torch
from torch import nn

from bilinear_lmmd.experiments.run_preprocessing_static_preflight import _tensor_fingerprint


def test_state_fingerprint_is_deterministic_and_sensitive():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    first = _tensor_fingerprint(model.state_dict())
    second = _tensor_fingerprint(model.state_dict())
    assert first == second
    with torch.no_grad():
        model[0].weight[0, 0] += 1.0
    assert _tensor_fingerprint(model.state_dict()) != first
