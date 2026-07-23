from pathlib import Path

import torch

from bilinear_lmmd.engine.train import (
    ExponentialMovingAverage,
    accumulate_gate_sum,
    atomic_torch_save,
    gate_stage_means,
    load_resume_checkpoint,
    prepare_images,
)


def test_atomic_torch_save_round_trip_and_removes_temporary(tmp_path: Path):
    destination = tmp_path / "last.pt"
    atomic_torch_save({"epoch": 3, "tensor": torch.arange(4)}, destination)

    loaded = torch.load(destination, weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["tensor"], torch.arange(4))
    assert not (tmp_path / "last.pt.tmp").exists()


def test_corrupt_resume_checkpoint_is_ignored(tmp_path: Path, capsys):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"interrupted checkpoint")

    loaded = load_resume_checkpoint(checkpoint, torch.device("cpu"))

    assert loaded is None
    assert "Training dimulai ulang" in capsys.readouterr().out


def test_prepare_images_can_use_channels_last_without_changing_values():
    images = torch.randn(2, 3, 8, 8)
    prepared = prepare_images(
        images,
        torch.device("cpu"),
        non_blocking=True,
        channels_last=True,
    )

    assert prepared.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(prepared, images)


def test_ema_averages_parameters_and_copies_buffers():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.BatchNorm1d(2),
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[1].running_mean.fill_(2.0)
    ema = ExponentialMovingAverage(model, decay=0.5)

    with torch.no_grad():
        model[0].weight.fill_(3.0)
        model[1].running_mean.fill_(7.0)
    ema.update(model)

    assert torch.allclose(ema.model[0].weight, torch.full((2, 2), 2.0))
    assert torch.equal(ema.model[1].running_mean, torch.full((2,), 7.0))
    assert not any(parameter.requires_grad for parameter in ema.model.parameters())


def test_gate_logging_supports_stage_channel_weights():
    first = torch.zeros(2, 3, 4)
    first[:, 0] = 0.2
    first[:, 1] = 0.3
    first[:, 2] = 0.5
    second = first.clone()

    total = accumulate_gate_sum(None, first)
    total = accumulate_gate_sum(total, second)
    compact = gate_stage_means(total, sample_count=4)

    assert total.shape == (3, 4)
    torch.testing.assert_close(compact, torch.tensor([0.2, 0.3, 0.5]))


def test_gate_logging_preserves_legacy_two_expert_shape():
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    total = accumulate_gate_sum(None, weights)
    compact = gate_stage_means(total, sample_count=2)

    torch.testing.assert_close(compact, torch.tensor([0.375, 0.625]))
