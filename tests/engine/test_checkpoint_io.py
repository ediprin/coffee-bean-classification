from pathlib import Path

import torch

from bilinear_lmmd.engine.train import (
    ExponentialMovingAverage,
    atomic_torch_save,
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
