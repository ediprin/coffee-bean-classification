from pathlib import Path

import torch

from bilinear_lmmd.train import atomic_torch_save, load_resume_checkpoint


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
