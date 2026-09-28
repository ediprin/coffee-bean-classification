from pathlib import Path

import pytest

from bilinear_lmmd.core.run_lock import exclusive_training_lock


def test_exclusive_lock_blocks_second_writer(tmp_path):
    with exclusive_training_lock(tmp_path, lock_name="same.lock"):
        with pytest.raises(RuntimeError, match="runtime lain"):
            with exclusive_training_lock(tmp_path, lock_name="same.lock"):
                pass
    assert not (tmp_path / "same.lock").exists()
