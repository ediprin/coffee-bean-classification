from __future__ import annotations

from pathlib import Path

import pytest

from bilinear_lmmd import artifact_store


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("outputs/MV4_seed123", "outputs/MV4_seed123"),
        (r"outputs\MV4_seed123", "outputs/MV4_seed123"),
        ("./val_reports/MV4_seed123", "val_reports/MV4_seed123"),
    ],
)
def test_normalize_remote_path(raw: str, expected: str) -> None:
    assert artifact_store.normalize_remote_path(raw) == expected


@pytest.mark.parametrize("path", ("", "/absolute", "../escape", "a/../../escape"))
def test_normalize_remote_path_rejects_unsafe_values(path: str) -> None:
    with pytest.raises(ValueError):
        artifact_store.normalize_remote_path(path)


def test_sync_artifacts_uploads_only_available_files(tmp_path, monkeypatch) -> None:
    (tmp_path / "last.pt").write_bytes(b"checkpoint")
    calls = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(artifact_store, "HfApi", FakeApi)
    uploaded = artifact_store.sync_artifacts(
        "user/repo",
        "outputs/run",
        tmp_path,
        filenames=("last.pt", "best.pt"),
        commit_message="checkpoint",
    )

    assert uploaded is True
    assert calls[0]["allow_patterns"] == ["last.pt"]
    assert calls[0]["path_in_repo"] == "outputs/run"


def test_restore_artifacts_copies_remote_file_atomically(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache-last.pt"
    cache.write_bytes(b"remote-state")
    destination = tmp_path / "run"

    def fake_download(**kwargs):
        assert kwargs["filename"] == "outputs/run/last.pt"
        return str(cache)

    monkeypatch.setattr(artifact_store, "hf_hub_download", fake_download)
    restored = artifact_store.restore_artifacts(
        "user/repo",
        "outputs/run",
        destination,
        filenames=("last.pt",),
    )

    assert restored == [destination / "last.pt"]
    assert (destination / "last.pt").read_bytes() == b"remote-state"
    assert not (destination / "last.pt.download").exists()
