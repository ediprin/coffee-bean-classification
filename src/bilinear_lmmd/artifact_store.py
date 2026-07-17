from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download


TRAINING_ARTIFACTS = (
    "last.pt",
    "best.pt",
    "history.json",
    "resolved_config.json",
    "artifact_manifest.json",
)


def normalize_remote_path(path: str) -> str:
    normalized = PurePosixPath(path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Path artefak Hugging Face tidak aman: {path!r}")
    value = normalized.as_posix().strip("./")
    if not value:
        raise ValueError("Path artefak Hugging Face tidak boleh kosong.")
    return value


def ensure_artifact_repo(repo_id: str, private: bool = True) -> None:
    """Create or validate a model repository before expensive training starts."""

    HfApi().create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )


def restore_artifacts(
    repo_id: str,
    remote_path: str,
    local_dir: Path,
    filenames: Iterable[str] = TRAINING_ARTIFACTS,
    *,
    overwrite: bool = True,
) -> list[Path]:
    """Restore known files from a Hub model repo.

    Missing files are normal for a new run. Authentication and connectivity
    errors are surfaced as warnings by the caller, while successfully fetched
    files are copied atomically into the requested run directory.
    """

    remote_path = normalize_remote_path(remote_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    restored: list[Path] = []
    for filename in filenames:
        destination = local_dir / filename
        if destination.is_file() and not overwrite:
            continue
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="model",
                    filename=f"{remote_path}/{filename}",
                )
            )
        except Exception as exc:
            # A run may not exist remotely yet. Do not print one warning per
            # absent optional file; the caller reports the restore count.
            if getattr(exc, "response", None) is not None and getattr(
                exc.response, "status_code", None
            ) == 404:
                continue
            message = str(exc).lower()
            if "entry not found" in message or "404" in message:
                continue
            raise
        temporary = destination.with_name(f"{destination.name}.download")
        shutil.copy2(cached, temporary)
        temporary.replace(destination)
        restored.append(destination)
    return restored


def sync_artifacts(
    repo_id: str,
    remote_path: str,
    local_dir: Path,
    filenames: Iterable[str] = TRAINING_ARTIFACTS,
    *,
    commit_message: str,
) -> bool:
    """Upload one atomic Hub commit containing the currently available files."""

    remote_path = normalize_remote_path(remote_path)
    available = [name for name in filenames if (local_dir / name).is_file()]
    if not available:
        return False
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=remote_path,
        allow_patterns=available,
        commit_message=commit_message,
    )
    return True
