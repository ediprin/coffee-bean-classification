from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_DIRECTORY = "Coffee_Bean_Classification"
PROJECT_MARKERS = ("artifact_index.json", "PROJECT_INDEX.md")
DEFAULT_DRIVE_SEARCH_ROOTS = (Path("/content/drive/MyDrive"),Path("/content/drive/.shortcut-targets-by-id"),)

def _candidate_rank(path: Path) -> tuple[int,int,str]:
    normalized=path.as_posix()
    if "/02_RISET_DAN_PROYEK/" in normalized: preference=0
    elif normalized.startswith("/content/drive/MyDrive/"): preference=1
    else: preference=2
    return preference,len(path.parts),normalized

def resolve_drive_project_root(search_roots: Iterable[str|Path]=DEFAULT_DRIVE_SEARCH_ROOTS,*,required_relative_paths: Iterable[str|Path]=()) -> Path:
    roots=[Path(value).expanduser() for value in search_roots]
    required=tuple(Path(value) for value in required_relative_paths)
    candidates: dict[str,Path]={}
    for root in roots:
        if not root.is_dir(): continue
        direct=root/PROJECT_DIRECTORY
        if direct.is_dir() and (all((direct/marker).is_file() for marker in PROJECT_MARKERS) or (required and all((direct/item).is_file() for item in required))): candidates[direct.as_posix()]=direct
        for marker in root.rglob(PROJECT_MARKERS[0]):
            candidate=marker.parent
            if all((candidate/name).is_file() for name in PROJECT_MARKERS): candidates[candidate.as_posix()]=candidate
    if not candidates and required:
        for anchor in required:
            for root in roots:
                if not root.is_dir(): continue
                for match in root.rglob(anchor.name):
                    if not match.is_file(): continue
                    candidate=match
                    for _ in anchor.parts: candidate=candidate.parent
                    if all((candidate/item).is_file() for item in required): candidates[candidate.as_posix()]=candidate
    if not candidates: raise FileNotFoundError("Folder Coffee_Bean_Classification yang bertanda artifact_index.json dan PROJECT_INDEX.md tidak ditemukan. Resolver tidak membuat root baru.")
    return sorted(candidates.values(),key=_candidate_rank)[0]
