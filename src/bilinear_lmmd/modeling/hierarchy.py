from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParentHierarchy:
    parent_names: tuple[str, ...]
    fine_to_parent: tuple[int, ...]


def build_parent_hierarchy(
    class_names: list[str],
    groups: dict[str, list[str]],
) -> ParentHierarchy:
    """Validate a complete fine-class partition and build index mapping."""

    if not groups:
        raise ValueError("hierarchy.groups tidak boleh kosong.")
    known = set(class_names)
    assignments: dict[str, int] = {}
    parent_names: list[str] = []
    for parent_index, (parent_name, members) in enumerate(groups.items()):
        if not parent_name or not members:
            raise ValueError("Setiap parent hierarchy harus bernama dan memiliki anggota.")
        parent_names.append(parent_name)
        for member in members:
            if member not in known:
                raise ValueError(f"Kelas hierarchy tidak ditemukan: {member}")
            if member in assignments:
                raise ValueError(f"Kelas hierarchy muncul lebih dari sekali: {member}")
            assignments[member] = parent_index

    missing = [name for name in class_names if name not in assignments]
    if missing:
        raise ValueError(f"Kelas belum memiliki parent hierarchy: {missing}")
    return ParentHierarchy(
        parent_names=tuple(parent_names),
        fine_to_parent=tuple(assignments[name] for name in class_names),
    )
