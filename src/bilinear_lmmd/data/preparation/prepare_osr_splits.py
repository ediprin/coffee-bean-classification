from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _images(class_root: Path) -> list[Path]:
    return sorted(
        path
        for path in class_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_protocol(protocol_path: Path) -> tuple[dict, str]:
    payload = protocol_path.read_bytes()
    protocol = yaml.safe_load(payload) or {}
    if protocol.get("status") != "frozen":
        raise ValueError("Protokol OSR harus berstatus frozen sebelum preparasi.")
    splits = protocol.get("splits", {})
    if set(splits) != {"near", "medium", "far"}:
        raise ValueError("Protokol harus memiliki split near, medium, dan far.")
    expected_unknown = int(protocol["dataset"]["unknown_classes_per_split"])
    for name, split in splits.items():
        unknown = list(split.get("unknown_classes", []))
        if len(unknown) != expected_unknown or len(set(unknown)) != len(unknown):
            raise ValueError(
                f"Split {name} harus memiliki {expected_unknown} unknown unik."
            )
    return protocol, hashlib.sha256(payload).hexdigest()


def _inspect_source(source_root: Path) -> tuple[list[str], dict[str, dict[str, list[Path]]]]:
    collected: dict[str, dict[str, list[Path]]] = {}
    split_classes: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_root = source_root / "source" / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split Coffee17 tidak ditemukan: {split_root}")
        classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        if not classes:
            raise ValueError(f"Tidak ada kelas pada {split_root}.")
        split_classes[split] = classes
        collected[split] = {
            name: _images(split_root / name)
            for name in classes
        }
        empty = [name for name, paths in collected[split].items() if not paths]
        if empty:
            raise ValueError(f"Kelas kosong pada {split}: {empty}")
    if not (
        split_classes["train"]
        == split_classes["val"]
        == split_classes["test"]
    ):
        raise ValueError("Daftar kelas Coffee17 berbeda antar train/val/test.")
    return split_classes["train"], collected


def prepare_osr_splits(
    source_root: Path,
    output_root: Path,
    protocol_path: Path,
) -> dict:
    """Create leakage-safe known/unknown views of one grouped Coffee17 fold."""

    source_root = source_root.resolve()
    protocol_path = protocol_path.resolve()
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    audit_path = output_root / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") == "complete"
            and audit.get("protocol_sha256") == protocol_sha256
            and audit.get("source_root") == str(source_root)
        ):
            print(f"SKIP: split OSR sudah lengkap: {output_root}", flush=True)
            return audit
        raise FileExistsError(
            f"Audit {audit_path} tidak cocok dengan source/protokol saat ini. "
            "Gunakan output-root baru; jangan menimpa eksperimen lama."
        )
    if output_root.exists():
        raise FileExistsError(
            f"{output_root} sudah ada tanpa audit lengkap. Gunakan output-root baru."
        )

    classes, collected = _inspect_source(source_root)
    expected_classes = int(protocol["dataset"]["expected_classes"])
    if len(classes) != expected_classes:
        raise ValueError(
            f"Protokol mengharapkan {expected_classes} kelas, ditemukan {len(classes)}."
        )
    balanced_n = int(protocol["dataset"]["balanced_test_samples_per_class"])
    if balanced_n <= 0:
        raise ValueError("balanced_test_samples_per_class harus positif.")
    balance_seed = int(protocol["dataset"]["balance_seed"])

    split_audits: dict[str, dict] = {}
    for tier, split_cfg in protocol["splits"].items():
        unknown_classes = list(split_cfg["unknown_classes"])
        missing = sorted(set(unknown_classes) - set(classes))
        if missing:
            raise ValueError(f"Unknown split {tier} tidak ada di Coffee17: {missing}")
        known_classes = sorted(set(classes) - set(unknown_classes))
        tier_root = output_root / tier
        copied: dict[str, dict[str, int]] = {
            "known": {"train": 0, "val": 0, "test": 0},
            "unknown": {"test": 0},
        }
        for split in ("train", "val", "test"):
            for class_name in known_classes:
                for source in collected[split][class_name]:
                    _link_or_copy(
                        source,
                        tier_root / "source" / split / class_name / source.name,
                    )
                    copied["known"][split] += 1
        for class_name in unknown_classes:
            for source in collected["test"][class_name]:
                _link_or_copy(
                    source,
                    tier_root / "unknown" / "test" / class_name / source.name,
                )
                copied["unknown"]["test"] += 1

        balanced_manifest = {"known": [], "unknown": []}
        for population, names, root_name in (
            ("known", known_classes, "source"),
            ("unknown", unknown_classes, "unknown"),
        ):
            for class_name in names:
                paths = collected["test"][class_name]
                if len(paths) < balanced_n:
                    raise ValueError(
                        f"{tier}/{class_name} hanya memiliki {len(paths)} test; "
                        f"dibutuhkan {balanced_n}."
                    )
                selected = list(paths)
                random.Random(
                    f"{balance_seed}:{tier}:{population}:{class_name}"
                ).shuffle(selected)
                balanced_manifest[population].extend(
                    {
                        "class": class_name,
                        "path": str(
                            Path(root_name) / "test" / class_name / path.name
                        ).replace("\\", "/"),
                    }
                    for path in sorted(selected[:balanced_n], key=lambda p: p.name)
                )

        manifest_path = tier_root / "balanced_test_manifest.json"
        manifest_path.write_text(
            json.dumps(balanced_manifest, indent=2), encoding="utf-8"
        )
        split_audits[tier] = {
            "unknown_classes": unknown_classes,
            "known_classes": known_classes,
            "rationale": split_cfg["rationale"],
            "counts": copied,
            "balanced_counts": {
                "known": len(balanced_manifest["known"]),
                "unknown": len(balanced_manifest["unknown"]),
                "per_class": balanced_n,
            },
            "unknown_seen_in_train_or_val": False,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "source_root": str(source_root),
        "classes": classes,
        "splits": split_audits,
        "leakage_policy": (
            "Unknown-class images exist only under unknown/test; known train, "
            "validation, test, model selection, and calibration contain known classes."
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\n=== COFFEE17 SEMANTIC OSR V1 ===", flush=True)
    for tier, values in split_audits.items():
        counts = values["balanced_counts"]
        print(
            f"{tier:6s}: unknown={values['unknown_classes']} | "
            f"balanced known={counts['known']} unknown={counts['unknown']}",
            flush=True,
        )
    print(f"SAVED: {audit_path}", flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare three leakage-safe Coffee17 semantic OSR splits"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/osr/coffee17_osr_v1.yaml"),
    )
    args = parser.parse_args()
    prepare_osr_splits(args.source_root, args.output_root, args.protocol)


if __name__ == "__main__":
    main()
