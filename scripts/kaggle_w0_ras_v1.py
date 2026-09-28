from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import kagglehub


HANDLE_DEFAULT = "situjukamkape/notebook40a6318933/versions/1"
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
DATA = WORK / "coffee17_canonical_recovered"
OUT = WORK / "coffee17-w0-ras-v1"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_project(root: Path) -> Path:
    if root.name == "coffee17-preprocessing-project":
        return root
    hits = sorted(
        path
        for path in root.rglob("coffee17-preprocessing-project")
        if path.is_dir()
    )
    if len(hits) != 1:
        raise RuntimeError(
            "Expected exactly one coffee17-preprocessing-project in "
            f"{root}, found {hits}"
        )
    return hits[0]


def exact_data_ready(data: Path, entries: list[dict]) -> bool:
    if not data.is_dir():
        return False
    files = [path for path in data.rglob("*") if path.is_file()]
    if len(files) != len(entries):
        return False
    for row in entries:
        path = data / row["identity"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            return False
    return True


def recover_clean_data(entries: list[dict]) -> Path:
    if exact_data_ready(DATA, entries):
        print(f"REUSE exact clean Coffee17: {DATA}", flush=True)
        return DATA

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    images = [
        path
        for path in INPUT.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    print(f"Mounted image files: {len(images)}", flush=True)
    if not images:
        raise RuntimeError("Coffee17 image dataset tidak mounted di /kaggle/input.")

    wanted_sizes = {int(row["bytes"]) for row in entries}
    candidates = [
        path for path in images if path.stat().st_size in wanted_sizes
    ]
    wanted_hashes = {row["sha256"] for row in entries}
    hash_to_path: dict[str, Path] = {}

    for index, path in enumerate(candidates, start=1):
        digest = sha256_file(path)
        if digest in wanted_hashes and digest not in hash_to_path:
            hash_to_path[digest] = path
        if index % 100 == 0 or index == len(candidates):
            print(
                f"Hash scan {index}/{len(candidates)} | "
                f"matched hashes {len(hash_to_path)}/{len(wanted_hashes)}",
                flush=True,
            )

    missing = [
        row["identity"]
        for row in entries
        if row["sha256"] not in hash_to_path
    ]
    if missing:
        raise RuntimeError(
            f"Exact Coffee17 match hanya {len(entries) - len(missing)}/"
            f"{len(entries)}. Missing examples: {missing[:10]}"
        )

    shutil.rmtree(DATA, ignore_errors=True)
    for row in entries:
        source = hash_to_path[row["sha256"]]
        destination = DATA / row["identity"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    if not exact_data_ready(DATA, entries):
        raise RuntimeError("Canonical recovery selesai tetapi final hash gate gagal.")
    print(f"Exact Coffee17 recovered: {len(entries)}/{len(entries)}", flush=True)
    return DATA


def merge_prior_outputs() -> None:
    candidates = sorted(
        path
        for path in INPUT.rglob("coffee17-w0-ras-v1")
        if path.is_dir()
    )
    if not candidates:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for prior in candidates:
        print(f"MERGE prior W0-RAS output: {prior}", flush=True)
        shutil.copytree(prior, OUT, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-handle", default=HANDLE_DEFAULT)
    args = parser.parse_args()

    if not INPUT.is_dir() or not WORK.is_dir():
        raise RuntimeError("Script ini dikunci untuk Kaggle Notebook.")

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    print("W0-RAS CODE COMMIT:", commit, flush=True)

    evidence_mount = Path(
        kagglehub.notebook_output_download(args.evidence_handle)
    )
    project = find_project(evidence_mount)

    data_evidence = project / "evidence/coffee17-preprocessing-data-v1"
    runtime_evidence = project / "evidence/coffee17-preprocessing-runtime-v1"
    static_evidence = project / "evidence/coffee17-preprocessing-static-v2"
    observability_evidence = (
        project / "evidence/coffee17-preprocessing-observability-v1"
    )
    control_root = project / "experiments/coffee17-preprocessing-primary-v1"

    clean_manifest = data_evidence / "clean_manifest.json"
    fold_manifest = data_evidence / "fold_manifest.json"
    environment = runtime_evidence / "runtime_environment.json"
    lock_file = runtime_evidence / "requirements_preprocessing_study_lock.txt"
    static_preflight = static_evidence / "static_preflight.json"
    observability = observability_evidence / "preprocessing_observability.json"

    required = (
        clean_manifest,
        fold_manifest,
        environment,
        lock_file,
        static_preflight,
        observability,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not control_root.is_dir():
        raise FileNotFoundError(control_root)

    clean = json.loads(clean_manifest.read_text(encoding="utf-8"))
    entries = clean["images"]
    if len(entries) != 965:
        raise RuntimeError(f"Expected 965 clean identities, got {len(entries)}")

    recover_clean_data(entries)
    merge_prior_outputs()
    OUT.mkdir(parents=True, exist_ok=True)

    # Reproduce the frozen preprocessing-study package set before importing
    # experiment code. The W0-RAS runner validates this environment again.
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(lock_file)])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "-e",
            str(repo),
        ]
    )

    manifest = {
        "format": "bilinear_lmmd.w0_ras.kaggle_run.v1",
        "code_commit": commit,
        "evidence_handle": args.evidence_handle,
        "clean_content_sha256": clean["clean_content_sha256"],
        "outer_test_oof_access_authorized": False,
    }
    (OUT / "kaggle_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for fold in range(1, 6):
        dev = WORK / f"coffee17_w0_ras_dev_fold_{fold}"
        shutil.rmtree(dev, ignore_errors=True)

        run(
            [
                sys.executable,
                "-m",
                "bilinear_lmmd.data.preparation.materialize_preprocessing_development",
                "--canonical-root",
                str(DATA),
                "--clean-manifest",
                str(clean_manifest),
                "--fold-manifest",
                str(fold_manifest),
                "--destination",
                str(dev),
                "--fold",
                str(fold),
            ],
            cwd=repo,
        )

        run(
            [
                sys.executable,
                "-u",
                "-m",
                "bilinear_lmmd.experiments.run_w0_ras",
                "--fold",
                str(fold),
                "--seed",
                "42",
                "--data-root",
                str(dev),
                "--development-contract",
                str(dev / "development_contract.json"),
                "--static-preflight",
                str(static_preflight),
                "--observability-audit",
                str(observability),
                "--environment",
                str(environment),
                "--output-root",
                str(OUT),
                "--required-commit",
                commit,
                "--device",
                "cuda:0",
                "--authorize-training",
            ],
            cwd=repo,
        )
        shutil.rmtree(dev, ignore_errors=True)

    aggregate_path = OUT / "W0_RAS_V1_AGGREGATE.json"
    run(
        [
            sys.executable,
            "-m",
            "bilinear_lmmd.experiments.aggregate_w0_ras",
            "--treatment-root",
            str(OUT),
            "--control-root",
            str(control_root),
            "--output",
            str(aggregate_path),
        ],
        cwd=repo,
    )

    zip_path = shutil.make_archive(
        str(WORK / "coffee17-w0-ras-v1"),
        "zip",
        root_dir=OUT,
    )

    print("\n========================================", flush=True)
    print("W0-RAS V1 COMPLETE", flush=True)
    print("Aggregate:", aggregate_path, flush=True)
    print("ZIP:", zip_path, flush=True)
    print("Save Version if you want resumable checkpoints.", flush=True)
    print("========================================", flush=True)


if __name__ == "__main__":
    main()
