from __future__ import annotations

import copy
import json
from pathlib import Path

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import canonical_json_sha256, sha256_file
from bilinear_lmmd.experiments.preprocessing_environment import verify_environment

ARMS = ("R0", "C0", "F0", "W0")
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {arm: REPO_ROOT / f"configs/preprocessing_study/{arm}.yaml" for arm in ARMS}

def _json(path: str | Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _common_config(cfg: dict) -> dict:
    value = copy.deepcopy(cfg); value.pop("preprocessing", None); value["training"].pop("output_dir", None); return value

def validate_primary_configs() -> dict:
    configs = {arm: load_config(path) for arm, path in CONFIGS.items()}
    common = {arm: canonical_json_sha256(_common_config(cfg)) for arm, cfg in configs.items()}
    if len(set(common.values())) != 1: raise RuntimeError(f"Primary arm common configs berbeda: {common}")
    for arm, cfg in configs.items():
        if cfg["preprocessing"]["code"] != arm: raise RuntimeError(f"{arm}: preprocessing.code tidak cocok.")
        if cfg["data"]["augmentation_mode"] != "preprocessing_study": raise RuntimeError(f"{arm}: augmentation_mode bukan preprocessing_study.")
        if cfg["adaptation"]["method"] != "source_only": raise RuntimeError(f"{arm}: adaptation bukan source_only.")
        if cfg["model"]["backbone"] != "mobilenetv3_large_100": raise RuntimeError(f"{arm}: backbone berubah.")
        if cfg["model"]["head"] != "gap" or cfg["model"]["out_indices"] != [4]: raise RuntimeError(f"{arm}: M0 GAP contract berubah.")
        if int(cfg["training"]["epochs"]) != 50: raise RuntimeError(f"{arm}: primary epochs harus 50.")
    return {"configs": configs,"common_config_sha256": next(iter(common.values())),"arm_config_sha256": {arm: sha256_file(path) for arm, path in CONFIGS.items()}}

def validate_development(development_root: Path, development_contract: Path) -> dict:
    root = Path(development_root).expanduser().resolve(); contract = _json(development_contract, "Development contract")
    if contract.get("format") != "bilinear_lmmd.coffee17.preprocessing_development.v1": raise RuntimeError("Development contract format tidak dikenal.")
    gates = {"decision_pass": contract.get("decision") == "PASS","train_present": (root / "source/train").is_dir(),"val_present": (root / "source/val").is_dir(),"test_absent": not (root / "source/test").exists(),"test_not_extracted": contract.get("test_images_extracted") is False,"training_not_executed_by_materializer": contract.get("training_executed") is False}
    if not all(gates.values()): raise RuntimeError(f"Development gate gagal: {[k for k,v in gates.items() if not v]}")
    return {"gates": gates,"development_contract_sha256": sha256_file(development_contract),"clean_content_sha256": contract["clean_content_sha256"],"fold_manifest_sha256": contract["fold_manifest_file_sha256"],"fold": int(contract["fold"])}

def validate_static_and_equivalence(static_preflight: Path, arm: str, equivalence: Path | None) -> dict:
    static = _json(static_preflight, "Static preflight")
    if static.get("decision") != "PASS_PREPROCESSING_STATIC_CONTRACT": raise RuntimeError("Static preprocessing preflight belum PASS.")
    if static.get("test_images_accessed") is not False: raise RuntimeError("Static preflight menyatakan test pernah diakses.")
    expected_initial = static["model_state_sha256"][arm]
    if arm == "R0": eq_hash = None
    else:
        if equivalence is None: raise RuntimeError(f"{arm} memerlukan reference-equivalence evidence.")
        eq = _json(equivalence, f"{arm} reference equivalence")
        if eq.get("arm") != arm or eq.get("decision") != "PASS" or eq.get("exact_equal") is not True or eq.get("training_executed") is not False or eq.get("test_images_accessed") is not False: raise RuntimeError(f"{arm} reference equivalence tidak valid.")
        eq_hash = sha256_file(equivalence)
    return {"static_preflight_sha256": sha256_file(static_preflight),"reference_equivalence_sha256": eq_hash,"expected_initial_model_sha256": expected_initial}

def validate_environment(reference: Path) -> dict:
    current = verify_environment(reference)
    return {"environment_reference_sha256": sha256_file(reference),"software_sha256": current["software_sha256"]}
