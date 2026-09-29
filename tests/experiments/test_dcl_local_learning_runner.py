from __future__ import annotations

import copy

from bilinear_lmmd.experiments.run_dcl_local_learning import (
    _control_contract_is_compatible,
)


def _contract(commit: str) -> dict:
    return {
        "format": "bilinear_lmmd.dcl_local_learning.arm_contract.v1",
        "protocol": "coffee17-dcl-local-learning-v1",
        "fold": 1,
        "seed": 42,
        "arm": "HBP_CE",
        "git_commit": commit,
        "validation_identity_label_sha256": "abc",
        "validation_count": 97,
        "initial_core_state_sha256": "core",
        "model": {"backbone": "mobilenetv3_large_100", "head": "hbp"},
        "dcl": None,
        "training": {"epochs": 50, "lr": 0.0003},
        "resolved_config_sha256": "cfg",
        "outer_test_accessed": False,
        "run_contract_sha256": "ignored-by-compatibility-check",
    }


def test_legacy_control_contract_can_be_reused_when_science_matches() -> None:
    existing = _contract("993b67aaac1a8bbefa2ddb32ab4ce23056dd6172")
    proposed = _contract("new-commit")
    assert _control_contract_is_compatible(existing, proposed)


def test_legacy_control_contract_rejects_scientific_change() -> None:
    existing = _contract("993b67aaac1a8bbefa2ddb32ab4ce23056dd6172")
    proposed = _contract("new-commit")
    proposed = copy.deepcopy(proposed)
    proposed["training"]["lr"] = 0.001
    assert not _control_contract_is_compatible(existing, proposed)


def test_unapproved_legacy_commit_is_not_reused() -> None:
    existing = _contract("unknown")
    proposed = _contract("new-commit")
    assert not _control_contract_is_compatible(existing, proposed)
