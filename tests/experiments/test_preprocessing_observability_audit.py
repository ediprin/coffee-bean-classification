from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from bilinear_lmmd.experiments.run_preprocessing_observability_audit import (
    run_observability_audit,
)


def test_observability_audit_is_training_free(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    rows = []
    for index in range(4):
        cls = "A" if index < 2 else "B"
        path = root / cls / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (32, 32),
            (40 + index * 20, 80 + index * 10, 120 + index * 5),
        ).save(path)
        rows.append(
            {
                "identity": f"{cls}/{index}.png",
                "class": cls,
                "filename": f"{index}.png",
            }
        )
    manifest = tmp_path / "raw.json"
    manifest.write_text(json.dumps({"images": rows}), encoding="utf-8")

    from bilinear_lmmd.experiments import (
        run_preprocessing_observability_audit as module,
    )

    configs = {}
    for arm in ("R0", "C0", "F0", "W0"):
        from bilinear_lmmd.core.config import load_config
        from bilinear_lmmd.experiments.preprocessing_contract import CONFIGS

        configs[arm] = load_config(CONFIGS[arm])

    monkeypatch.setattr(
        module,
        "validate_primary_configs",
        lambda: {
            "configs": configs,
            "arm_config_sha256": {arm: arm for arm in configs},
        },
    )
    result = run_observability_audit(
        root,
        manifest,
        tmp_path / "audit",
        device_name="cpu",
        image_size=32,
        expected_count=4,
    )
    assert result["decision"] == "PASS_PREPROCESSING_OBSERVABILITY_AUDIT"
    assert result["training_executed"] is False
    assert result["model_accessed"] is False
    assert result["image_count"] == 4
    assert result["arms"]["R0"]["mean_abs_delta"] == 0.0
