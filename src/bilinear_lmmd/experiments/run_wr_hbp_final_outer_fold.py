from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import current_git_commit, sha256_file
from bilinear_lmmd.data.preprocessing.runtime import imagenet_normalize
from bilinear_lmmd.data.preprocessing.study_data import (
    build_preprocessing_evaluation_loader,
    identity_from_path,
)
from bilinear_lmmd.engine.train import classification_metrics, resolve_device
from bilinear_lmmd.engine.wavelet_residual_hbp import build_candidate, build_control


PROTOCOL = "wr-hbp-final-confirmation-v1"
ARMS = ("R0_HBP", "WR_HBP")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "hard_class_f1",
    "worst_class_f1",
)


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("Final confirmation dikunci seed=42.")
    data = cfg["data"]
    if data.get("source") != "source" or data.get("test_split") != "test":
        raise ValueError("Final confirmation harus source/test.")
    if int(data.get("image_size", -1)) != 224:
        raise ValueError("image_size harus 224.")
    if bool(data.get("object_crop", False)):
        raise ValueError("object_crop harus false.")

    model = cfg["model"]
    expected = {
        "backbone": "mobilenetv3_large_100",
        "pretrained": False,
        "head": "hbp",
        "classifier": "linear",
        "num_classes": 17,
        "projection_dim": 512,
        "dropout": 0.2,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"model.{key} harus {value!r}.")
    if list(model.get("out_indices", [])) != [1, 3, 4]:
        raise ValueError("out_indices harus [1,3,4].")

    wavelet = cfg["wavelet_residual"]
    expected_wavelet = {
        "domain": "luminance",
        "level": "L1",
        "bands": ["LH", "HL", "HH"],
        "threshold": "visushrink_soft",
        "injection": "shallow_residual",
        "hidden_channels": 16,
        "gate": "tanh_zero_init",
        "eps": 1.0e-8,
    }
    for key, value in expected_wavelet.items():
        if wavelet.get(key) != value:
            raise ValueError(f"wavelet_residual.{key} berubah.")

    expected_hard_groups = {
        "sour_black": ["Partial Black", "Partial Sour", "Full Sour"],
        "shape_withered": ["Withered", "Immature", "Cut"],
        "insect_damage": ["Slight Insect Damage", "Severe Insect Damage"],
    }
    expected_pairs = [
        ["Withered", "Immature"],
        ["Severe Insect Damage", "Slight Insect Damage"],
        ["Cut", "Slight Insect Damage"],
        ["Partial Sour", "Full Sour"],
        ["Slight Insect Damage", "Fade"],
        ["Full Black", "Partial Black"],
    ]
    evaluation = cfg["evaluation"]
    if evaluation.get("hard_groups") != expected_hard_groups:
        raise ValueError("Frozen development-matched hard_groups berubah.")
    if evaluation.get("targeted_confusion_pairs") != expected_pairs:
        raise ValueError("Frozen targeted_confusion_pairs berubah.")

    gate = cfg["confirmation_gate"]
    if gate != {
        "pooled_macro_delta_gt": 0.0,
        "macro_positive_folds_at_least": 3,
        "pooled_hard_delta_ge": 0.0,
        "pooled_worst_delta_ge": 0.0,
    }:
        raise ValueError("Frozen confirmation gate berubah.")

    uncertainty = cfg["uncertainty"]
    if uncertainty != {
        "paired_stratified_bootstrap_replicates": 10000,
        "seed": 42,
    }:
        raise ValueError("Frozen bootstrap contract berubah.")


def _checkpoint_compatible(checkpoint: dict, cfg: dict, arm: str) -> None:
    if checkpoint.get("arm") != arm:
        raise RuntimeError(f"Checkpoint arm salah: {checkpoint.get('arm')} != {arm}")
    observed = checkpoint.get("config", {})
    observed_model = observed.get("model", {})
    expected_model = cfg["model"]
    for key in (
        "backbone", "head", "classifier", "num_classes",
        "out_indices", "projection_dim", "dropout",
    ):
        if observed_model.get(key) != expected_model.get(key):
            raise RuntimeError(
                f"Checkpoint {arm} model.{key} berbeda: "
                f"{observed_model.get(key)!r} != {expected_model.get(key)!r}"
            )
    if arm == "WR_HBP":
        observed_wavelet = observed.get("wavelet_residual", {})
        for key, value in cfg["wavelet_residual"].items():
            if observed_wavelet.get(key) != value:
                raise RuntimeError(f"Checkpoint WR_HBP wavelet_residual.{key} berbeda.")


def _load_model(
    checkpoint_path: Path,
    cfg: dict,
    *,
    arm: str,
    classes: list[str],
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _checkpoint_compatible(checkpoint, cfg, arm)
    if checkpoint.get("classes") != classes:
        raise RuntimeError(f"Urutan kelas checkpoint {arm} berbeda dari test runtime.")
    model = build_control(cfg) if arm == "R0_HBP" else build_candidate(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict(model, loader, device: torch.device, arm: str):
    y_true: list[int] = []
    y_pred: list[int] = []
    probs_all: list[list[float]] = []
    for raw, labels in loader:
        raw = raw.to(device, non_blocking=True)
        normalized = imagenet_normalize(raw)
        if arm == "R0_HBP":
            output = model(normalized)
        else:
            output = model(normalized, raw_rgb=raw)
        probs = output.logits.softmax(1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(probs.argmax(1).tolist())
        probs_all.extend(probs.tolist())
    identities = [identity_from_path(path) for path, _ in loader.dataset.samples]
    if len(identities) != len(y_true):
        raise RuntimeError("Jumlah identity test berbeda dari prediction rows.")
    return y_true, y_pred, probs_all, identities


def _write_predictions(
    path: Path,
    *,
    identities: list[str],
    y_true: list[int],
    y_pred: list[int],
    probs: list[list[float]],
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["identity", "actual", "predicted", "correct"] + [
        f"p::{name}" for name in classes
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for identity, actual, predicted, probability in zip(
            identities, y_true, y_pred, probs
        ):
            row = {
                "identity": identity,
                "actual": classes[actual],
                "predicted": classes[predicted],
                "correct": "1" if actual == predicted else "0",
            }
            row.update(
                {f"p::{name}": float(probability[i]) for i, name in enumerate(classes)}
            )
            writer.writerow(row)


def _targeted_confusions(
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
    pairs: list[list[str]],
) -> dict[str, int]:
    actual = [classes[i] for i in y_true]
    predicted = [classes[i] for i in y_pred]
    out = {}
    for left, right in pairs:
        key = f"{left} <-> {right}"
        out[key] = sum(
            (a == left and p == right) or (a == right and p == left)
            for a, p in zip(actual, predicted)
        )
    out["TOTAL"] = int(sum(out.values()))
    return out


def _paired_outcomes(y_true: list[int], left: list[int], right: list[int]) -> dict:
    rescue = damage = both_correct = both_wrong = 0
    for y, a, b in zip(y_true, left, right):
        ac, bc = a == y, b == y
        if not ac and bc:
            rescue += 1
        elif ac and not bc:
            damage += 1
        elif ac and bc:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "count": len(y_true),
        "rescue": rescue,
        "damage": damage,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_correct": rescue - damage,
    }


def run_fold(
    *,
    config_path: Path,
    fold: int,
    test_root: Path,
    development_root: Path,
    authority_path: Path,
    output_root: Path,
    required_commit: str,
    device: str,
) -> dict:
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Scientific code pin salah: {actual_commit} != {required_commit}"
        )

    cfg = load_config(config_path)
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(test_root).resolve())
    validate_config(cfg)

    authority_path = Path(authority_path).resolve()
    authority = _json(authority_path)
    if authority.get("decision") != "AUTHORIZE_OOF_TEST_EVALUATION":
        raise RuntimeError("Final-test authority tidak valid.")
    if authority.get("scope") != PROTOCOL:
        raise RuntimeError("Authority scope salah.")
    if authority.get("further_primary_tuning_authorized") is not False:
        raise RuntimeError("Authority masih mengizinkan tuning.")
    expected_sha = authority["development_checkpoint_sha256"][str(fold)]

    test_contract = _json(Path(test_root) / "test_contract.json")
    if test_contract.get("training_executed") is not False:
        raise RuntimeError("Test runtime menunjukkan training.")
    if test_contract.get("test_images_accessed") is not True:
        raise RuntimeError("Test runtime belum materialized.")
    if test_contract.get("authority_sha256") != sha256_file(authority_path):
        raise RuntimeError("Test runtime berasal dari authority berbeda.")

    loader, classes = build_preprocessing_evaluation_loader(
        cfg["data"], split="test", seed=42
    )
    if len(classes) != 17:
        raise RuntimeError("Final test harus 17 kelas.")

    development_root = Path(development_root).resolve()
    pair_root = development_root / f"fold_{fold}" / "seed42"
    checkpoint_paths = {
        arm: pair_root / arm / "best.pt"
        for arm in ARMS
    }
    for arm in ARMS:
        digest = sha256_file(checkpoint_paths[arm])
        if digest != expected_sha[arm]:
            raise RuntimeError(f"Fold {fold} {arm}: checkpoint berubah setelah authority.")

    resolved = resolve_device(device)
    outputs = {}
    for arm in ARMS:
        model = _load_model(
            checkpoint_paths[arm], cfg, arm=arm, classes=classes, device=resolved
        )
        y_true, y_pred, probs, identities = _predict(model, loader, resolved, arm)
        metrics = classification_metrics(
            y_true, y_pred, classes, cfg["evaluation"]["hard_groups"]
        )
        outputs[arm] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "probs": probs,
            "identities": identities,
            "metrics": metrics,
            "targeted": _targeted_confusions(
                y_true, y_pred, classes,
                cfg["evaluation"]["targeted_confusion_pairs"],
            ),
        }

    if outputs["R0_HBP"]["identities"] != outputs["WR_HBP"]["identities"]:
        raise RuntimeError("R0/WR test identity order berbeda.")
    if outputs["R0_HBP"]["y_true"] != outputs["WR_HBP"]["y_true"]:
        raise RuntimeError("R0/WR test labels berbeda.")

    run_dir = Path(output_root).resolve() / f"fold_{fold}"
    for arm in ARMS:
        arm_dir = run_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "metrics.json").write_text(
            json.dumps(outputs[arm]["metrics"], indent=2) + "\n",
            encoding="utf-8",
        )
        _write_predictions(
            arm_dir / "predictions.csv",
            identities=outputs[arm]["identities"],
            y_true=outputs[arm]["y_true"],
            y_pred=outputs[arm]["y_pred"],
            probs=outputs[arm]["probs"],
            classes=classes,
        )

    delta = {
        metric: float(outputs["WR_HBP"]["metrics"][metric])
        - float(outputs["R0_HBP"]["metrics"][metric])
        for metric in METRICS
    }
    result = {
        "format": "bilinear_lmmd.wr_hbp_final_outer_fold.v1",
        "protocol": PROTOCOL,
        "fold": fold,
        "git_commit": actual_commit,
        "classes": classes,
        "test_count": len(outputs["R0_HBP"]["y_true"]),
        "R0_HBP": {
            metric: float(outputs["R0_HBP"]["metrics"][metric])
            for metric in METRICS
        },
        "WR_HBP": {
            metric: float(outputs["WR_HBP"]["metrics"][metric])
            for metric in METRICS
        },
        "DELTA_WR_MINUS_R0": delta,
        "targeted_confusions": {
            arm: outputs[arm]["targeted"] for arm in ARMS
        },
        "paired_outcomes": _paired_outcomes(
            outputs["R0_HBP"]["y_true"],
            outputs["R0_HBP"]["y_pred"],
            outputs["WR_HBP"]["y_pred"],
        ),
        "development_checkpoint_sha256": expected_sha,
        "authority_sha256": sha256_file(authority_path),
        "training_executed": False,
        "outer_test_accessed": True,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--development-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_fold(
        config_path=args.config,
        fold=args.fold,
        test_root=args.test_root,
        development_root=args.development_root,
        authority_path=args.authority,
        output_root=args.output_root,
        required_commit=args.required_commit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
