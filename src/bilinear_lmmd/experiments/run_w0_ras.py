from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.core.reproducibility import (
    canonical_json_sha256,
    capture_rng_state,
    current_git_commit,
    model_state_fingerprint,
    restore_rng_state,
    seed_everything,
    sha256_file,
)
from bilinear_lmmd.core.run_lock import exclusive_training_lock
from bilinear_lmmd.data.preprocessing.runtime import PreprocessingRuntime
from bilinear_lmmd.data.preprocessing.study_data import (
    PreprocessingStudyImageFolder,
    build_preprocessing_study_loaders,
    set_study_epoch,
)
from bilinear_lmmd.data.preprocessing.wav1 import (
    haar_dwt2,
    soft_threshold,
    visushrink_threshold,
)
from bilinear_lmmd.engine.preprocessing_study import (
    _evaluate_model,
    evaluate_preprocessing_checkpoint,
)
from bilinear_lmmd.engine.shared_multiview import validation_identity_label_sha256
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.modeling.models import build_model


PROTOCOL = "coffee17-w0-ras-v1"


def _json(path: Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_w0_ras_config(cfg: dict) -> None:
    if int(cfg.get("seed", -1)) != 42:
        raise ValueError("W0-RAS V1 dikunci seed 42.")
    if cfg["adaptation"]["method"] != "source_only":
        raise ValueError("W0-RAS V1 harus source_only.")
    if cfg["model"]["backbone"] != "mobilenetv3_large_100":
        raise ValueError("W0-RAS V1 dikunci MobileNetV3-Large.")
    if cfg["model"]["head"] != "gap" or cfg["model"]["out_indices"] != [4]:
        raise ValueError("W0-RAS V1 dikunci ke GAP out_indices=[4].")
    if cfg["model"]["classifier"] != "linear":
        raise ValueError("W0-RAS V1 dikunci classifier linear.")
    if cfg["preprocessing"]["code"] != "R0":
        raise ValueError("W0-RAS classifier input harus raw RGB (R0).")
    if int(cfg["training"]["epochs"]) != 50:
        raise ValueError("W0-RAS V1 dikunci 50 epoch.")
    if cfg["training"]["classification_loss"] != "cross_entropy":
        raise ValueError("W0-RAS V1 dikunci CrossEntropy.")

    aux = cfg.get("auxiliary", {})
    expected = {
        "method": "w0_l1_hh_retention",
        "lambda": 0.05,
        "wavelet_levels": 1,
        "detail_band": "HH",
        "channel_reduce": "mean_rgb",
        "threshold": "visushrink",
        "target_standardization": "train_unrotated",
        "target_std_epsilon": 1.0e-8,
        "epoch1_weighted_aux_to_ce_abort_ratio": 1.0,
    }
    if aux != expected:
        raise RuntimeError(f"W0-RAS auxiliary contract berubah: {aux}")


def w0_l1_hh_retention(
    images: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Mean-RGB L1-HH retained-energy ratio using the frozen W0 operators."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("W0-RAS target membutuhkan BCHW RGB.")
    if not torch.is_floating_point(images):
        raise TypeError("W0-RAS target membutuhkan tensor floating point.")

    bands, _ = haar_dwt2(images)
    details = bands[:, :, 1:]
    threshold = visushrink_threshold(details, eps=eps)
    post = soft_threshold(details, threshold)

    # Frozen Haar order is [LL, LH, HL, HH], so detail index 2 is HH.
    hh_pre = details[:, :, 2]
    hh_post = post[:, :, 2]
    pre_energy = hh_pre.square().mean(dim=(-2, -1))
    post_energy = hh_post.square().mean(dim=(-2, -1))
    per_channel = post_energy / pre_energy.clamp_min(eps)
    return per_channel.mean(dim=1)


def _compute_train_target_stats(
    cfg: dict,
    *,
    seed: int,
    device: torch.device,
) -> dict:
    data_cfg = cfg["data"]
    train_root = (
        Path(data_cfg["root"])
        / str(data_cfg.get("source", "source"))
        / str(data_cfg.get("train_split", "train"))
    )
    if not train_root.is_dir():
        raise FileNotFoundError(f"Train split tidak ditemukan: {train_root}")

    dataset = PreprocessingStudyImageFolder(
        train_root,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        seed=seed,
        rotation_angles=[0.0],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("workers", 4)),
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    eps = float(cfg["auxiliary"]["target_std_epsilon"])
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in loader:
            values.append(
                w0_l1_hh_retention(
                    images.to(device, non_blocking=True),
                    eps=eps,
                ).cpu()
            )

    target = torch.cat(values)
    mean = target.mean()
    std = target.std(unbiased=False)
    if not torch.isfinite(mean) or not torch.isfinite(std):
        raise RuntimeError("W0-RAS train target stats non-finite.")
    if float(std) <= eps:
        raise RuntimeError("W0-RAS train target std terlalu kecil.")

    return {
        "count": int(target.numel()),
        "mean": float(mean),
        "std": float(std),
        "minimum": float(target.min()),
        "maximum": float(target.max()),
        "source": "development_train_unrotated_resize224_only",
        "validation_accessed": False,
    }


def _target_stats_match(left: dict, right: dict, atol: float = 1.0e-7) -> bool:
    if (
        int(left.get("count", -1)) != int(right.get("count", -2))
        or left.get("source") != right.get("source")
        or left.get("validation_accessed") is not False
        or right.get("validation_accessed") is not False
    ):
        return False
    for key in ("mean", "std", "minimum", "maximum"):
        if abs(float(left[key]) - float(right[key])) > atol:
            return False
    return True


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return False
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) >= epochs


def train_w0_ras(
    cfg: dict,
    *,
    run_dir: Path,
    run_contract_sha256: str,
    expected_initial_model_sha256: str,
    resume: bool,
) -> dict:
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["device"]))
    loaders = build_preprocessing_study_loaders(cfg["data"], seed=seed)

    if len(loaders.classes) != int(cfg["model"]["num_classes"]):
        raise ValueError("Jumlah kelas data/model berbeda.")

    model = build_model(cfg["model"]).to(device)
    initial_sha = model_state_fingerprint(model)
    if initial_sha != expected_initial_model_sha256:
        raise RuntimeError(
            "Initial model berbeda dari matched R0 reference: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )

    # Adding the training-only head must not shift the base training RNG stream.
    rng_after_model = capture_rng_state()
    aux_head = nn.Linear(int(model.pool.output_dim), 1).to(device)
    aux_initial_sha = model_state_fingerprint(aux_head)
    restore_rng_state(rng_after_model)

    runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)

    # Fold-only target-stat computation must not perturb paired training RNG.
    rng_before_stats = capture_rng_state()
    try:
        target_stats = _compute_train_target_stats(
            cfg,
            seed=seed,
            device=device,
        )
    finally:
        restore_rng_state(rng_before_stats)

    target_mean = float(target_stats["mean"])
    target_std = float(target_stats["std"])
    target_eps = float(cfg["auxiliary"]["target_std_epsilon"])
    lambda_ras = float(cfg["auxiliary"]["lambda"])
    abort_ratio = float(
        cfg["auxiliary"]["epoch1_weighted_aux_to_ce_abort_ratio"]
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    ce_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0))
    )
    mse_fn = nn.MSELoss()
    hard_groups = cfg.get("evaluation", {}).get("hard_groups", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history: list[dict] = []
    best_f1 = -1.0
    start_epoch = 0

    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError("Checkpoint resume berasal dari run-contract berbeda.")
        if checkpoint.get("classes") != loaders.classes:
            raise RuntimeError("Urutan kelas checkpoint berbeda.")
        if not _target_stats_match(
            checkpoint.get("target_stats", {}),
            target_stats,
        ):
            raise RuntimeError("Target standardization stats berubah saat resume.")

        required = {
            "model",
            "aux_head",
            "optimizer",
            "scheduler",
            "history",
            "best_f1",
            "rng_state",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint resume tidak lengkap: {missing}")

        model.load_state_dict(checkpoint["model"])
        aux_head.load_state_dict(checkpoint["aux_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_f1 = float(checkpoint["best_f1"])
        start_epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"RESUME: epoch {start_epoch + 1}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        set_study_epoch(loaders.train, epoch)
        model.train()
        aux_head.train()

        ce_sum = 0.0
        ras_sum = 0.0
        total_sum = 0.0
        sample_count = 0

        progress = tqdm(loaders.train, desc=f"epoch {epoch + 1}/{epochs}")
        for images, labels in progress:
            batch = int(labels.shape[0])
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                raw_target = w0_l1_hh_retention(
                    images.to(device, non_blocking=True),
                    eps=target_eps,
                )
                standardized_target = (
                    raw_target - target_mean
                ) / max(target_std, target_eps)

            model_input = runtime(images)
            optimizer.zero_grad(set_to_none=True)
            output = model(model_input, labels=labels)
            predicted_target = aux_head(output.embedding).squeeze(1)

            ce_loss = ce_fn(output.logits, labels)
            ras_loss = mse_fn(predicted_target, standardized_target)
            loss = ce_loss + lambda_ras * ras_loss
            loss.backward()
            optimizer.step()

            ce_sum += float(ce_loss.item()) * batch
            ras_sum += float(ras_loss.item()) * batch
            total_sum += float(loss.item()) * batch
            sample_count += batch
            progress.set_postfix(
                ce=f"{ce_loss.item():.4f}",
                ras=f"{ras_loss.item():.4f}",
            )

        mean_ce = ce_sum / max(sample_count, 1)
        mean_ras = ras_sum / max(sample_count, 1)
        mean_total = total_sum / max(sample_count, 1)
        weighted_ratio = lambda_ras * mean_ras / max(mean_ce, 1.0e-12)

        if epoch == 0 and weighted_ratio > abort_ratio:
            failure = {
                "format": "bilinear_lmmd.w0_ras.loss_scale_failure.v1",
                "epoch": 1,
                "mean_ce": mean_ce,
                "mean_ras": mean_ras,
                "lambda_ras": lambda_ras,
                "weighted_aux_to_ce_ratio": weighted_ratio,
                "abort_ratio": abort_ratio,
                "decision": "ABORT_LOSS_CONTRACT_FAILURE",
            }
            (run_dir / "loss_scale_failure.json").write_text(
                json.dumps(failure, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "W0-RAS epoch-1 weighted auxiliary loss mendominasi CE: "
                f"{weighted_ratio:.6f} > {abort_ratio:.6f}"
            )

        scheduler.step()

        metrics, _, _, _, _ = _evaluate_model(
            model,
            loaders.val,
            runtime,
            device,
            loaders.classes,
            hard_groups,
        )

        record = {
            "epoch": epoch + 1,
            "loss": mean_total,
            "ce_loss": mean_ce,
            "ras_loss": mean_ras,
            "weighted_aux_to_ce_ratio": weighted_ratio,
            "source": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "ce_loss": mean_ce,
                    "ras_loss": mean_ras,
                    "weighted_aux_to_ce_ratio": weighted_ratio,
                    "macro_f1": metrics["macro_f1"],
                    "hard_class_f1": metrics["hard_class_f1"],
                    "worst_class_f1": metrics["worst_class_f1"],
                }
            ),
            flush=True,
        )

        is_best = float(metrics["macro_f1"]) > best_f1
        if is_best:
            best_f1 = float(metrics["macro_f1"])

        checkpoint = {
            "model": model.state_dict(),
            "aux_head": aux_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": loaders.classes,
            "config": cfg,
            "epoch": epoch + 1,
            "history": history,
            "best_f1": best_f1,
            "target_stats": target_stats,
            "rng_state": capture_rng_state(),
            "run_contract_sha256": run_contract_sha256,
            "initial_model_state_sha256": expected_initial_model_sha256,
            "aux_initial_model_state_sha256": aux_initial_sha,
        }
        atomic_torch_save(checkpoint, last_path)

        if is_best:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "aux_head": aux_head.state_dict(),
                    "classes": loaders.classes,
                    "config": cfg,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                    "weights": "raw",
                    "target_stats": target_stats,
                    "run_contract_sha256": run_contract_sha256,
                    "initial_model_state_sha256": expected_initial_model_sha256,
                    "aux_initial_model_state_sha256": aux_initial_sha,
                },
                best_path,
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

    if not _run_complete(run_dir, epochs):
        raise RuntimeError("W0-RAS training belum menyelesaikan seluruh epoch.")

    evaluate_preprocessing_checkpoint(
        best_path,
        data_root=Path(cfg["data"]["root"]),
        split=str(cfg["data"].get("val_split", "val")),
        output_dir=run_dir / "validation",
    )

    return {
        "target_stats": target_stats,
        "epoch1_weighted_aux_to_ce_ratio": float(
            history[0]["weighted_aux_to_ce_ratio"]
        ),
        "training_only_auxiliary_params": int(
            sum(parameter.numel() for parameter in aux_head.parameters())
        ),
        "deployment_auxiliary_params": 0,
    }


def run_w0_ras(
    *,
    config_path: Path,
    reference_path: Path,
    fold: int,
    data_root: Path,
    output_root: Path,
    required_commit: str,
    device: str = "auto",
    resume: bool = False,
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training.")
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold harus 1..5.")

    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = current_git_commit(repo_root)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari frozen commit: "
            f"{actual_commit} != {required_commit}"
        )

    cfg = copy.deepcopy(load_config(config_path))
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    validate_w0_ras_config(cfg)

    reference = _json(reference_path, "Matched R0 reference")
    if reference.get("format") != "coffee17.w0_ras.r0_reference.v1":
        raise RuntimeError("Matched R0 reference format tidak dikenal.")
    if int(reference.get("seed", -1)) != 42:
        raise RuntimeError("Matched R0 reference seed bukan 42.")

    expected = reference["folds"][f"fold_{fold}"]
    count, val_sha = validation_identity_label_sha256(data_root)
    if count != int(expected["count"]):
        raise RuntimeError(
            f"Validation count berbeda: {count} != {expected['count']}"
        )
    if val_sha != expected["identity_label_sha256"]:
        raise RuntimeError("Validation identity hash berbeda dari matched R0.")

    seed_everything(42)
    probe = build_model(copy.deepcopy(cfg["model"]))
    current_initial = model_state_fingerprint(probe)
    del probe
    expected_initial = reference["expected_initial_model_state_sha256"]
    if current_initial != expected_initial:
        raise RuntimeError(
            "Initial primary model fingerprint berubah dari matched R0: "
            f"{current_initial} != {expected_initial}"
        )

    run_dir = (
        Path(output_root).expanduser().resolve()
        / "W0_RAS"
        / f"fold_{fold}"
        / "seed42"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["output_dir"] = str(run_dir)

    contract = {
        "format": "bilinear_lmmd.w0_ras.run_contract.v1",
        "protocol": PROTOCOL,
        "scientific_scope": "post-OOF exploratory selective W0 retention transfer",
        "fold": int(fold),
        "seed": 42,
        "git_commit": actual_commit,
        "reference_sha256": sha256_file(reference_path),
        "validation_identity_label_sha256": val_sha,
        "validation_count": count,
        "primary_initial_model_state_sha256": current_initial,
        "deployment_view": "R0",
        "target": "mean_rgb_l1_hh_retained_energy_ratio_after_visushrink",
        "target_stats_population": "development_train_unrotated_only",
        "objective": {
            "primary_ce_weight": 1.0,
            "auxiliary_loss": "mse_on_standardized_scalar_retention_target",
            "auxiliary_weight": float(cfg["auxiliary"]["lambda"]),
        },
        "training_only_auxiliary_head": "linear_gap_to_scalar",
        "extra_inference_parameters": 0,
        "extra_inference_forward_passes": 0,
        "epochs": int(cfg["training"]["epochs"]),
        "evaluation_split_during_training": "validation_R0_primary_only",
        "test_images_accessed": False,
        "resolved_config_sha256": canonical_json_sha256(cfg),
    }
    contract_sha = canonical_json_sha256(contract)
    contract["run_contract_sha256"] = contract_sha

    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        if _json(contract_path, "Existing contract") != contract:
            raise RuntimeError("Existing W0-RAS run contract berbeda.")
    else:
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    result_path = run_dir / "result.json"
    if result_path.is_file():
        old = _json(result_path, "Existing result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing result berasal dari kontrak berbeda.")
        return old

    with exclusive_training_lock(
        output_root,
        lock_name=f"W0_RAS_fold{fold}_seed42.training.lock",
        stale_seconds=900,
    ):
        training_executed = False
        if not _run_complete(run_dir, int(cfg["training"]["epochs"])):
            train_summary = train_w0_ras(
                cfg,
                run_dir=run_dir,
                run_contract_sha256=contract_sha,
                expected_initial_model_sha256=current_initial,
                resume=resume or (run_dir / "last.pt").is_file(),
            )
            training_executed = True
        else:
            last = torch.load(
                run_dir / "last.pt",
                map_location="cpu",
                weights_only=False,
            )
            if not (run_dir / "validation/metrics.json").is_file():
                evaluate_preprocessing_checkpoint(
                    run_dir / "best.pt",
                    data_root=Path(cfg["data"]["root"]),
                    split=str(cfg["data"].get("val_split", "val")),
                    output_dir=run_dir / "validation",
                )
            train_summary = {
                "target_stats": last["target_stats"],
                "epoch1_weighted_aux_to_ce_ratio": float(
                    last["history"][0]["weighted_aux_to_ce_ratio"]
                ),
                "training_only_auxiliary_params": int(
                    last["aux_head"]["weight"].numel()
                    + last["aux_head"]["bias"].numel()
                ),
                "deployment_auxiliary_params": 0,
            }

    metrics = _json(run_dir / "validation/metrics.json", "Validation metrics")
    baseline = expected["r0_control"]
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "worst_class_f1",
        "hard_class_f1",
    )

    result = {
        "format": "bilinear_lmmd.w0_ras.result.v1",
        "protocol": PROTOCOL,
        "method": "W0_RAS",
        "fold": int(fold),
        "seed": 42,
        "metrics": {key: metrics[key] for key in metric_keys},
        "frozen_matched_r0_control": baseline,
        "delta_vs_r0_control": {
            key: metrics[key] - baseline[key]
            for key in metric_keys
        },
        "target_stats": train_summary["target_stats"],
        "epoch1_weighted_aux_to_ce_ratio": train_summary[
            "epoch1_weighted_aux_to_ce_ratio"
        ],
        "training_only_auxiliary_params": train_summary[
            "training_only_auxiliary_params"
        ],
        "deployment_auxiliary_params": 0,
        "best_checkpoint": str(run_dir / "best.pt"),
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()

    run_w0_ras(
        config_path=args.config,
        reference_path=args.reference,
        fold=args.fold,
        data_root=args.data_root,
        output_root=args.output_root,
        required_commit=args.required_commit,
        device=args.device,
        resume=args.resume,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
