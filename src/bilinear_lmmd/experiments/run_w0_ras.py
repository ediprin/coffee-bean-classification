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
from bilinear_lmmd.engine.train import atomic_torch_save, resolve_device
from bilinear_lmmd.experiments.preprocessing_contract import (
    CONFIGS,
    validate_development,
    validate_environment,
    validate_observability,
    validate_primary_configs,
    validate_static_and_equivalence,
)
from bilinear_lmmd.modeling.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/w0_ras/W0_RAS.yaml"
R0_CONFIG_PATH = CONFIGS["R0"]


def _json(path: Path, label: str = "JSON") -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _base_config(cfg: dict) -> dict:
    value = copy.deepcopy(cfg)
    value.pop("auxiliary", None)
    value["training"].pop("output_dir", None)
    return value


def _validate_w0_ras_config() -> dict:
    cfg = load_config(CONFIG_PATH)
    r0 = load_config(R0_CONFIG_PATH)

    if canonical_json_sha256(_base_config(cfg)) != canonical_json_sha256(
        _base_config(r0)
    ):
        raise RuntimeError(
            "W0-RAS base config tidak identik dengan preprocessing-study R0."
        )

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

    if cfg["preprocessing"]["code"] != "R0":
        raise RuntimeError("W0-RAS deployment/training classifier input harus R0.")
    if cfg["model"]["head"] != "gap" or cfg["model"]["out_indices"] != [4]:
        raise RuntimeError("W0-RAS V1 dikunci ke MobileNetV3-Large GAP.")
    if int(cfg["training"]["epochs"]) != 50:
        raise RuntimeError("W0-RAS V1 dikunci 50 epoch.")

    return cfg


def w0_l1_hh_retention(images: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Mean-RGB retained HH energy ratio using the exact frozen W0 operators."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("W0-RAS target membutuhkan BCHW RGB.")
    if not torch.is_floating_point(images):
        raise TypeError("W0-RAS target membutuhkan tensor floating point.")

    bands, _ = haar_dwt2(images)
    details = bands[:, :, 1:]
    threshold = visushrink_threshold(details, eps=eps)
    post = soft_threshold(details, threshold)

    # Existing Haar band order is [LL, LH, HL, HH], hence detail index 2 = HH.
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
    root = Path(data_cfg["root"])
    source = root / str(data_cfg.get("source", "source"))
    train_root = source / str(data_cfg.get("train_split", "train"))
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

    values = []
    eps = float(cfg["auxiliary"]["target_std_epsilon"])
    with torch.no_grad():
        for images, _ in loader:
            target = w0_l1_hh_retention(
                images.to(device, non_blocking=True),
                eps=eps,
            )
            values.append(target.cpu())

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
            "Initial base model berbeda dari frozen R0 preflight: "
            f"{initial_sha} != {expected_initial_model_sha256}"
        )

    # Do not shift the base training RNG stream merely because an auxiliary
    # head is added. This follows the paired-init pattern used elsewhere here.
    rng_after_model = torch.random.get_rng_state()
    aux_head = nn.Linear(int(model.pool.output_dim), 1).to(device)
    aux_initial_sha = model_state_fingerprint(aux_head)
    torch.random.set_rng_state(rng_after_model)

    runtime = PreprocessingRuntime.from_config(cfg["preprocessing"], device)
    target_stats = _compute_train_target_stats(cfg, seed=seed, device=device)
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
        optimizer, T_max=epochs
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
        if checkpoint.get("target_stats") != target_stats:
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
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                "W0-RAS epoch-1 weighted auxiliary loss mendominasi CE: "
                f"ratio={weighted_ratio:.6f} > {abort_ratio:.6f}"
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

        is_best = metrics["macro_f1"] > best_f1
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
            best_checkpoint = {
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
            }
            atomic_torch_save(best_checkpoint, best_path)

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
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
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "completed_epochs": epochs,
        "best_validation_macro_f1": best_f1,
        "target_stats": target_stats,
        "epoch1_weighted_aux_to_ce_ratio": float(
            history[0]["weighted_aux_to_ce_ratio"]
        ),
        "initial_model_state_sha256": expected_initial_model_sha256,
        "aux_initial_model_state_sha256": aux_initial_sha,
        "training_only_auxiliary_params": int(
            sum(parameter.numel() for parameter in aux_head.parameters())
        ),
        "deployment_auxiliary_params": 0,
    }


def run_w0_ras(
    *,
    fold: int,
    seed: int,
    data_root: Path,
    development_contract: Path,
    static_preflight: Path,
    observability_audit: Path,
    environment: Path,
    output_root: Path,
    required_commit: str,
    authorize_training: bool = False,
    device: str = "auto",
) -> dict:
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training.")
    if seed != 42:
        raise ValueError("W0-RAS V1 dikunci seed 42.")
    if fold not in {1, 2, 3, 4, 5}:
        raise ValueError("Fold W0-RAS harus 1..5.")
    if not required_commit.strip():
        raise RuntimeError("required_commit wajib diisi.")

    actual_commit = current_git_commit(REPO_ROOT)
    if actual_commit != required_commit:
        raise RuntimeError(
            f"Git commit berbeda dari frozen commit: {actual_commit} != {required_commit}"
        )

    validate_primary_configs()
    cfg = _validate_w0_ras_config()

    development = validate_development(data_root, development_contract)
    if development["fold"] != fold:
        raise RuntimeError("Development contract fold mismatch.")

    observability = validate_observability(observability_audit)
    static = validate_static_and_equivalence(static_preflight, "R0", None)
    environment_gate = validate_environment(environment)

    cfg["seed"] = seed
    cfg["device"] = device
    cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())

    run_dir = (
        Path(output_root).expanduser().resolve()
        / "treatment"
        / f"fold_{fold}"
        / f"seed{seed}"
    )
    cfg["training"]["output_dir"] = str(run_dir)

    frozen_config_text = yaml.safe_dump(
        cfg, sort_keys=False, allow_unicode=True
    )
    frozen_config_sha = canonical_json_sha256(cfg)

    primary = validate_primary_configs()
    contract = {
        "format": "bilinear_lmmd.w0_ras.arm_contract.v1",
        "protocol": "coffee17-w0-ras-v1",
        "scientific_evidence": True,
        "fold": fold,
        "seed": seed,
        "git_commit": actual_commit,
        "clean_content_sha256": development["clean_content_sha256"],
        "fold_manifest_sha256": development["fold_manifest_sha256"],
        "development_contract_sha256": development[
            "development_contract_sha256"
        ],
        "observability_audit_sha256": observability[
            "observability_audit_sha256"
        ],
        "static_preflight_sha256": static["static_preflight_sha256"],
        "environment_reference_sha256": environment_gate[
            "environment_reference_sha256"
        ],
        "software_sha256": environment_gate["software_sha256"],
        "r0_common_config_sha256": primary["common_config_sha256"],
        "r0_arm_config_sha256": primary["arm_config_sha256"]["R0"],
        "w0_ras_config_sha256": sha256_file(CONFIG_PATH),
        "resolved_run_config_sha256": frozen_config_sha,
        "common_initialized_model_state_sha256": static[
            "expected_initial_model_sha256"
        ],
        "target": "mean_rgb_l1_hh_retained_energy_ratio_after_visushrink",
        "target_stats_population": "development_train_unrotated_only",
        "lambda_ras": float(cfg["auxiliary"]["lambda"]),
        "epochs": int(cfg["training"]["epochs"]),
        "checkpoint_selection": "validation_macro_f1",
        "evaluation_split_during_training": "val",
        "test_images_accessed": False,
        "outer_test_oof_access_authorized": False,
    }
    contract_sha = canonical_json_sha256(contract)
    contract["run_contract_sha256"] = contract_sha

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.yaml"
    contract_path = run_dir / "run_contract.json"

    if contract_path.is_file():
        if _json(contract_path, "Existing run contract") != contract:
            raise RuntimeError("Existing W0-RAS run contract berbeda.")
    else:
        config_path.write_text(frozen_config_text, encoding="utf-8")
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )

    result_path = run_dir / "result.json"
    if result_path.is_file():
        previous = _json(result_path, "Existing W0-RAS result")
        if previous.get("run_contract") != contract:
            raise RuntimeError("Existing W0-RAS result berasal dari kontrak berbeda.")
        return previous

    lock_name = f"W0_RAS_fold{fold}_seed{seed}.training.lock"
    with exclusive_training_lock(output_root, lock_name=lock_name):
        complete = _run_complete(run_dir, int(cfg["training"]["epochs"]))
        training_executed = False
        if not complete:
            train_summary = train_w0_ras(
                cfg,
                run_dir=run_dir,
                run_contract_sha256=contract_sha,
                expected_initial_model_sha256=static[
                    "expected_initial_model_sha256"
                ],
                resume=(run_dir / "last.pt").is_file(),
            )
            training_executed = True
        else:
            last = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=False
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
    result = {
        "format": "bilinear_lmmd.w0_ras.arm_result.v1",
        "protocol": contract["protocol"],
        "scientific_evidence": True,
        "fold": fold,
        "seed": seed,
        "metrics": {
            key: metrics[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "worst_class_f1",
                "hard_class_f1",
            )
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
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--development-contract", required=True, type=Path)
    parser.add_argument("--static-preflight", required=True, type=Path)
    parser.add_argument("--observability-audit", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-commit", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()

    run_w0_ras(
        fold=args.fold,
        seed=args.seed,
        data_root=args.data_root,
        development_contract=args.development_contract,
        static_preflight=args.static_preflight,
        observability_audit=args.observability_audit,
        environment=args.environment,
        output_root=args.output_root,
        required_commit=args.required_commit,
        authorize_training=args.authorize_training,
        device=args.device,
    )


if __name__ == "__main__":
    main()
