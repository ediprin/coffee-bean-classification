# WR-HBP checkpoint-loss recovery audit — 2026-09-30

Status: **pre-outer-test audit complete**

## Scope

This audit covers the Coffee17 WR-HBP V1 development-to-final-confirmation
handoff after the original saved development checkpoint output could no longer
be located.

The audit does **not** reopen method search. It checks only whether one
transparent checkpoint-loss recovery can preserve the frozen WR-HBP scientific
contract before any outer-test identity is materialized.

## Historical anchors

- Original WR-HBP development scientific commit:
  `01c9212965bc9040ef151204b9404d564f523a0f`
- Frozen final-inference commit:
  `b62963faec8e337359ba4244a3ee5814709bc18f`
- Audited checkpoint-loss recovery runtime:
  `9004462d4c114e594aced9954a56435231c84ac3`
- Supported Kaggle entrypoint:
  `notebooks/Coffee17_WR_HBP_Recovery_And_Final_V2_Kaggle.ipynb`

## What failed before this audit

The first recovery-capable notebook pinned
`98229149291f1f0f54a71cc9b3cc291b81825ab7` and ran the full
`tests/experiments/test_wr_hbp_final_confirmation.py` from that historical
checkout.

That historical test file still contained notebook-source assertions from the
older inference-only workflow, including an assertion that
`--authorize-training` must not appear in the notebook. The recovery workflow
correctly contains `--authorize-training` before the outer-test boundary, so
the historical notebook-source test failed before recovery training began.

This was an **operational preflight failure**, not a model-training failure and
not an outer-test failure.

## Why the recovery runtime is pinned to 9004462

The audited `9004462...` runtime retains the frozen WR-HBP V1
architecture/configuration and adds two recovery-safety checks that were absent
from the earlier `982291...` pin:

1. the strict recovery runner fails before training if the shared RGB-HBP
   initialization fingerprint differs from the original WR-HBP V1 fingerprint;
2. the final-test authority independently verifies that same original
   fingerprint in every fold and arm contract.

Its test fixtures were updated for those guards, and the exact `9004462...`
commit passed repository CI before being registered as the recovery runtime.

Kaggle runtime preflight now excludes tests whose purpose is to inspect the
repository notebook source itself (`-k "not kaggle_notebook"`). Those
source-level notebook assertions remain covered by branch CI. Kaggle runtime
preflight instead checks the scientific/runtime implementation that will
actually train/evaluate the models.

## Scientific equivalence audit

The following scientific model/data/final-inference files are byte-identical between the audited recovery runtime
`9004462...` and frozen final-inference commit `b62963...`:

- `configs/wavelet_residual_hbp/WR_HBP_V1.yaml`
- `src/bilinear_lmmd/modeling/wavelet_residual_hbp.py`
- `src/bilinear_lmmd/modeling/models.py`
- `src/bilinear_lmmd/engine/wavelet_residual_hbp.py`
- `src/bilinear_lmmd/data/preprocessing/study_data.py`
- `src/bilinear_lmmd/data/preparation/materialize_preprocessing_test.py`
- `src/bilinear_lmmd/experiments/run_wr_hbp_final_outer_fold.py`
- `src/bilinear_lmmd/experiments/run_wr_hbp_final_outer_summary.py`

The original WR-HBP V1 YAML and training engine are also byte-identical between
the original development commit `01c921...` and the audited recovery runtime.
The relevant model-code changes since the original development commit are
strict-determinism compatibility paths only:

1. VisuShrink median is computed on CPU only when deterministic CUDA algorithms
   are enabled, avoiding nondeterministic CUDA reduction while preserving the
   same threshold definition.
2. HBP spatial alignment uses exact block means only in strict deterministic
   mode when the feature-map ratios are integral; otherwise the ordinary path
   remains adaptive average pooling.
3. The matched recovery runner enables deterministic algorithms, disables
   cuDNN benchmark, enables cuDNN deterministic mode, disables TF32, and pins
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

No WR-HBP architecture, loss, optimizer, schedule, HBP stages, projection
dimension, wavelet bands, injection point, residual width, or gate definition
is changed.

## Frozen recovery training contract

- Coffee17 original population: 979 mounted images / 17 classes before content
  cleaning.
- Clean population/fold manifest is reconstructed by the existing provenance
  pipeline.
- 5 folds, seed 42.
- validation ratio = 0.10.
- Each fold materializes only train + validation during recovery.
- Locked fold test identities are not copied into the recovery runtime.
- R0-HBP and WR-HBP are matched on fold identities and shared-core
  initialization.
- 50 epochs.
- AdamW, lr 3e-4, weight decay 1e-4.
- Cross-entropy with label smoothing 0.1.
- Cosine scheduler.
- Image size 224, batch size 32.
- MobileNetV3-Large, HBP stages [1,3,4], projection 512, dropout 0.2.
- WR branch: luminance L1 Haar, LH/HL/HH, VisuShrink soft threshold,
  16-channel shallow residual, tanh zero-init gate.

## Frozen recovery gate

No outer-test identity may be materialized unless recovery development satisfies
all four original WR-HBP development criteria:

1. mean Macro-F1 delta > 0;
2. Macro-F1 delta > 0 on at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0.

If any criterion fails, the recovery stops and outer confirmation is not
authorized.

## Authority boundary

Before any outer-test materialization, the authority builder checks all five
folds and both arms for:

- protocol;
- seed;
- matched initialization and validation rows;
- completed 50-epoch checkpoints;
- checkpoint SHA-256 agreement with `pair_result.json`;
- run contract;
- `outer_test_accessed = false`;
- strict deterministic recovery metadata when recovery mode is used;
- the frozen four-part development gate.

Recovery authority must record:

- `development_mode = checkpoint_loss_recovery_v1`;
- recovery development commit = `9004462...`;
- `test_images_accessed = false`;
- `further_primary_tuning_authorized = false`.

## Outer-test boundary

After authority passes, the notebook checks out the original frozen
final-inference commit `b62963...` **before** any test identity is
materialized.

From the `# 4. ONE-SHOT outer test` section onward:

- no `--authorize-training` is present;
- no training is permitted;
- no checkpoint reselection is permitted;
- no method or hyperparameter changes are permitted.

Each fold test subset is materialized only after authority, evaluated for both
frozen arms, then the temporary test runtime is destroyed.

## Final decision gate

WR-HBP is confirmed only if all hold on outer predictions:

1. pooled Macro-F1 delta > 0;
2. positive Macro-F1 delta on at least 3/5 outer folds;
3. pooled Hard-F1 delta >= 0;
4. pooled Worst-F1 delta >= 0.

Otherwise the decision is `WR_HBP_NOT_CONFIRMED`.

The paired class-stratified 10,000-replicate bootstrap is informational and does
not modify this gate.

## Operational entrypoint

Use only:

`notebooks/Coffee17_WR_HBP_Recovery_And_Final_V2_Kaggle.ipynb`

Expected beginning of a checkpoint-loss run:

```text
NOTEBOOK_BUILD: WR-HBP-RECOVERY-AND-FINAL-V2
DEVELOPMENT SOURCE: exact checkpoint project/bundle not found
RECOVERY MODE: checkpoint_loss_recovery_v1
Outer test remains locked until deterministic development recovery passes.
```

A failure before the first `RECOVERY FOLD 1/5` line is a preflight/runtime
failure and does not count as a recovery training attempt or outer-test access.
