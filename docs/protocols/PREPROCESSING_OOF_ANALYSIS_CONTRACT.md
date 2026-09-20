# Coffee17 preprocessing study — OOF and final-analysis contract v1

Status: **implementation draft; execute only after primary training is frozen**

## Pre-test authority

Outer-test inference is not authorized by the presence of checkpoints alone.
`run_preprocessing_primary_confirmation.py` requires exactly:

- R0, C0, F0 and W0;
- folds 1–5;
- seed 42;
- 50 completed primary epochs;
- validation-only result reports;
- `scientific_evidence=true`;
- `test_images_accessed=false`;
- checkpoint hashes matching each primary result;
- one common code/runtime/data/model-initialization contract.

It then writes an authority whose decision is
`AUTHORIZE_OOF_TEST_EVALUATION` and closes further primary tuning.

## OOF opening

`run_preprocessing_oof.py` is inference-only and requires
`--authorize-test`. For one fold at a time, it materializes only that frozen
outer-test fold into a temporary directory, evaluates all four matched
checkpoints, persists reports and destroys the temporary test runtime.

Cached test reports are reusable only when checkpoint SHA, authority SHA and
fold-test identity SHA are unchanged.

## Paired OOF population

Every clean identity must appear exactly once per arm in OOF and all four arms
must have identical identity and ground-truth sets.

Primary estimands:

- C0 Macro-F1 minus R0 Macro-F1;
- F0 Macro-F1 minus R0 Macro-F1;
- W0 Macro-F1 minus R0 Macro-F1.

No winner/ranking is part of the protocol.

## Bootstrap

The primary uncertainty analysis is a 10,000-iteration paired stratified
bootstrap. Sampling is with replacement within each of the 17 actual classes
and the same sampled identities are used for R0 and the compared arm.

This estimates OOF sample uncertainty conditional on the frozen seed-42 trained
models. It must not be described as full training-seed uncertainty.

## Explanatory analysis

Per-class F1 effects and candidate-minus-R0 confusion-count matrices are
secondary explanatory analyses. They do not redefine primary arms or metrics.

## Efficiency

Model parameter count is common across arms. Efficiency reporting separately
measures pre-model processing, CNN-only latency and end-to-end latency on
preloaded resized images. C0's canonical OpenCV frontend is identified as CPU;
F0/W0 tensor frontends run on the selected tensor device.

## Closure

After OOF is opened:

`primary_protocol_closed=true`

and further primary tuning is not authorized. Any later preprocessing variant
is post-primary exploratory work unless evaluated on a new independent dataset.
