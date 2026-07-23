# Agent guide — Coffee Bean Classification

## Purpose of this file

This file contains **stable operating rules**, not a preferred algorithm and
not a transcript-derived research conclusion. Current experiments and recent
decisions belong in `docs/CURRENT_STATE.md`.

An agent must not treat the newest experiment, newest chat message, or most
recently edited file as automatically superior.

## Repository identity

This is the coffee-bean **classification** research repository:

```text
https://github.com/ediprin/coffee-bean-classification
```

The local directory may still be named `bilinear-LMMD`, and the historical
Python namespace remains `bilinear_lmmd` for compatibility. This repository is
separate from the coffee-bean detection/YOLO project.

## Required orientation

Before proposing or implementing work, read:

1. `AGENTS.md`;
2. `README.md`;
3. `docs/CURRENT_STATE.md`;
4. `docs/architecture/REPOSITORY_STRUCTURE.md`;
5. `docs/results/EXPERIMENT_MASTER_LOG.md`;
6. the protocol and verified result files relevant to the requested task.

If these sources disagree, do not silently pick the newest one. Report the
conflict and prefer, in order:

1. raw report/checkpoint metadata;
2. frozen experiment protocol;
3. verified result document;
4. master experiment log;
5. dated current-state snapshot;
6. README summary;
7. chat recollection.

## Research neutrality

- Do not assume HBP, GAP, attention, multiscale fusion, transformers, metric
  learning, hierarchy, or any other method is the center of the thesis.
- Do not choose a method merely because it is recent, complex, or produced one
  favorable seed.
- Do not discard a method merely because it failed on a different dataset or
  label granularity.
- Separate dataset effects, backbone effects, head effects, loss effects, and
  protocol effects through controlled comparisons.
- Treat negative results as evidence, not as files to ignore or experiments to
  rerun until positive.
- Literature claims must be checked against the actual paper/PDF or curated
  literature table in `docs/`, not reconstructed from model memory.

## Experimental safeguards

1. Define the research question, dataset version, split, metrics, baselines,
   and acceptance rule before expensive training.
2. Use validation for model selection. Test must remain locked until the
   protocol explicitly authorizes it.
3. A one-seed result is a screen, not a final conclusion.
4. Expand seeds only after the predefined screen passes.
5. Reuse compatible checkpoints; do not retrain baselines unnecessarily.
6. Change one causal factor at a time or include an explicit
   capacity/compute-matched control.
7. Report Macro-F1 and per-class/lower-tail behavior. Accuracy alone is
   insufficient for imbalanced fine-grained tasks.
8. Report parameter count, model size, and latency when claiming practical
   superiority.
9. Do not claim defect localization from image-level labels or CAM alone.
10. Record both successful and failed completed experiments in the master log.

## Dataset safeguards

- Use the repository preparation pipeline instead of inventing ad hoc splits.
- Deduplicate/group by source identity before splitting.
- Apply stochastic augmentation to training only.
- Never move augmented siblings or crops from one source image across splits.
- Record class counts and imbalance explicitly.
- Do not merge datasets with incompatible labels without a frozen mapping and
  an audit.
- Dataset-specific conclusions must not be generalized automatically to all
  coffee-defect datasets.

## Repository architecture

```text
configs/                         experiment configurations
docs/architecture/               repository design
docs/protocols/                  frozen experimental protocols
docs/results/                    verified results and negative evidence
notebooks/                       Colab entry points
src/bilinear_lmmd/core/          configuration and artifact persistence
src/bilinear_lmmd/data/          loaders and dataset preparation
src/bilinear_lmmd/engine/        generic training/evaluation
src/bilinear_lmmd/experiments/   controlled study runners
src/bilinear_lmmd/modeling/      architectures, heads, and losses
src/bilinear_lmmd/reporting/     aggregation and reporting
tests/                           unit and integration tests
```

Keep reusable execution logic in `engine`, model definitions in `modeling`,
experiment orchestration in `experiments`, and factual outcomes in
`docs/results`.

## Persistent training

Git stores source code, not trained weights. Long remote training must use:

- GitHub for code, protocols, configs, notebooks, and tests;
- Google Drive or equivalent for a readable result copy;
- a remote artifact store such as Hugging Face for cross-account checkpoint
  recovery.

The generic trainer supports:

```text
--artifact-repo
--artifact-path
--artifact-sync-every
--artifact-required
```

Never start a long Colab job that stores its only checkpoint under `/content`.
Do not put access tokens directly in a notebook or commit.

## Code-change rules

- Inspect the working tree before editing.
- Preserve unrelated user changes.
- Do not use destructive Git commands.
- Add or update tests for behavioral changes.
- Run targeted tests, then the full suite before publishing.
- Do not claim a model was trained because architecture/unit tests passed.
- Keep datasets, checkpoints, tokens, and generated outputs out of Git.
- Stage only intended files and use an explicit commit message.

## Updating project state

`docs/CURRENT_STATE.md` is a dated handoff, not permanent truth.

After a material experiment:

1. save raw reports;
2. update the appropriate result document;
3. update `EXPERIMENT_MASTER_LOG.md`;
4. replace the dated snapshot in `CURRENT_STATE.md`;
5. do **not** rewrite `AGENTS.md` to favor the latest method.
