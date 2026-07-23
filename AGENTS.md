# Agent guide — Coffee Bean Classification

## Read this first

This repository is the **classification** research codebase. It is not the
coffee-bean detection/YOLO repository. Do not redirect classification work to
YOLO unless the user explicitly asks to work in the separate detection repo.

The GitHub repository is:

```text
https://github.com/ediprin/coffee-bean-classification
```

The local folder may still be named `bilinear-LMMD`, and the Python package is
still `bilinear_lmmd` for checkpoint and command compatibility. Do not rename
the Python namespace casually.

## Required reading order

Before proposing or implementing another experiment, read:

1. `AGENTS.md` (this file);
2. `README.md`;
3. `docs/architecture/REPOSITORY_STRUCTURE.md`;
4. `docs/results/EXPERIMENT_MASTER_LOG.md`;
5. the protocol for the currently active experiment.

For claims about prior literature, inspect the actual PDF or curated
literature table under `docs/`; do not reconstruct paper methods from memory.

## Current research scope

The active problem is **closed-set fine-grained coffee-bean defect
classification**. The immediate screening dataset is Coffee17. OSR, LMMD/UDA,
and YOLO detection are not the current direction.

The current active experiment is:

```text
Coffee17 multistage recalibration v1
BE2G: EfficientNetV2-B0 + GAP baseline
BE2H: EfficientNetV2-B0 + HBP comparator
MSF0: fixed three-stage spatial fusion
MSF1: adaptive stage/channel multistage recalibration
```

Read:

- `docs/protocols/COFFEE17_MULTISTAGE_RECALIBRATION_V1.md`
- `src/bilinear_lmmd/modeling/multistage_recalibration.py`
- `src/bilinear_lmmd/experiments/run_multistage_recalibration_screening.py`
- `notebooks/coffee17_multistage_recalibration_colab.ipynb`

Screening is locked to Coffee17 validation and seed 42. The test set and
seeds 123/2026 must not be opened merely because training is available.

## Current decision gate

MSF1 only passes initial screening if both comparisons pass:

```text
MSF1 versus MSF0
MSF1 versus BE2G
```

For each required comparison:

- Macro-F1 must improve;
- hard-class F1 must improve;
- mean F1 of the bottom three classes must not drop by more than 1 point.

Worst-class F1 is reported but is not the sole gate because Coffee17 has small
per-class validation support. If screening passes, add the capacity-matched
control described in the protocol before running more seeds or test.

## Dataset facts

### Coffee17

- Public source: Coffee Green Bean with 17 Defects (original).
- Original images: 979.
- Clean grouped dataset used by the current protocol: 965 images.
- Fold-1 counts: train 669, validation 97, test 199.
- Splits are grouped/deduplicated before online train augmentation.
- Current selection split: validation.
- Test remains locked.

The Colab notebook prepares Coffee17 automatically. Do not upload or create a
new split ad hoc when the existing preparation pipeline is available.

### SNI instance crops

- Built from object-detection and instance-segmentation archives.
- Audited output: 31,074 crops, 21 shared classes.
- Split by source image, not independently by crop.
- This dataset is valid for later classification work, but it is not the
  dataset used by the current fast MSF screening.

Read `docs/protocols/SNI_INSTANCE_CROP_PROTOCOL.md` before touching it.

## Evidence already obtained

Do not present HBP as universally superior:

- HBP helped Coffee17 on average, especially lower-tail/hard metrics, but its
  effect varied by seed and backbone.
- EfficientNetV2-B0 was stronger than MobileNetV3 in the controlled backbone
  comparison.
- HBP was neutral or harmful on some coarser/different public datasets.
- Many proposed additions failed multi-seed or safety criteria even after a
  positive seed-123 screening.

The experiment log is authoritative:

```text
docs/results/EXPERIMENT_MASTER_LOG.md
```

Check it before suggesting HBP, hierarchy, ArcFace, LMMD, attention, DSConv,
MPN-COV, factorized bilinear convolution, PMG, or other previously tested
ideas.

## Stopped or paused directions

Do not silently restart these:

- Open-set recognition: stopped by the user.
- LMMD/UDA robustness work: not the current thesis direction.
- SNI ontology expert SNIB2: failed screening.
- Selective residual HBP on SNI: stopped after failure.
- Hierarchical auxiliary heads on Coffee17: failed the full gate.
- SPPF-attention, MPN-COV, confusion-aware pairwise loss, DSConv confirmation,
  factorized bilinear convolution, and progressive multigranularity: retain
  their recorded negative/mixed results; do not rerun without a new,
  literature-grounded hypothesis.
- Jiao Swin-HSSAM training: paused/cancelled after Colab state loss. Its code
  remains for reference, but do not resume it automatically.

Negative results are research evidence. Do not erase, reinterpret, or rerun
them just because one seed looked favorable.

## Repository map

```text
configs/                         experiment YAML files by concern
docs/architecture/               repository design
docs/protocols/                  frozen experiment protocols
docs/results/                    verified positive and negative results
notebooks/                       Colab entry points
src/bilinear_lmmd/core/          config and artifact persistence
src/bilinear_lmmd/data/          loaders and dataset preparation
src/bilinear_lmmd/engine/        generic train/evaluate code
src/bilinear_lmmd/experiments/   controlled study runners
src/bilinear_lmmd/modeling/      models, heads, and losses
src/bilinear_lmmd/reporting/     aggregation and reports
tests/                           unit and integration tests
```

Keep reusable training/evaluation logic in `engine`, model definitions in
`modeling`, and experiment-specific orchestration in `experiments`.

## Experiment safety rules

1. Freeze the research question, split, metrics, and acceptance gate before
   expensive training.
2. Use validation for model selection. Do not inspect test to decide whether a
   method is worth keeping.
3. Start with one seed only when the protocol explicitly defines a fail-fast
   screen.
4. Do not expand to additional seeds until the screen passes.
5. Reuse compatible BE2G/BE2H checkpoints instead of retraining baselines.
6. Compare using the same dataset preparation, split, augmentation, seed,
   image resolution, epochs, optimizer, and selection metric unless the
   ablation explicitly changes one of them.
7. Report Macro-F1, hard-class F1, bottom-tail/worst-class behavior, and
   efficiency. Accuracy alone is insufficient.
8. Never claim localization from image-level labels or CAM alone.
9. Never call a one-seed result final.
10. Record failures in `docs/results/EXPERIMENT_MASTER_LOG.md`.

## Persistent training requirements

Git stores source code, not model weights. Long Colab training must use:

- GitHub for code/notebooks/configs;
- Google Drive for a readable result copy;
- Hugging Face model repository for cross-account checkpoint restore.

The generic trainer supports:

```text
--artifact-repo
--artifact-path
--artifact-sync-every
--artifact-required
```

Colab must define a write-enabled `HF_TOKEN` secret. For protected runs, sync
every epoch and enable required mode. Never start a long experiment that only
writes to `/content`.

Current artifact repository and namespace:

```text
repo:      ediprin/coffee-backbone-checkpoints
namespace: coffee17-multistage-recalibration-v1
```

On reset or account change, use the same authorized HF token and rerun the
notebook. The runner restores `last.pt`, `best.pt`, `history.json`, config, and
reports before deciding whether to resume.

## Validation commands

Run targeted tests while editing, then the full suite before publishing:

```powershell
python -m pytest tests/modeling/test_multistage_recalibration.py `
  tests/experiments/test_multistage_recalibration_screening.py `
  tests/integration/test_multistage_recalibration_notebook.py -q

python -m pytest -q
```

Do not claim that training was performed merely because architecture or runner
tests pass.

## Working-tree and publishing rules

- Preserve unrelated user changes.
- Do not use destructive Git commands.
- Stage only intended files.
- Keep checkpoints, datasets, and generated outputs out of Git.
- Commit protocols, code, tests, notebooks, and verified result summaries.
- Push the branch used by the notebook so a new Colab account receives the
  same code.

## Handoff status

At the time this guide was created:

- branch: `agent/sni-instance-crops`;
- active notebook:
  `notebooks/coffee17_multistage_recalibration_colab.ipynb`;
- active screen: MSF0/MSF1 on Coffee17 validation seed 42;
- training result: not yet available;
- test opened: no;
- source/checkpoint persistence: implemented.

When results arrive, update this section and the master experiment log rather
than relying on chat history.
