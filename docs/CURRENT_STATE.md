# Current project state

Snapshot date: **2026-07-23**

This is a mutable handoff snapshot. It describes what was active when the file
was last updated; it is not evidence that the active method is superior.
Agents must verify it against protocols, raw reports, and the experiment log.

## Current user direction

- Repository scope: coffee-bean classification, not the separate YOLO/detection
  project.
- Immediate task type: closed-set fine-grained classification.
- OSR and LMMD/UDA are not currently requested.
- Avoid additional expensive training without a frozen, literature-grounded
  comparison.

## Current runnable screen

The most recently prepared—not yet empirically validated—experiment is the
Coffee17 multistage recalibration screen:

| Code | Model |
|---|---|
| BE2G | EfficientNetV2-B0 + GAP baseline |
| BE2H | EfficientNetV2-B0 + HBP comparator |
| MSF0 | Fixed three-stage spatial fusion |
| MSF1 | Adaptive stage/channel multistage recalibration |

Relevant files:

- `docs/protocols/COFFEE17_MULTISTAGE_RECALIBRATION_V1.md`
- `src/bilinear_lmmd/modeling/multistage_recalibration.py`
- `src/bilinear_lmmd/experiments/run_multistage_recalibration_screening.py`
- `notebooks/coffee17_multistage_recalibration_colab.ipynb`

Current gate:

- validation only;
- seed 42 only;
- test locked;
- MSF1 must pass both `MSF0_vs_MSF1` and `BE2G_vs_MSF1`;
- Macro-F1 and hard-class F1 must improve;
- bottom-three class F1 may not decline by more than one point.

If the screen passes, the protocol requires a capacity-matched control before
additional seeds or test. If it fails, record it and stop this direction.

## Dataset snapshot

### Coffee17

- 979 original images;
- 965 after the clean grouped audit;
- fold-1: 669 train, 97 validation, 199 test;
- current MSF screen uses this dataset because it is faster and directly
  comparable with existing BE2G/BE2H checkpoints.

### SNI instance crops

- 31,074 audited crops;
- 21 shared classes;
- grouped by source image;
- available for later work but not used by the current quick MSF screen.

## Important prior evidence

These are summaries only. Use
`docs/results/EXPERIMENT_MASTER_LOG.md` and the raw reports for exact values.

- EfficientNetV2-B0 was stronger than MobileNetV3 in the controlled Coffee17
  backbone comparison.
- HBP showed a positive average Coffee17 effect in some protocols but was
  seed- and backbone-sensitive and was not universally useful across datasets.
- Several candidates produced a favorable single seed and failed multi-seed
  confirmation or lower-tail criteria.
- SNI multiresolution backbone screening passed its initial gate, while the
  ontology extension and selective residual HBP diagnostic failed.

## Paused or stopped work at this snapshot

The following must not be resumed automatically:

- open-set recognition;
- LMMD/UDA robustness;
- SNI ontology expert extension;
- selective residual HBP on SNI;
- Jiao Swin-HSSAM training, which was cancelled after Colab state loss.

Other completed failures and mixed results are listed in the master log.
Their status can be revisited only with a new explicit hypothesis, not merely
to seek a positive seed.

## Persistence status

- Branch: `agent/sni-instance-crops`.
- Source code is pushed to GitHub.
- Generic per-epoch Hugging Face checkpoint persistence is implemented.
- Current notebook artifact namespace:
  `coffee17-multistage-recalibration-v1`.
- Current artifact repository:
  `ediprin/coffee-backbone-checkpoints`.
- A write-enabled `HF_TOKEN` is required in each Colab account.

## Result status

At this snapshot:

- MSF architecture, runner, notebook, protocol, and tests exist;
- MSF0/MSF1 training results do not yet exist;
- Coffee17 test has not been opened for this experiment.

