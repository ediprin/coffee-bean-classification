# W0-RAS V1 Handoff

## Completed evidence

The Coffee17 frequency/texture cue audit has been completed on 965 exact clean OOF
identities.

Authoritative summary:

docs/evidence/W0_CUE_AUDIT_V1_SUMMARY.md

Key completed result:

- W0 relative retained-energy cues after VisuShrink formed the clearest
  FDR-significant family associated with W0 true-class probability improvement
  over R0.
- the strongest nonredundant candidate is mean-RGB L1-HH retained-energy ratio;
- F0 angular cues did not yield a stable FDR-significant selective-transfer
  candidate.

## Frozen experiment

Protocol:

docs/protocols/W0_RAS_V1.md

Implementation:

- configs/w0_ras/W0_RAS.yaml
- src/bilinear_lmmd/experiments/run_w0_ras.py
- src/bilinear_lmmd/experiments/aggregate_w0_ras.py

W0-RAS V1 is a raw-RGB MobileNetV3-Large GAP model with one training-only scalar
regression head. The target is the standardized mean-RGB level-1 HH
retained-energy ratio from the exact frozen Haar + VisuShrink operators.

The auxiliary head is discarded at inference.

## Status boundary

**No W0-RAS training result exists yet.**

Until all five development folds are complete and aggregated, W0-RAS must not be
used as evidence that selective wavelet retention transfer improves Coffee17.

The already-opened outer-test OOF population is not authorized for W0-RAS
development or screening.

## Promotion gate

All must hold on the five development folds:

- mean Macro-F1 delta > 0;
- Macro-F1 improves on at least 3/5 folds;
- mean Hard-F1 delta >= 0;
- mean Worst-F1 delta >= 0;
- epoch-1 weighted auxiliary/CE ratio <= 1.0 on every fold.

Otherwise decision is STOP_W0_RAS_V1.
