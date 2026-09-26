# MVFD-SBN Final Ablation Summary

## Main controlled sequence

| Experiment | Main change | Macro-F1 | Delta vs matched R0 | Interpretation |
|---|---|---:|---:|---|
| R0_CONTROL | raw RGB baseline | 90.65% | -- | matched deployment baseline |
| MVCE_ALL4 | fully shared multi-view CE with R0 BN statistics | 88.93% | -1.72 pp | transformed-view optimization failure |
| MVCE-SBN | add Selective BatchNorm | 90.68% | +0.02 pp | fixes optimization collapse, no meaningful classifier gain |
| AT-SBN | separate auxiliary heads + R0->aux distillation + late merging | 90.67% | +0.02 pp | transfer direction does not improve R0 |
| AUXCE-SBN | SBN + training-only auxiliary CE, no feature transfer | 89.94% | -0.71 pp | auxiliary multi-view supervision alone is insufficient |
| **MVFD-SBN** | explicit C0/F0/W0 feature -> R0 transfer | **91.77%** | **+1.12 pp** | retained final method |
| CA-MVFD-SBN | CE-confidence teacher weighting | 91.38% | +0.73 pp | more complex aggregation does not improve frozen MVFD |

## Final causal comparison

The most important comparison is not MVFD-SBN versus every historical
experiment. It is the matched objective ablation:

`AUXCE-SBN`

`L = CE_R + 0.05(CE_C+CE_F+CE_W)`

versus:

`MVFD-SBN`

`L = CE_R + 0.05(CE_C+CE_F+CE_W) + 0.007 L_feat`

Results:

| Metric | AUXCE-SBN | MVFD-SBN | MVFD - AUXCE |
|---|---:|---:|---:|
| Accuracy | 90.52% | 91.55% | +1.03 pp |
| Balanced Accuracy | 90.13% | 92.09% | +1.96 pp |
| Macro-F1 | 89.94% | 91.77% | +1.83 pp |
| Hard-F1 | 86.08% | 85.26% | -0.82 pp |
| Worst-F1 | 61.58% | 67.88% | +6.30 pp |

Per-fold Macro-F1 difference, MVFD-SBN minus AUXCE-SBN:

- fold 1: +1.90 pp
- fold 2: +0.35 pp
- fold 3: +0.07 pp
- fold 4: +5.37 pp
- fold 5: +1.44 pp

The feature-transfer contribution is therefore positive on all five folds for
Macro-F1 in this exploratory protocol.

## Mechanistic ablation

AUXCE-SBN:
- R0-teacher cosine: ~0.95295
- R0-teacher L2: ~6.429
- raw feature discrepancy: ~43.572

MVFD-SBN:
- R0-teacher cosine: ~0.98997
- R0-teacher L2: ~2.430
- raw feature discrepancy: ~6.236

The explicit feature loss materially changes the R0 representation in the
intended direction.

## Why the confidence-aware refinement was rejected

CA-MVFD-SBN changes only teacher aggregation.

Results relative to equal-mean MVFD:
- Macro-F1: -0.384 pp
- Balanced Accuracy: -0.286 pp
- Hard-F1: -0.221 pp
- Worst-F1: -2.303 pp
- positive Macro-F1 folds: 1/5

The confidence weights remained almost uniform. Therefore the extra mechanism
added complexity without producing a more informative teacher.

## Main limitation

The final method improves Macro-F1, Balanced Accuracy, and Worst-F1, but lowers
Hard-F1 relative to matched R0 by 1.75 pp.

This trade-off must remain visible in the thesis. The final conclusion is an
aggregate representation improvement, not a universal improvement across all
hard classes.
