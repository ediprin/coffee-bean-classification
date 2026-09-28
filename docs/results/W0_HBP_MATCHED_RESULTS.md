# W0-HBP Matched Results

Date analyzed: **2026-09-28**

Source artifact:

- `w0-hbp-matched-analysis-package.zip`
- SHA-256: `f12670b3691dd2e482962d3edaa6fee1bfafdef9b600ceaccf87592952640bf4`

Protocol: `coffee17-w0-hbp-matched-v1`.

## Integrity

The uploaded package contains all 5 folds.

For every fold:

- seed = 42;
- validation count = 97;
- R0-HBP and W0-HBP use the same validation rows;
- both arms use the same initial model-state fingerprint;
- R0-HBP was retrained as the matched control;
- W0-HBP was trained as the candidate;
- outer test was not accessed.

The common initial model-state SHA-256 is:

`6d425c6c149005afde1224ed74ccfe4eb84c95e574d21e25a0e728578d79f9cc`

The five validation folds contain 485 unique validation paths in total.

## Aggregate result

| Metric | R0-HBP | W0-HBP | W0 - R0 |
|---|---:|---:|---:|
| Accuracy | 91.55% ± 1.98 | 90.10% ± 2.37 | -1.44 ± 0.92 pp |
| Balanced Accuracy | 91.63% ± 2.42 | 90.13% ± 2.89 | -1.50 ± 0.90 pp |
| Macro-F1 | 91.27% ± 2.22 | 90.04% ± 2.46 | -1.23 ± 0.90 pp |
| Hard-F1 | 86.23% ± 3.54 | 84.09% ± 4.33 | -2.13 ± 1.97 pp |
| Worst-F1 | 64.95% ± 6.16 | 63.78% ± 10.19 | -1.17 ± 8.74 pp |

## Per-fold Macro-F1

| Fold | R0-HBP | W0-HBP | Delta |
|---|---:|---:|---:|
| 1 | 92.23% | 92.37% | +0.14 pp |
| 2 | 92.80% | 91.80% | -1.00 pp |
| 3 | 93.42% | 91.19% | -2.23 pp |
| 4 | 88.24% | 86.91% | -1.33 pp |
| 5 | 89.66% | 87.90% | -1.75 pp |

Positive Macro-F1 folds: **1/5**.

Hard-F1 improved in **1/5** folds.

Worst-F1 improved in **2/5** folds, tied in fold 1, and dropped strongly in
fold 4.

## Matched prediction outcomes

Across the 485 paired validation observations:

- W0 rescued an R0-HBP error: 5;
- W0 damaged an R0-HBP correct prediction: 12;
- both correct: 432;
- both wrong: 36.

Net top-1 effect: **-7 correct predictions** for W0-HBP.

## Hard-group behavior

Mean W0-HBP minus R0-HBP delta:

- sour/black: **-4.80 pp**;
- shape/withered: **+0.71 pp**;
- insect damage: **-2.41 pp**.

The aggregate Hard-F1 decline is driven mainly by sour/black and insect-damage
groups.

## Per-class mean F1 delta

Largest mean gains:

- Fade: +4.81 pp;
- Immature: +2.72 pp;
- Withered: +2.55 pp.

Largest mean losses:

- Partial Sour: -7.78 pp;
- Severe Insect Damage: -4.64 pp;
- Floater: -4.44 pp;
- Partial Black: -3.36 pp;
- Full Sour: -3.27 pp;
- Cut: -3.13 pp.

Dry Cherry, Full Black, Husk, and Parchment had zero mean delta across these
five validation folds.

## Screening gate

Frozen gate:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 positive in at least 3/5 folds;
3. mean paired Hard-F1 delta >= 0;
4. mean paired Worst-F1 delta >= 0.

Observed:

- Macro mean positive: **FAIL**;
- Macro positive >= 3/5 folds: **FAIL**;
- Hard mean nonnegative: **FAIL**;
- Worst mean nonnegative: **FAIL**.

Final gate decision: **FAIL**.

## Scientific conclusion

Direct W0 preprocessing does not improve HBP under the frozen matched
Coffee17 preprocessing-study protocol.

The candidate decreases mean Macro-F1 by 1.23 pp and Hard-F1 by 2.13 pp, with
Macro-F1 improving in only 1/5 folds. The negative result is stronger than the
original GAP preprocessing point estimate because this experiment isolates the
interaction between W0 preprocessing and HBP with a reconstructed matched
R0-HBP control.

Decision:

> **Stop W0-HBP. Do not tune wavelet level, VisuShrink threshold, HBP stages,
> image resolution, or loss on these reused development folds.**

This result closes the direct W0+HBP path under the current development
protocol. It does not invalidate the earlier preprocessing complementarity and
cue-analysis findings; it shows that feeding W0 directly into HBP is not an
effective way to exploit them.
