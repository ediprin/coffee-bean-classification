# WRC-HBP V1 Results — FAIL

Date analyzed: **2026-09-28**

Source artifact:

- `wavelet-clahe-residual-hbp-analysis-package.zip`
- SHA-256: `4b698e98a01ea81e894927cbf7b50303a0ac9526f94602f860232b85141e003f`

Scientific commit:

`762e0af63b911b9fa2c695b16214e37aa085a723`

Protocol:

`coffee17-wavelet-clahe-residual-hbp-v1`

## Integrity

All five matched folds completed with seed 42.

For every fold:

- WR-HBP and WRC-HBP used the same validation identities/labels;
- the complete initial WR-HBP shared state was matched;
- initial logits were exactly equal before training;
- WR-HBP was retrained as the matched control;
- WRC-HBP was trained as the candidate;
- outer test was not accessed.

Common shared-WR initial SHA-256:

`9696393683ba7a4c75d96f78ba223d546c0836be251a1d767ee75cc615938cf2`

## Aggregate result

| Metric | WR-HBP | WRC-HBP | Delta WRC-WR |
|---|---:|---:|---:|
| Accuracy | 91.34% | 91.34% | 0.00 pp |
| Balanced Accuracy | 91.72% | 91.34% | -0.38 pp |
| Macro-F1 | 91.44% | 91.08% | **-0.36 pp** |
| Hard-F1 | 86.02% | 85.93% | **-0.08 pp** |
| Worst-F1 | 64.62% | 64.43% | **-0.19 pp** |

Macro-F1 delta by fold:

- fold 1: +1.04 pp;
- fold 2: -1.16 pp;
- fold 3: +0.09 pp;
- fold 4: -1.66 pp;
- fold 5: -0.10 pp.

Macro-F1 improved in **2/5 folds**.

Positive-delta folds:

- Accuracy: 2/5;
- Balanced Accuracy: 1/5;
- Macro-F1: 2/5;
- Hard-F1: 3/5;
- Worst-F1: 2/5.

## Paired prediction outcomes

Across 485 paired validation observations:

- WRC rescue: 8;
- WRC damage: 8;
- both correct: 435;
- both wrong: 34;
- net top-1 effect: **0**.

## Per-class behavior

Largest mean F1 gains:

- Immature: +4.36 pp;
- Broken: +1.54 pp;
- Fungus Damage: +1.33 pp;
- Partial Sour: +1.18 pp;
- Full Sour: +1.17 pp.

Largest mean F1 losses:

- Fade: -6.48 pp;
- Withered: -2.97 pp;
- Parchment: -1.82 pp;
- Cut: -1.54 pp;
- Partial Black: -1.54 pp.

The original sour-boundary hypothesis was partially supported locally:
Partial Sour and Full Sour improved on mean. The cost moved to other classes,
especially Fade and Withered.

## Hard-group behavior

Mean WRC-HBP minus WR-HBP:

- sour/black: **+0.27 pp**;
- shape/withered: **-0.05 pp**;
- insect damage: **-0.66 pp**.

Thus the CLAHE contrast residual slightly helped the targeted sour/black group,
but the overall Macro/Hard/Worst trade-off remained negative.

## Probability/margin diagnostic

Across the 485 paired observations:

- mean true-class probability delta: **+4.14 pp**;
- median true-class probability delta: **-0.19 pp**;
- true-class probability increased on 232/485 observations;
- mean true-class-vs-best-rival margin delta: **+4.43 pp**;
- median margin delta: **-0.24 pp**;
- margin increased on 233/485 observations.

Large positive confidence shifts on a minority of samples dominate the mean,
while the median sample shifts slightly downward. This explains why stronger
average confidence does not translate into better Macro-F1.

## Learned gates

Contrast-gate `tanh(alpha_c)` at selected checkpoints:

- fold 1: -0.1345;
- fold 2: -0.1383;
- fold 3: -0.1421;
- fold 4: -0.1473;
- fold 5: -0.1327.

Mean: **-0.1390 ± 0.0059**.

The contrast route is consistently used by all folds; the failure is therefore
not caused by the new gate remaining near zero.

## Efficiency

Parameter count:

- WR-HBP: 3,563,202;
- WRC-HBP: 3,563,531;
- overhead: **329 parameters = 0.0092%**.

## Frozen screening gate

Required:

1. mean Macro-F1 delta > 0;
2. Macro-F1 positive in at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0.

Observed:

- Macro mean positive: **FAIL**;
- Macro positive >=3/5 folds: **FAIL**;
- Hard mean nonnegative: **FAIL**;
- Worst mean nonnegative: **FAIL**.

Decision:

> **FAIL / STOP WRC-HBP.**

Do not tune CLAHE clip limit, grid size, contrast branch width, contrast
representation, injection stage, or gate on these reused folds.

## Reproducibility warning discovered by the matched control

The retrained WR-HBP control in this WRC experiment did not numerically
reproduce the earlier WR-HBP development package despite using the same
Coffee17 folds, seed 42, WR-HBP source files, and WR-HBP config.

Earlier WR-HBP package:

- Macro-F1 mean: **91.77%**.

Retrained WR-HBP control here:

- Macro-F1 mean: **91.44%**.

Mean difference: **-0.33 pp**.

Per-fold retraining difference relative to the earlier WR package:

- fold 1: -0.99 pp;
- fold 2: -2.15 pp;
- fold 3: -0.01 pp;
- fold 4: +1.97 pp;
- fold 5: -0.46 pp.

The relevant WR model, WR training engine, data loader, and WR config blobs are
identical between the two scientific commits. The current
`seed_everything()` implementation seeds Python/NumPy/Torch RNGs but does not
enforce deterministic CUDA algorithms.

Therefore the earlier WR-HBP development PASS should be interpreted as
**directional single-seed development evidence with measurable run-to-run
optimization nondeterminism**, not as bitwise-reproducible evidence.

This reproducibility finding must be resolved before a final superiority claim.
