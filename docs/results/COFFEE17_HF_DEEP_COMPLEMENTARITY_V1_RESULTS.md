# Coffee17 HF–Deep Complementarity V1 — Results

Date recorded: 2026-09-30

Source analysis package SHA-256:
`8b1ea9b739519f19d779d061380cfb1027c27cea0b368de4e287b3733b2b6ff5`

Source summary SHA-256:
`0362b5eaf3727013ee15a49cbb2fc6e22384c3f83841eb2d33f7cb29c5fc87d1`

Scientific code used by all five folds:
`ac6eceee8319a6fd970d9f1de955bf300d3be115`

Protocol:
- Coffee17 original identities only;
- strict five-fold development protocol, seed 42;
- no train/validation identity overlap in any fold;
- outer test untouched;
- Tulsi-aligned 71D handcrafted descriptor extraction;
- MRMR-20 fitted separately on each training fold only;
- HBP embedding and handcrafted features evaluated with the same fixed
  multinomial logistic-regression probe;
- fusion used equal-L2 block weighting;
- no validation tuning;
- all five folds passed the registered HBP strict-CUDA smoke test.

## Aggregate result

| Metric | HF20 | HBP-EMB | HF20 + HBP-EMB | Fusion - HBP |
|---|---:|---:|---:|---:|
| Accuracy | 61.65% | 91.55% | 91.55% | +0.00 pp |
| Balanced Accuracy | 61.24% | 91.77% | 91.69% | -0.08 pp |
| Macro-F1 | 58.28% | 91.53% | 91.47% | -0.06 pp |
| Hard-F1 | 46.27% | 87.48% | 87.04% | -0.44 pp |
| Worst-F1 | 0.00% | 65.64% | 63.28% | -2.36 pp |

Fusion Macro-F1 delta by fold:

1. +0.09 pp
2. -0.94 pp
3. -1.12 pp
4. -1.14 pp
5. +2.81 pp

Macro-F1 improved in 2/5 folds.
Hard-F1 improved in 1/5 folds.
Worst-F1 improved in 0/5 folds.

## Hard-pair errors and paired outcomes

Total preregistered hard-pair confusions:

- HBP-EMB: **25**
- HF20 + HBP-EMB: **25**
- delta: **0**

Across 485 validation observations:

- rescue: **3**
- damage: **3**
- both correct: **441**
- both wrong: **38**
- net correct: **0**

Only seven validation predictions changed between HBP-EMB and fusion. Three
were rescues, three were damages, and one changed from one wrong class to
another wrong class. The positive fold-5 effect therefore did not reproduce
across the other folds.

## Handcrafted-only result

HF20 was substantially weaker than the learned HBP representation:

- Macro-F1: **58.28%**
- Hard-F1: **46.27%**
- Worst-F1: **0.00%**

The selected MRMR features were nevertheless highly stable across folds.
Thirteen features appeared in all 5/5 folds, including mean RGB/CIELAB color
features, circularity/eccentricity, first-level DWT detail standard deviations,
Gabor descriptors, and LBP energy. Stable selection therefore did not translate
into complementary predictive value under the frozen fusion protocol.

## Frozen decision gate

All five registered criteria failed:

- mean Macro-F1 delta > 0: **FAIL**
- Macro-F1 positive in >=4/5 folds: **FAIL** (2/5)
- mean Hard-F1 delta > 0: **FAIL**
- targeted hard-pair confusions reduced: **FAIL** (25 -> 25)
- mean Worst-F1 delta >= 0: **FAIL**

Frozen decision:

`NO_CLEAR_HF_COMPLEMENTARITY`

## Decision

> **STOP HF–DEEP COMPLEMENTARITY V1 on the reused development folds.**
> Do not rescue the same hypothesis by trying RealMLP, FT-Transformer,
> different MRMR K, different logistic C, feature-family pruning, learned
> fusion, or alternate weighting on these folds.

Interpretation:

> The tested explicit color/texture/shape descriptor set contains class-related
> information, but it does not provide reproducible complementary information
> beyond the learned HBP embedding under the registered fixed-probe fusion
> protocol. This closes the specific Tulsi-style HF20 + HBP-embedding fusion
> route; it does not imply that handcrafted descriptors are universally
> useless for coffee analysis.

Outer test remains untouched.
