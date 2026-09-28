# Preprocessing–HBP Decision Record

Last updated: **2026-09-28**

This document records the completed Coffee17 findings that motivated the
current matched `R0-HBP vs W0-HBP` experiment. It is intended to prevent
duplicate experiments, renamed reruns, and accidental reuse of rejected method
families.

## 1. Scope and protocol boundary

The original preprocessing OOF has already been opened. All later
fusion/distillation/transfer experiments that reuse the Coffee17 development
folds are **post-primary exploratory method development**.

Do not present them as independent confirmation.

The current preprocessing-study development protocol uses:

- Coffee17 clean identities;
- 5 folds;
- seed 42;
- image size 224;
- MobileNetV3-Large;
- AdamW, lr 3e-4, weight decay 1e-4;
- label smoothing 0.1;
- cosine scheduler;
- 50 epochs;
- deterministic train rotations 0/45/90/135/180/225/270.

The direct W0-HBP experiment defined later keeps this protocol and changes only
the model head from GAP to HBP for both matched arms.

---

## 2. Frozen preprocessing definitions

### R0

Raw RGB.

### C0

RGB -> Lab -> CLAHE on L -> RGB.

Frozen parameters:

- clip limit 2.0;
- tile grid 8x8.

### F0

Coffee17 adaptation of angular/frequency preprocessing:

- Rec.709 luminance;
- patch size 32;
- overlap 0.5;
- 360 angular bins;
- gamma 0.1;
- FFT magnitude used to estimate angular-frequency density;
- low-density directions suppressed;
- phase preserved;
- inverse FFT;
- overlap averaging;
- luminance-derived gate shared over RGB;
- residual form `x + x * G_Y`.

F0 is **not** a full reproduction of LFDet AFAB+CGFI.

### W0

Channel-wise RGB Haar wavelet reconstruction:

- 4 DWT levels;
- noise estimate from the finest detail coefficients using
  `median(abs(detail))/0.6745`;
- universal VisuShrink threshold;
- soft threshold on detail coefficients;
- inverse DWT reconstruction;
- epsilon 1e-8.

No W0 threshold, wavelet-level, or band search is authorized after the current
matched protocol is frozen.

---

## 3. Primary preprocessing OOF result

Population: 965 clean identities.

| Arm | Macro-F1 | Balanced Acc | Accuracy | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0 | 86.79% | 86.85% | 86.94% | 80.48% | 66.67% |
| C0 | 86.10% | 85.97% | 86.42% | 79.43% | 61.02% |
| F0 | 86.11% | 86.13% | 86.22% | 78.66% | 66.09% |
| W0 | 87.14% | 87.15% | 87.05% | 80.32% | 69.03% |

Macro-F1 delta vs R0:

- C0: -0.69 pp;
- F0: -0.68 pp;
- W0: +0.35 pp.

Paired bootstrap 95% CI:

- C0: [-2.62, +1.20] pp;
- F0: [-2.70, +1.27] pp;
- W0: [-1.68, +2.38] pp.

Important correction:

> W0 did **not** improve aggregate Hard-F1 over R0. W0 Hard-F1 was 80.32%
> versus R0 80.48%.

Primary interpretation:

> Fixed preprocessing alone did not establish a convincing aggregate
> improvement over raw RGB. Its effects are class-dependent.

---

## 4. Complementarity findings

Out of 965 OOF identities:

- R0 correct: 839;
- C0 correct: 834;
- F0 correct: 832;
- W0 correct: 840;
- all four correct: 739;
- all four wrong: 57;
- at least one arm correct: 908/965 = 94.09%.

R0 produced 126 errors.

Among those R0 errors:

- C0 rescued 40 and harmed 45;
- F0 rescued 43 and harmed 50;
- W0 rescued 46 and harmed 45;
- at least one transformed arm rescued 69/126 = 54.8%.

Unique rescues among the transformed arms:

- C0: 4;
- F0: 9;
- W0: 13.

Uniform late fusion confirms exploitable complementarity:

- R0+W0 Macro-F1: ~88.19%;
- F0+W0 Macro-F1: ~88.23%;
- ALL4 Macro-F1: ~88.17%;
- oracle over the four arms: ~94.09% accuracy-level coverage.

The transformed representations therefore contain complementary information
even though no fixed arm is a reliable standalone replacement for R0.

The late-fusion path is not the desired final deployment because it requires
multiple preprocessing/model paths.

---

## 5. W0 cue audit

A post-hoc cue audit was performed on the opened Coffee17 OOF to understand
what W0 changes were associated with changes in true-class probability.

The working table contained 965 samples and 490 descriptors.

W0 descriptors included:

- 4 levels;
- LH/HL/HH subbands;
- RGB channels;
- pre-threshold energy;
- post-threshold energy;
- retained-energy ratio;
- VisuShrink threshold values.

Statistical model for each W0 cue:

`delta_true_probability ~ standardized_cue + class + fold + R0_true_probability`

with HC3 robust covariance and BH FDR over 111 W0 features.

### Result

**15/111 W0 cues passed q < 0.05.**

The coherent signal was **relative detail survivability after VisuShrink**, not
raw high-frequency energy.

Strong cue family:

- L1-HH retained-energy ratio, positive association;
- several LH/HL retained ratios, positive association;
- RGB VisuShrink thresholds, negative association.

Examples:

- threshold G beta about -0.0444, q about 0.00465;
- threshold R beta about -0.0421, q about 0.00465;
- threshold B beta about -0.0404, q about 0.00465;
- L1-HH retained ratio R beta about +0.0449, q about 0.00647;
- L1-HH retained ratio G beta about +0.0448, q about 0.00647;
- L1-HH retained ratio B beta about +0.0447, q about 0.00647.

The three RGB L1-HH retained ratios were highly redundant
(correlations approximately 0.99+).

Mean-RGB L1-HH retained ratio showed strong one-way class association:

- eta-squared approximately 0.628.

Pre-energy and post-energy alone did not emerge as the stable cue family.

F0 angular descriptors did not produce a stable FDR-significant family
explaining the F0 improvement relative to R0.

Interpretation boundary:

> The cue audit is exploratory association on already-opened OOF data. It does
> not establish that L1-HH retained energy is causal.

---

## 6. Completed preprocessing-to-raw transfer family

The following experiments attempted to transfer information from transformed
views into one raw-RGB deployment model.

### 6.1 MVFD-SBN on GAP

Raw-only deployment with C0/F0/W0 training-only auxiliary views and detached
equal-mean GAP feature teacher.

Mean 5-fold result:

| Model | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| matched R0 GAP | 90.65% | 87.01% | 64.55% |
| MVFD-SBN | 91.77% | 85.26% | 67.88% |
| delta | +1.115 pp | -1.750 pp | +3.333 pp |

Macro-F1 improved in 4/5 folds.

This is the strongest completed raw-only GAP transfer result, with an explicit
trade-off: Macro/Worst improved while Hard-F1 declined.

### 6.2 AUXCE ablation

Removing explicit feature transfer and retaining auxiliary CE produced roughly
89.94% Macro-F1 versus 91.77% for MVFD-SBN.

This supports a contribution from the explicit final GAP feature-transfer term
within that frozen experiment.

### 6.3 Confidence-aware MVFD

Confidence weighting produced approximately 91.38% Macro-F1 versus 91.77% for
equal-mean MVFD.

Decision:

> Do not continue confidence-weighted teacher aggregation on this development
> path.

### 6.4 ML-MVFD intermediate tensor transfer

Direct intermediate spatial tensor MSE failed strongly.

Mean result:

- Macro-F1: 86.78%;
- Hard-F1: 79.55%;
- Worst-F1: 57.85%.

Relative to GAP MVFD:

- Macro-F1 about -4.99 pp;
- Hard-F1 about -5.71 pp;
- Worst-F1 about -10.03 pp;
- positive improvement: 0/5 folds.

Decision:

> Direct intermediate tensor matching is closed.

### 6.5 HBP + MVFD

MVFD was also applied to normalized HBP embeddings.

Mean result:

| Model | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| HBP R0 control | ~91.58% | ~86.23% | ~65.26% |
| HBP + MVFD | ~90.61% | ~84.38% | ~67.00% |
| delta | -0.97 pp | -1.86 pp | +1.74 pp |

Macro-F1 improved in only 1/5 folds and Hard-F1 in only 1/5 folds.

Interpretation:

> The GAP MVFD formulation does not transfer cleanly to normalized HBP
> embeddings.

Do not stack HBP with further variants of the same MVFD formulation.

---

## 7. W0-RAS: scalar cue supervision

W0-RAS encoded the exploratory W0 cue as one scalar auxiliary target:

`r_c = E_post,c / (E_pre,c + eps)`

`r(x) = mean(r_R, r_G, r_B)`

A training-only linear regression head predicted standardized mean-RGB L1-HH
retained-energy ratio from the raw GAP embedding.

Loss:

`L = CE + 0.05 * MSE(r_hat, r)`

Deployment remained raw RGB -> MobileNetV3-Large -> GAP -> Linear17.

### Result

| Metric | R0 control | W0-RAS | Delta |
|---|---:|---:|---:|
| Accuracy | 90.93% | 90.31% | -0.62 pp |
| Balanced Acc | 90.56% | 90.45% | -0.11 pp |
| Macro-F1 | 90.65% | 90.33% | -0.33 pp |
| Hard-F1 | 87.01% | 84.88% | -2.12 pp |
| Worst-F1 | 64.55% | 65.70% | +1.15 pp |

Per-fold Macro delta:

- -3.28 pp;
- +1.22 pp;
- -1.09 pp;
- +0.95 pp;
- +0.58 pp.

Hard-F1 mean declined by ~2.12 pp.

The auxiliary regression loss fell by roughly 90%, so the cue was learnable.
Its weighted contribution was modest (~1.15% of CE at epoch 1).

Decision:

> **FAIL / STOP W0-RAS V1.**

Do not tune lambda, swap bands, or derive more scalar variants from this same
opened cue audit.

Scientific interpretation:

> Association of a W0 cue with prediction changes did not imply that forcing the
> raw embedding to encode that cue would improve fine-grained classification.

---

## 8. W0-AT: normalized spatial attention transfer

W0-AT tested an intermediate representation between one scalar target and full
tensor matching.

For feature tensor F:

`A(F) = mean_c(F_c^2)`

`q(F) = vec(A) / ||vec(A)||_2`

`L_AT = mean((q_R - stopgrad(q_W))^2)`

Total loss:

`L = CE_R + 1000 * L_AT`

The W0 forward was training-only; inference remained raw RGB only.

### Result

| Metric | R0 control | W0-AT | Delta |
|---|---:|---:|---:|
| Accuracy | 90.93% | 89.28% | -1.65 pp |
| Balanced Acc | 90.56% | 89.41% | -1.15 pp |
| Macro-F1 | 90.65% | 89.21% | -1.44 pp |
| Hard-F1 | 87.01% | 83.23% | -3.78 pp |
| Worst-F1 | 64.55% | 61.36% | -3.18 pp |

Per-fold Macro delta:

- -1.09 pp;
- -6.21 pp;
- +0.36 pp;
- -0.03 pp;
- -0.23 pp.

Hard-F1 declined in **5/5 folds**.

The weighted AT/CE ratio at epoch 1 was only ~6.20%, so the failure was not
caused by an exploding auxiliary term.

Mean raw/W0 spatial-attention cosine:

- epoch 1: approximately 0.984;
- selected best checkpoints: approximately 0.9976.

The objective successfully made the two spatial maps almost identical, yet
classification degraded.

Decision:

> **FAIL / STOP W0-AT V1.**

Do not search beta, stage, exponent, or stack W0-AT with HBP.

---

## 9. Transfer-family conclusion

Across representation levels:

- scalar W0 cue -> R0: failed;
- normalized spatial attention map -> R0: failed;
- full intermediate tensor -> R0: failed strongly;
- final GAP embedding MVFD -> improved Macro/Worst but reduced Hard;
- final HBP embedding MVFD -> reduced Macro/Hard.

Therefore:

> **Close the preprocessing -> raw-only representation-transfer family on the
> reused Coffee17 development folds.**

This decision does **not** remove preprocessing from the thesis.

The direct role of preprocessing remains scientifically relevant because:

1. primary R0/C0/F0/W0 experiments established class-dependent effects;
2. transformed views rescued 69/126 R0 OOF errors;
3. W0 produced the strongest standalone transformed-arm point estimate and
   Worst-F1 point estimate;
4. W0 cue analysis identified a coherent wavelet-detail-survivability signal;
5. transfer experiments showed that compressing transformed information into a
   raw-only representation is not automatically beneficial.

The next defensible question is therefore whether the preprocessing itself can
be useful when paired with the strongest completed fine-grained representation
head, HBP.

---

## 10. HBP evidence relevant to the next experiment

Independent completed HBP evidence showed that HBP can improve Coffee17
fine-grained classification relative to GAP, with the strongest mean gain on
weak/worst classes, while also showing higher seed variance.

The separate final three-seed HBP protocol reported:

- GAP Macro-F1: 85.46 +/- 1.31%;
- HBP Macro-F1: 86.78 +/- 3.20%;
- GAP Hard-F1: 81.52 +/- 2.13%;
- HBP Hard-F1: 84.01 +/- 4.58%;
- GAP Worst-F1: 55.64 +/- 2.60%;
- HBP Worst-F1: 63.27 +/- 3.33%.

These numbers come from a **different protocol** and must not be numerically
mixed with the 5-fold preprocessing-study results.

Cross-dataset USK-Coffee showed that HBP is not universal; MobileNetV3-GAP
outperformed MobileNetV3-HBP there.

---

## 11. Completed matched R0-HBP vs W0-HBP

Canonical branch:

`codex/w0-hbp-matched-v1`

Canonical protocol:

`docs/protocols/W0_HBP_MATCHED_V1.md`

Canonical notebook:

`notebooks/Coffee17_W0_HBP_MATCHED_Kaggle.ipynb`

Completed result:

`docs/results/W0_HBP_MATCHED_RESULTS.md`

Machine-readable summary:

`docs/results/w0_hbp_matched_summary.json`

Source package SHA-256:

`f12670b3691dd2e482962d3edaa6fee1bfafdef9b600ceaccf87592952640bf4`

### Integrity

Five matched folds completed, seed 42.

For every fold:

- R0-HBP and W0-HBP used the same clean validation rows;
- both arms had the same initial HBP model-state fingerprint;
- R0-HBP was retrained as the matched control;
- W0-HBP was trained as the candidate;
- all non-preprocessing training variables were matched;
- outer test was not accessed.

Common initial model-state SHA-256:

`6d425c6c149005afde1224ed74ccfe4eb84c95e574d21e25a0e728578d79f9cc`

### Aggregate result

| Metric | R0-HBP | W0-HBP | W0 - R0 |
|---|---:|---:|---:|
| Accuracy | 91.55% | 90.10% | -1.44 pp |
| Balanced Accuracy | 91.63% | 90.13% | -1.50 pp |
| Macro-F1 | 91.27% | 90.04% | -1.23 pp |
| Hard-F1 | 86.23% | 84.09% | -2.13 pp |
| Worst-F1 | 64.95% | 63.78% | -1.17 pp |

Macro-F1 delta by fold:

- fold 1: +0.14 pp;
- fold 2: -1.00 pp;
- fold 3: -2.23 pp;
- fold 4: -1.33 pp;
- fold 5: -1.75 pp.

Macro-F1 improved in **1/5 folds**.

Across 485 unique paired validation observations:

- W0 rescue: 5;
- W0 damage: 12;
- both correct: 432;
- both wrong: 36.

Net top-1 effect: **-7 correct predictions** for W0-HBP.

Mean hard-group delta:

- sour/black: -4.80 pp;
- shape/withered: +0.71 pp;
- insect damage: -2.41 pp.

Largest mean per-class gains:

- Fade: +4.81 pp;
- Immature: +2.72 pp;
- Withered: +2.55 pp.

Largest mean per-class losses:

- Partial Sour: -7.78 pp;
- Severe Insect Damage: -4.64 pp;
- Floater: -4.44 pp;
- Partial Black: -3.36 pp;
- Full Sour: -3.27 pp;
- Cut: -3.13 pp.

### Frozen gate

All four gate conditions failed:

1. mean paired Macro-F1 delta > 0: FAIL;
2. Macro-F1 positive in >=3/5 folds: FAIL;
3. mean paired Hard-F1 delta >= 0: FAIL;
4. mean paired Worst-F1 delta >= 0: FAIL.

Decision:

> **FAIL / STOP W0-HBP.**

Direct W0 preprocessing does not improve HBP under the frozen matched
Coffee17 preprocessing-study protocol.

Do not tune wavelet level, VisuShrink threshold, HBP stages, image resolution,
or loss on these reused development folds.

This closes the direct W0+HBP path while preserving the earlier preprocessing
complementarity and cue-analysis findings as separate evidence.

---

## 12. Deprecated W0-HBP notebook paths

The earlier W0-HBP notebook attempts on branch `codex/w0-hbp-v1` are
deprecated:

- `Coffee17_W0_HBP_Kaggle.ipynb`;
- `Coffee17_W0_HBP_V2_Kaggle.ipynb`;
- `Coffee17_W0_HBP_V3_Kaggle.ipynb`.

Reason:

- V1/V2 incorrectly depended on an old
  `hbp-mvfd-sbn-analysis-package.zip` reference being mounted in Kaggle;
- V3 avoided that dependency by using only rounded historical aggregate HBP
  numbers, which would not provide a clean matched per-fold comparison.

Do not use those notebooks.

Use only the matched-control notebook on
`codex/w0-hbp-matched-v1`.

---

## 13. Anti-duplication rules after this record

Do not propose or rerun the following as new methods:

- W0-RAS or scalar retained-energy variants;
- W0-AT or stage/beta/exponent variants;
- direct intermediate tensor MSE / ML-MVFD;
- confidence-aware MVFD;
- HBP + the same MVFD formulation;
- HBP + W0-AT;
- generic output KD of preprocessing teachers;
- naive shared multi-view CE without selective normalization.

The matched direct preprocessing test `R0-HBP vs W0-HBP` is now completed
and failed its frozen gate.

Do not rerun or tune direct W0-HBP on these reused development folds.

Any later method proposal must first be checked against the experiment master
record and this decision record.
