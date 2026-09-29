# Coffee17 Experiment Master Results

Last updated: 2026-09-29

This document is the canonical running record for the Coffee17 preprocessing,
fusion, distillation, shared-multiview, selective-BN, representation, and
fine-grained local-learning experiments conducted so far.

**Before proposing any new experiment, read**
`docs/experiments/COFFEE17_EXPERIMENT_LEDGER.md`.
The ledger contains explicit STOP/CLOSED/BLOCKED entries to prevent old
hypotheses from being proposed again under a new name.

Important scope note: the original preprocessing OOF was opened before the
later fusion/KD/MVCE/SBN methods were designed. Therefore all later methods are
post-primary exploratory method development. Do not treat repeated reuse of the
opened OOF as confirmatory evidence.

---

## 1. Common experimental context

Dataset: Coffee17, 17 fine-grained coffee-bean classes.

Backbone in this experimental chain:
- MobileNetV3-Large
- GAP head
- input 224x224
- seed 42
- AdamW, lr 3e-4
- weight decay 1e-4
- label smoothing 0.1
- cosine scheduler
- 50 epochs
- deterministic rotations: 0, 45, 90, 135, 180, 225, 270 degrees

Input representations:
- R0: raw RGB
- C0: Lab-L CLAHE
- F0: luminance + angular/frequency preprocessing
- W0: Haar wavelet + VisuShrink reconstruction

---

## 2. Primary preprocessing OOF study

OOF population: 965 images.

| Arm | Macro-F1 | Balanced Acc | Accuracy | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0 | 86.7899% | 86.85% | 86.94% | 80.48% | 66.67% |
| C0 | 86.10% | 85.97% | 86.42% | 79.43% | 61.02% |
| F0 | 86.11% | 86.13% | 86.22% | 78.66% | 66.09% |
| W0 | 87.14% | 87.15% | 87.05% | 80.32% | 69.03% |

Macro-F1 delta vs R0:
- C0: -0.6948 pp
- F0: -0.6790 pp
- W0: +0.3458 pp

Paired bootstrap 95% CI for Macro-F1 delta:
- C0: [-2.62, +1.20] pp
- F0: [-2.70, +1.27] pp
- W0: [-1.68, +2.38] pp

Primary conclusion:

> No fixed preprocessing arm produced a convincing aggregate improvement over
> raw RGB. Effects are heterogeneous and class-dependent.

### Per-class F1 deltas vs R0

C0 positives:
- Dry Cherry +4.854
- Partial Sour +2.809
- Immature +2.334
- Full Sour +1.889
- Husk +1.781
- Broken +1.750
- Fungus +0.469
- Parchment 0

C0 negatives:
- Fade -6.017
- Slight Insect -5.650
- Withered -4.717
- Severe Insect -3.841
- Floater -3.290

F0 positives:
- Dry Cherry +3.920
- Immature +3.092
- Husk +2.751
- Fade +1.888
- Parchment +0.867
- Shell +0.824
- Partial Black +0.697
- Cut +0.264
- Full Black 0

F0 negatives:
- Withered -7.132
- Severe Insect -4.532
- Floater -3.978
- Partial Sour -3.574
- Full Sour -2.778

W0 positives:
- Fade +8.358
- Slight Insect +2.360
- Floater +2.062
- Partial Sour +1.482
- Full Sour +1.471
- Fungus +1.370
- Parchment +0.782
- Immature +0.699
- Full Black 0

W0 negatives:
- Withered -3.495
- Partial Black -2.424
- Dry Cherry -1.812
- Husk -1.751
- Broken -1.556
- Severe Insect -0.708
- Cut -0.625
- Shell -0.335

---

## 3. Complementarity analysis

Correct predictions out of 965:
- R0: 839
- C0: 834
- F0: 832
- W0: 840

All four correct: 739.
All four wrong: 57.
At least one arm correct: 908 / 965 = 94.09%.

Oracle balanced recall: approximately 94.28%.

R0 errors: 126.
R0 errors rescued by at least one C0/F0/W0 arm: 69.

Rescue fraction:

69 / 126 = 54.8%

Unique-correct samples:
- R0: 11
- C0: 4
- F0: 9
- W0: 13

Pairwise rescue/damage relative to R0:
- C0: rescue 40, damage 45
- F0: rescue 43, damage 50
- W0: rescue 46, damage 45

Interpretation:

> The transformed representations contain materially complementary information
> even though none is a consistent standalone replacement for raw RGB.

---

## 4. Uniform late probability fusion

| Fusion | Macro-F1 | Balanced Acc | Accuracy | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0+C0 | 87.424% | 87.407% | 87.772% | 80.653% | 63.636% |
| R0+F0 | 87.174% | 87.168% | 87.358% | 80.400% | 65.455% |
| R0+W0 | 88.188% | 88.141% | 88.187% | 81.292% | 67.857% |
| F0+W0 | 88.234% | 88.073% | 88.290% | 81.516% | 68.376% |
| R0+C0+W0 | 88.535% | 88.483% | 88.705% | 81.628% | 63.717% |
| C0+F0+W0 | 88.272% | -- | -- | -- | -- |
| R0+F0+W0 | 88.110% | -- | -- | -- | -- |
| R0+C0+F0 | 87.823% | -- | -- | -- | -- |
| ALL4 | 88.172% | 88.168% | 88.394% | 81.899% | 66.055% |

ALL4 Macro-F1 improvement vs R0:

+1.382 pp

Exploratory paired bootstrap 95% CI:
- W0: +0.346 pp [-1.702, +2.407]
- R0+W0: +1.398 pp [-0.202, +2.989]
- F0+W0: +1.444 pp [-0.451, +3.351]
- R0+C0+W0: +1.745 pp [+0.245, +3.283]
- ALL4: +1.382 pp [-0.249, +3.051]

Do not promote R0+C0+W0 as a selected final method: subset choice was post-hoc.

Interpretation:

> Late fusion demonstrates exploitable complementarity, but requires multiple
> model/preprocessing paths and conflicts with the lightweight deployment goal.

---

## 5. Adaptive weighting / routing attempts

Global validation-accuracy weighting:
- weights were nearly uniform
- produced the same argmax as uniform ALL4

Global per-fold NLL-optimized weights:
- Macro-F1 87.865%
- Accuracy 88.083%
- Balanced Acc 87.741%
- Worst-F1 66.667%
- Hard-F1 81.787%
- exploratory Macro-F1 delta vs R0 +1.075 pp
- exploratory 95% CI [-0.425, +2.635]

Class-conditional 4x17 weights:
- Macro-F1 87.579%
- Accuracy 87.876%
- Balanced Acc 87.586%
- Worst-F1 68.468%
- Hard-F1 81.009%
- Macro-F1 delta vs R0 +0.790 pp
- exploratory 95% CI [-0.991, +2.537]

Observed instability:
- mean maximum per-class weight ~0.957-0.980
- roughly 15-16 / 17 classes assigned >0.9 weight to one view

Validation-selected equal-weight subset per fold:
- Macro-F1 86.294%
- Accuracy 86.425%
- Balanced Acc 86.373%
- Worst-F1 60.0%
- Hard-F1 79.090%

Conclusion:

> The approximately 97-image validation set per fold is too small for stable
> learned view routing or class-conditional weighting.

---

## 6. Efficiency reference

One MobileNetV3-Large:
- 2,988,289 parameters
- 11.3994 MB FP32

T4 batch-1 end-to-end latency:
- R0: 5.8439 ms/image
- C0: 8.8042 ms/image
- F0: 8.1036 ms/image
- W0: 10.1423 ms/image

Rough sequential four-model total:
- 32.894 ms/image
- 11.953M parameters
- ~45.6 MB FP32

The multi-model late-fusion solution is therefore not the desired deployment
architecture.

---

## 7. Knowledge Distillation experiment

Protocol:
- student input: raw RGB
- T = 2
- hard CE weight = 0.5
- soft KL weight = 0.5
- label smoothing = 0.1
- KD-R0 control
- KD-ALL4 treatment
- 5 folds each, 10/10 runs complete
- validation only for this exploratory phase

Mean validation metrics:

| Model | Accuracy | Balanced Acc | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| Historical R0 | 89.69% | 89.71% | 89.65% | 83.18% | 64.79% |
| KD-R0 | 89.69% | 89.40% | 89.35% | 84.74% | 62.96% |
| KD-ALL4 | 89.90% | 89.67% | 89.54% | 84.54% | 58.95% |

Macro-F1:
- KD-ALL4 vs KD-R0: +0.19 pp
- KD-ALL4 vs historical R0: -0.11 pp

Per-fold Macro-F1:

| Fold | Historical R0 | KD-R0 | KD-ALL4 |
|---|---:|---:|---:|
| 1 | 89.37% | 89.28% | 90.24% |
| 2 | 95.02% | 93.72% | 91.91% |
| 3 | 90.44% | 89.03% | 90.04% |
| 4 | 84.44% | 85.92% | 85.60% |
| 5 | 88.99% | 88.82% | 89.92% |

Teacher audit:
- train any-teacher disagreement ~1.10%
- train mean pairwise JS ~0.0098
- ALL4 train teacher accuracy 100%
- validation disagreement ~12.58%
- validation mean pairwise JS ~0.0374

Soft-KL during training dropped from roughly 2.3-2.45 initially to roughly
0.07-0.12.

KD-ALL4 vs KD-R0 across 485 validation-fold observations:
- rescue: 8
- damage: 7
- net: +1 correct observation

Interpretation:

> Vanilla output-level ensemble distillation can fit the soft target but does
> not meaningfully transfer the ensemble's useful diversity because the
> teachers are nearly identical on the training inputs where distillation is
> applied.

---

## 8. Shared-weight multi-view CE: MVCE_ALL4

Objective:

L_raw = CE(f(R0(x)), y)

L_aux = [CE(f(C0(x)), y) + CE(f(F0(x)), y) + CE(f(W0(x)), y)] / 3

L = 0.5 * L_raw + 0.5 * L_aux

One MobileNetV3-Large is shared across all four training views.
Inference remains raw RGB only.

A new matched R0_CONTROL was trained in the same runtime/environment because
historical training was not bit-identical on rerun.

Mean validation metrics:

| Model | Accuracy | Balanced Acc | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0_CONTROL | 90.93% | 90.56% | 90.65% | 87.01% | 64.55% |
| MVCE_ALL4 | 89.07% | 89.28% | 88.93% | 83.08% | 62.12% |

Delta MVCE_ALL4 vs matched R0_CONTROL:
- Macro-F1: -1.72 pp
- Accuracy: -1.86 pp
- Balanced Acc: -1.28 pp
- Hard-F1: -3.93 pp
- Worst-F1: approximately -2.42 pp

Per-fold Macro-F1:

| Fold | R0_CONTROL | MVCE_ALL4 | Delta |
|---|---:|---:|---:|
| 1 | 92.47% | 89.42% | -3.04 pp |
| 2 | 93.86% | 88.63% | -5.23 pp |
| 3 | 91.29% | 89.28% | -2.01 pp |
| 4 | 86.58% | 87.22% | +0.64 pp |
| 5 | 89.07% | 90.10% | +1.03 pp |

Validation prediction comparison:
- rescue: 7
- damage: 16
- net: -9

### MVCE optimization diagnosis

Training CE at the best validation epoch:

| Fold | R0 CE | C0 CE | F0 CE | W0 CE |
|---|---:|---:|---:|---:|
| 1 | 0.626 | 0.619 | 1.015 | 0.608 |
| 2 | 0.622 | 0.638 | 2.641 | 0.603 |
| 3 | 0.749 | 0.732 | 2.773 | 0.709 |
| 4 | 0.634 | 0.629 | 0.851 | 0.611 |
| 5 | 0.776 | 0.764 | 2.832 | 0.742 |

Random 17-class CE is approximately ln(17) = 2.83.

F0 therefore collapsed close to random loss on folds 2, 3, and 5.

The failed MVCE implementation allowed only R0 to update BatchNorm running
statistics and forced auxiliary C0/F0/W0 forwards to consume R0 running
statistics.

This motivated a targeted normalization experiment rather than view-weight
tuning or feature-fusion changes.

---

## 9. MVCE-SBN: Selective Batch Normalization

Only the normalization mechanism changed.

Loss, views, backbone, optimizer, training schedule, and deployment path were
kept fixed.

For R0:
- standard BN training behavior
- R0 updates persistent running mean/variance

For auxiliary views C0/F0/W0:
- use that auxiliary mini-batch's own current mean/variance
- do not persist those auxiliary statistics into running buffers
- BN affine gamma/beta remain shared and trainable

Inference:
- raw RGB only
- one MobileNetV3-Large
- R0 running statistics
- no extra inference branch
- no extra inference parameters

Mean validation metrics:

| Model | Accuracy | Balanced Acc | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0_CONTROL | 90.93% | 90.56% | 90.65% | 87.01% | 64.55% |
| failed MVCE_ALL4 | 89.07% | 89.28% | 88.93% | 83.08% | 62.12% |
| MVCE-SBN | 90.52% | 90.84% | 90.68% | 84.49% | 64.48% |

MVCE-SBN vs matched R0_CONTROL:
- Macro-F1: +0.023 pp
- Balanced Acc: approximately +0.28 pp
- Accuracy: approximately -0.41 pp
- Hard-F1: -2.52 pp
- Worst-F1: approximately -0.07 pp

Per-fold Macro-F1:

| Fold | R0_CONTROL | failed MVCE | MVCE-SBN | SBN - R0 |
|---|---:|---:|---:|---:|
| 1 | 92.47% | 89.42% | 89.50% | -2.97 pp |
| 2 | 93.86% | 88.63% | 93.20% | -0.66 pp |
| 3 | 91.29% | 89.28% | 92.27% | +0.98 pp |
| 4 | 86.58% | 87.22% | 88.40% | +1.82 pp |
| 5 | 89.07% | 90.10% | 90.01% | +0.94 pp |

Positive Macro-F1 folds: 3 / 5.

### Mechanistic result: F0 collapse removed

F0 CE at best epoch before SBN:

[1.015, 2.641, 2.773, 0.851, 2.832]

F0 CE at best epoch with SBN:

[0.609, 0.647, 0.609, 0.653, 0.627]

Mean:

2.022 -> 0.629

This is the strongest mechanistic result in this sequence.

Interpretation:

> Forcing auxiliary transformed views to consume R0 BatchNorm statistics caused
> a severe optimization failure, especially for F0. Selective normalization
> removed that collapse.

SBN also recovered the failed shared-MVCE Macro-F1:

88.93% -> 90.68%

Improvement relative to failed MVCE:

+1.74 pp Macro-F1

However, the recovered model only returned to approximately matched R0
performance and did not establish a classifier improvement.

### Per-class SBN delta vs matched R0

Largest gains:
- Fade +11.67 pp
- Parchment +4.04 pp
- Broken +2.49 pp
- Fungus Damage +2.03 pp
- Floater +1.82 pp
- Partial Sour +1.77 pp

Largest losses:
- Cut -6.84 pp
- Severe Insect Damage -5.94 pp
- Slight Insect Damage -4.26 pp
- Partial Black -3.61 pp
- Shell -1.54 pp

Validation prediction comparison vs matched R0:
- rescue: 8
- damage: 10
- net: -2

Therefore SBN redistributes class performance rather than producing a uniform
improvement.

---

## 10. Historical R0 vs matched R0

Historical mean validation R0 Macro-F1:
- approximately 89.65%

Matched same-environment R0_CONTROL Macro-F1:
- approximately 90.65%

Validation identities and initial model fingerprint matched.

Initial model-state SHA-256:

e288e9d23781017c7afd8e97fc5a522eaef84ada4751ca8034d133017a70b995

The training trajectory was not bit-identical across environments. Therefore
all MVCE/SBN comparisons must use the new matched R0_CONTROL, not the historical
89.65% R0 mean.

---

## 11. Current scientific interpretation

The evidence chain currently supports the following:

1. Fixed preprocessing alone is not a consistent aggregate improvement.
2. C0/F0/W0 contain class-dependent complementary information.
3. Late fusion can exploit that complementarity, but it is too expensive for
   the intended lightweight deployment.
4. Tiny validation folds are not reliable enough for learned routing or
   class-wise fusion weights.
5. Vanilla multi-teacher KD does not successfully transfer the useful
   complementarity into a single raw-RGB student.
6. Naive fully shared multi-view CE fails because heterogeneous preprocessing
   views cannot safely share R0 BatchNorm statistics.
7. Selective BatchNorm removes the auxiliary-view optimization collapse and
   recovers the lost Macro-F1.
8. Even after optimization is repaired, raw-only inference does not yet inherit
   enough useful information from C0/F0/W0 to outperform the matched R0 control.

Current status after later follow-up experiments:

> The preprocessing -> raw-only representation-transfer family is closed on the
> reused Coffee17 development folds. Scalar W0-RAS, spatial W0-AT, direct
> intermediate tensor matching, and HBP+MVFD all failed their intended
> discrimination objective; GAP MVFD improved Macro/Worst while reducing Hard.

Preprocessing remains a substantive thesis variable. The current frozen
question is a direct matched comparison of R0-HBP versus W0-HBP, with both arms
retrained on the same 5 folds and seed 42 so that preprocessing is the only
experimental factor.

---

## 12. Protocol boundary

The original primary preprocessing OOF has already been opened.

Accordingly:
- primary preprocessing results remain the original primary study;
- fusion, KD, MVCE, and SBN are exploratory post-primary development;
- a future final method should be frozen before an independent final evaluation
  if a clean confirmatory thesis claim is required.


---

## 13. AT-SBN: Auxiliary Training + Selective BatchNorm

Artifact analyzed:
- `at-sbn-analysis-package.zip`
- SHA-256: `470fd28bfe125ac19a89a8f8d2d765edbcbfa2e4efd13ed72d34336f444f9b64`
- 5/5 folds complete
- 50 epochs per fold
- 97 validation images per fold
- scientific commit: `1330b9a9dcf125b5ff9b6fd598ed513ecf88d002`
- primary initialization SHA-256:
  `e288e9d23781017c7afd8e97fc5a522eaef84ada4751ca8034d133017a70b995`
- `test_images_accessed=false` on all folds

AT-SBN retained the validated SBN mechanism but decoupled the classifiers:
- R0 uses the primary deployment classifier;
- C0/F0/W0 use separate training-only linear classifiers;
- auxiliary hard-label CE weight: 0.05 per view;
- detached R0-to-auxiliary self-distillation weight: 0.05 per view;
- temperature: 1.0;
- late squared-L2 classifier merging begins at epoch 44;
- auxiliary classifiers are discarded at inference.

Mean validation metrics:

| Model | Accuracy | Balanced Acc | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0_CONTROL | 90.93% | 90.56% | 90.65% | 87.01% | 64.55% |
| MVCE-SBN | 90.52% | 90.84% | 90.68% | 84.49% | 64.48% |
| AT-SBN | 91.13% | 90.85% | 90.67% | 85.39% | 61.86% |

AT-SBN vs matched R0_CONTROL:
- Accuracy: +0.206 pp
- Balanced Accuracy: +0.286 pp
- Macro-F1: +0.020 pp
- Hard-F1: -1.617 pp
- Worst-F1: -2.684 pp

AT-SBN vs MVCE-SBN:
- Accuracy: +0.619 pp
- Balanced Accuracy: +0.003 pp
- Macro-F1: -0.002 pp
- Hard-F1: +0.899 pp
- Worst-F1: -2.623 pp

Per-fold Macro-F1:

| Fold | R0_CONTROL | MVCE-SBN | AT-SBN | AT-SBN - R0 |
|---|---:|---:|---:|---:|
| 1 | 92.47% | 89.50% | 91.24% | -1.23 pp |
| 2 | 93.86% | 93.20% | 93.78% | -0.08 pp |
| 3 | 91.29% | 92.27% | 91.27% | -0.01 pp |
| 4 | 86.58% | 88.40% | 86.16% | -0.42 pp |
| 5 | 89.07% | 90.01% | 90.91% | +1.84 pp |

Positive Macro-F1 folds vs R0_CONTROL: 1 / 5.

Validation-fold observation comparison vs matched R0:
- R0 correct: 441 / 485 fold-observations
- AT-SBN correct: 442 / 485 fold-observations
- rescue: 7
- damage: 6
- net: +1

These 485 entries are fold-observations and are not 485 independent images.

### AT-SBN optimization diagnostics

Best epoch per fold:

[30, 32, 31, 49, 28]

Only fold 4 selected a checkpoint after classifier merging became active at
epoch 44. Thus 4/5 selected models come from the pre-merge phase.

Mean best pre-merge Macro-F1:
- 90.572%

Mean best merge-active Macro-F1:
- 89.894%

Difference:
- -0.677 pp

Mean best merge-active Macro-F1 vs matched R0:
- -0.758 pp

At merge activation, the total training objective jumps sharply because the
un-normalized squared classifier-distance term is large. Example epoch-43 to
epoch-44 total loss:
- fold 1: 0.785 -> 31.444
- fold 2: 0.787 -> 31.300
- fold 3: 0.783 -> 31.204
- fold 4: 0.797 -> 31.320
- fold 5: 0.776 -> 31.092

The classifier distances do decrease after merging, but this does not translate
into a consistent R0 validation improvement.

### Auxiliary optimization and diversity

F0 CE at the R0-selected best epoch remains healthy:

[0.646, 0.643, 0.643, 0.627, 0.643]

Mean F0 CE:
- 0.640

Therefore SBN continues to prevent the prior F0 optimization collapse.

However, auxiliary predictions become very close to the R0 primary prediction
on training samples. At the selected best epoch:

Mean F0 top-1 disagreement:
- 1.17%

Mean F0 Jensen-Shannon divergence:
- 0.01286

The other auxiliary heads show similarly low disagreement/JS.

Mean transformed-view validation Macro-F1 at the R0-selected best checkpoint:
- C0 auxiliary head: 73.84%
- F0 auxiliary head: 76.60%
- W0 auxiliary head: 72.28%

Thus the auxiliary heads fit the training objective but do not form strong
standalone validation classifiers, while the R0-to-auxiliary distillation also
drives their outputs toward the primary prediction.

### Per-class AT-SBN delta vs matched R0

Largest gains:
- Broken +4.68 pp
- Fungus Damage +3.53 pp
- Parchment +2.22 pp
- Dry Cherry +2.22 pp
- Floater +1.82 pp
- Immature +1.52 pp
- Withered +1.29 pp

Largest losses:
- Slight Insect Damage -8.14 pp
- Cut -3.79 pp
- Severe Insect Damage -2.61 pp
- Fade -1.19 pp
- Full Sour -1.18 pp

Notably, the large Fade gain produced by MVCE-SBN (+11.67 pp vs R0) is not
retained by AT-SBN.

### AT-SBN conclusion

AT-SBN does not establish an aggregate improvement over the matched R0 control.

The key diagnostic is structural: the tested self-distillation direction is
R0 -> auxiliary. It makes the auxiliary predictions more similar to R0, but it
does not provide a direct auxiliary/multi-view -> R0 supervision path. Any
benefit to R0 can only arrive indirectly through shared feature-extractor
gradients and the late classifier-merging penalty.

Given the observed low auxiliary disagreement and neutral R0-only Macro-F1,
the next method should explicitly transfer a training-only multi-view
representation toward the R0 deployment representation, rather than further
forcing auxiliary heads to imitate R0.


---

## 14. MVFD-SBN: explicit transformed-view feature -> R0 transfer

Artifact analyzed:
- `mvfd-sbn-analysis-package.zip`
- SHA-256: `d2f432be885a9c507fdca0cbeb02d8d3dbef70d2080f306c8d84d84bb6ffecf6`
- 5/5 folds complete
- 50 epochs per fold
- 97 validation images per fold
- scientific commit: `fdae1fff10348f78ca67bb00948ae5768e9c98d9`
- primary initialization SHA-256:
  `e288e9d23781017c7afd8e97fc5a522eaef84ada4751ca8034d133017a70b995`
- `test_images_accessed=false` on all folds

MVFD-SBN removes AT-SBN's R0->auxiliary output distillation and late classifier
merging. It retains SBN and training-only C0/F0/W0 auxiliary classifiers, then
constructs a detached equal-mean teacher from transformed-view GAP embeddings:

`t = stopgrad((e_C + e_F + e_W) / 3)`

and explicitly pulls the raw-RGB R0 GAP embedding toward that teacher:

`L_feat = mean_i ||e_R_i - t_i||_2^2`

with total training loss:

`L = CE_R + 0.05(CE_C + CE_F + CE_W) + 0.007 L_feat`

The 0.007 feature coefficient is the fixed squared-L2 consistency coefficient
reported by Dong et al. for their multi-view consistency distillation setup; it
was not tuned on Coffee17 validation.

Mean validation metrics:

| Model | Accuracy | Balanced Acc | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|---:|
| R0_CONTROL | 90.93% | 90.56% | 90.65% | 87.01% | 64.55% |
| MVCE-SBN | 90.52% | 90.84% | 90.68% | 84.49% | 64.48% |
| AT-SBN | 91.13% | 90.85% | 90.67% | 85.39% | 61.86% |
| MVFD-SBN | 91.55% | 92.09% | 91.77% | 85.26% | 67.88% |

MVFD-SBN vs matched R0_CONTROL:
- Accuracy: +0.619 pp
- Balanced Accuracy: +1.527 pp
- Macro-F1: +1.115 pp
- Hard-F1: -1.750 pp
- Worst-F1: +3.333 pp

MVFD-SBN vs MVCE-SBN:
- Accuracy: +1.031 pp
- Balanced Accuracy: +1.244 pp
- Macro-F1: +1.092 pp
- Hard-F1: +0.766 pp
- Worst-F1: +3.394 pp

MVFD-SBN vs AT-SBN:
- Accuracy: +0.412 pp
- Balanced Accuracy: +1.241 pp
- Macro-F1: +1.094 pp
- Hard-F1: -0.133 pp
- Worst-F1: +6.017 pp

Per-fold Macro-F1:

| Fold | R0_CONTROL | MVFD-SBN | Delta |
|---|---:|---:|---:|
| 1 | 92.47% | 91.57% | -0.90 pp |
| 2 | 93.86% | 94.27% | +0.41 pp |
| 3 | 91.29% | 91.34% | +0.05 pp |
| 4 | 86.58% | 90.10% | +3.51 pp |
| 5 | 89.07% | 91.56% | +2.49 pp |

Positive Macro-F1 folds vs R0_CONTROL: 4 / 5.

Validation-fold observation comparison vs matched R0:
- R0 correct: 441 / 485 fold-observations
- MVFD-SBN correct: 444 / 485 fold-observations
- rescue: 12
- damage: 9
- net: +3

These 485 entries are fold-observations and are not 485 independent images.

### Feature-transfer diagnostics

Best epoch per fold:

[37, 26, 26, 21, 46]

At the selected best checkpoints:
- mean primary CE: 0.636
- mean F0 CE: 0.725
- mean raw feature-distillation loss: 6.236
- mean weighted feature contribution: 0.04365
- mean R0-teacher cosine similarity: 0.98997
- mean R0-teacher L2 distance: 2.430
- mean teacher feature norm: 17.084
- mean R0 feature norm: 17.402
- mean C0 disagreement: 1.32%
- mean F0 disagreement: 2.28%
- mean W0 disagreement: 1.44%

At epoch 1, mean R0-teacher cosine is only approximately 0.875 and feature
loss is approximately 12.9. Across training the explicit feature-consistency
objective therefore measurably brings R0 toward the transformed-view teacher.

F0 remains well optimized under SBN. The prior near-random F0 collapse does
not recur.

A second structural observation is that the equal-mean teacher is more similar
to R0 than each individual transformed view at the selected checkpoints. Mean
R0-teacher cosine is about 0.990, while R0-to-individual-view cosine values are
generally around 0.96-0.98. Equal averaging therefore emphasizes the common
cross-view component and can cancel some view-specific residual information.

### Hard-group behavior

Mean hard-group F1 relative to matched R0:

- sour/black group: -1.21 pp
- shape/withered group: -3.76 pp
- insect-damage group: +0.46 pp

The overall Hard-F1 decrease is therefore not uniform. It is driven mainly by
the shape/withered and sour/black groups, while insect-damage performance
slightly improves.

### Per-class MVFD-SBN delta vs matched R0

Largest gains:
- Fade +13.49 pp
- Parchment +7.69 pp
- Floater +4.00 pp
- Severe Insect Damage +2.13 pp
- Dry Cherry +2.04 pp
- Shell +1.64 pp
- Broken +1.58 pp

Largest losses:
- Cut -9.41 pp
- Partial Black -3.02 pp
- Immature -1.52 pp
- Full Sour -1.14 pp

Slight Insect Damage also improves slightly (+0.67 pp), so unlike the previous
SBN/AT-SBN variants the insect-damage group is no longer the main source of
Hard-F1 degradation.

### MVFD-SBN conclusion

MVFD-SBN is the first single-model, raw-RGB-inference method in this development
chain to show a non-trivial positive mean Macro-F1 shift over the matched R0
control while also improving balanced accuracy and Worst-F1:

- Macro-F1 +1.115 pp
- Balanced Accuracy +1.527 pp
- Worst-F1 +3.333 pp
- positive Macro-F1 in 4/5 folds

However, this remains validation-only post-primary exploratory evidence and
Hard-F1 is still 1.750 pp below the matched R0 control. It is therefore
promising but not yet a clean final method claim.

A required causal ablation remains: the current objective combines low-weight
auxiliary CE and explicit feature distillation. Before adding reliability
weighting, attention, or class-specific mechanisms, a matched SBN +
auxiliary-head CE-only control (lambda_feat = 0) is needed to determine whether
the observed gain is specifically attributable to transformed-view feature
distillation rather than auxiliary-task regularization alone.



---

## 15. Post-MVFD follow-up: W0 cue audit and transfer-family closure

Full decision record:

`docs/experiments/PREPROCESSING_HBP_DECISION_RECORD_2026-09-28.md`

### W0 cue audit

A post-hoc cue audit on the opened preprocessing OOF found 15/111 W0
descriptors with BH-FDR q < 0.05 in the adjusted continuous true-class
probability analysis.

The coherent family was relative detail survivability after VisuShrink:

- RGB L1-HH retained-energy ratios: positive association;
- several LH/HL retained ratios: positive association;
- VisuShrink thresholds: negative association.

Mean-RGB L1-HH retained ratio had one-way class eta-squared approximately
0.628. Pre-energy/post-energy alone did not form the stable signal family.
F0 angular descriptors did not yield a stable FDR-significant family.

This is exploratory association, not causal evidence.

### W0-RAS

Training-only scalar supervision for mean-RGB L1-HH retained-energy ratio
failed.

Mean delta vs matched R0:

- Accuracy: -0.62 pp;
- Balanced Accuracy: -0.11 pp;
- Macro-F1: -0.33 pp;
- Hard-F1: -2.12 pp;
- Worst-F1: +1.15 pp.

The auxiliary regression target was learnable, yet classification did not
improve consistently.

Decision: **STOP W0-RAS**. No lambda/band variants.

### W0-AT

Normalized stage-3 spatial attention transfer from W0 to R0 also failed.

Mean delta vs matched R0:

- Accuracy: -1.65 pp;
- Balanced Accuracy: -1.15 pp;
- Macro-F1: -1.44 pp;
- Hard-F1: -3.78 pp;
- Worst-F1: -3.18 pp.

Hard-F1 decreased on 5/5 folds.

Raw/W0 attention cosine was already about 0.984 at epoch 1 and about 0.9976 at
selected best checkpoints. The auxiliary loss did not dominate CE.

Decision: **STOP W0-AT**. No beta/stage/exponent search and no HBP+W0-AT.

### ML-MVFD

Direct intermediate tensor MSE failed strongly:

- Macro-F1: 86.78%;
- Hard-F1: 79.55%;
- Worst-F1: 57.85%;
- 0/5 folds improved versus GAP MVFD.

Decision: close direct intermediate tensor matching.

### HBP + MVFD

Applying the existing MVFD formulation to HBP embeddings did not preserve the
GAP MVFD benefit:

| Model | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| HBP R0 control | ~91.58% | ~86.23% | ~65.26% |
| HBP + MVFD | ~90.61% | ~84.38% | ~67.00% |
| Delta | -0.97 pp | -1.86 pp | +1.74 pp |

Macro improved in 1/5 folds; Hard improved in 1/5 folds.

Decision: do not stack HBP with more variants of the same MVFD formulation.

### Transfer-family decision

Completed representation-level tests now cover:

- scalar cue transfer: failed;
- normalized spatial map transfer: failed;
- full intermediate tensor transfer: failed strongly;
- final GAP feature transfer: Macro/Worst gain with Hard decline;
- final HBP feature transfer: Macro/Hard decline.

Therefore the preprocessing -> raw-only transfer family is closed on the reused
Coffee17 development folds.

This does **not** remove preprocessing from the thesis. Primary preprocessing,
complementarity, rescue analysis, and the W0 cue audit remain substantive
evidence.

---

## 16. Current canonical direct preprocessing test: R0-HBP vs W0-HBP

Canonical branch:

`codex/w0-hbp-matched-v1`

Protocol:

`docs/protocols/W0_HBP_MATCHED_V1.md`

Notebook:

`notebooks/Coffee17_W0_HBP_MATCHED_Kaggle.ipynb`

Design:

- 5 Coffee17 preprocessing-study folds;
- seed 42;
- R0-HBP control retrained;
- W0-HBP candidate trained;
- identical initialization, model, optimizer, schedule, augmentation, and
  validation rows;
- only preprocessing differs.

Retraining R0-HBP is intentional matched-control reconstruction because the
exact old per-fold HBP control artifacts are unavailable. It is not treated as
a new method.

Screening gate:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 positive in at least 3/5 folds;
3. mean paired Hard-F1 delta >= 0;
4. mean paired Worst-F1 delta >= 0.

If the gate fails, stop W0-HBP without post-hoc W0/HBP tuning.

Outer test remains untouched.

Deprecated notebooks on `codex/w0-hbp-v1`:

- `Coffee17_W0_HBP_Kaggle.ipynb`;
- `Coffee17_W0_HBP_V2_Kaggle.ipynb`;
- `Coffee17_W0_HBP_V3_Kaggle.ipynb`.

Do not use them.


---

## 17. Matched direct W0-HBP result: FAIL

Full result:

`docs/results/W0_HBP_MATCHED_RESULTS.md`

Source artifact SHA-256:

`f12670b3691dd2e482962d3edaa6fee1bfafdef9b600ceaccf87592952640bf4`

Five matched Coffee17 preprocessing-study folds completed with seed 42.
R0-HBP and W0-HBP were retrained from the same initialization and training
recipe; only input preprocessing differed. Outer test was not accessed.

| Metric | R0-HBP | W0-HBP | Delta W0-R0 |
|---|---:|---:|---:|
| Accuracy | 91.55% | 90.10% | -1.44 pp |
| Balanced Accuracy | 91.63% | 90.13% | -1.50 pp |
| Macro-F1 | 91.27% | 90.04% | -1.23 pp |
| Hard-F1 | 86.23% | 84.09% | -2.13 pp |
| Worst-F1 | 64.95% | 63.78% | -1.17 pp |

Macro-F1 improved in only **1/5 folds**.

Matched validation outcome counts across 485 unique observations:

- W0 rescue: 5;
- W0 damage: 12;
- both correct: 432;
- both wrong: 36.

Net top-1 effect: **-7 correct predictions**.

Hard-group mean delta:

- sour/black: -4.80 pp;
- shape/withered: +0.71 pp;
- insect damage: -2.41 pp.

The pre-registered/frozen screening gate failed all conditions:

- mean Macro delta > 0: FAIL;
- Macro positive in >=3/5 folds: FAIL;
- mean Hard delta >= 0: FAIL;
- mean Worst delta >= 0: FAIL.

Decision:

> **STOP W0-HBP.**

Direct W0 preprocessing does not improve the established HBP representation
under this matched development protocol. No W0/HBP hyperparameter tuning is
authorized on the reused folds.

### Updated preprocessing conclusion

The completed evidence now supports three separate statements:

1. preprocessing representations are class-dependent and complementary;
2. attempts to compress transformed-view information into a raw-only model are
   not reliably beneficial across representation levels;
3. directly feeding W0 into HBP also fails under matched evaluation.

Preprocessing remains a valid analysis contribution through the primary study,
complementarity/rescue evidence, cue audit, and the documented negative
transfer/direct-combination results. It should not be represented as a
universally accuracy-improving input transform.



---

## 18. WR-HBP V1 result: development gate PASS

Full result:

`docs/results/WAVELET_RESIDUAL_HBP_RESULTS.md`

Source artifact SHA-256:

`3fff1a877e22787b2135810afd39ad490a22e2eb43bb83d336ba4ade6c922ee9`

Scientific commit:

`01c9212965bc9040ef151204b9404d564f523a0f`

Five matched folds, seed 42, completed with identical initial RGB-HBP core and
outer test untouched.

| Metric | R0-HBP | WR-HBP | Delta |
|---|---:|---:|---:|
| Accuracy | 91.34% | 91.75% | +0.41 pp |
| Balanced Accuracy | 91.45% | 91.93% | +0.48 pp |
| Macro-F1 | 91.02% | **91.77%** | **+0.75 pp** |
| Hard-F1 | 86.12% | **86.24%** | **+0.11 pp** |
| Worst-F1 | 65.33% | **66.76%** | **+1.42 pp** |

Macro-F1 delta was positive in **5/5 folds**.

Paired validation outcomes:

- rescue: 7;
- damage: 5;
- net: +2 correct.

Selected-checkpoint wavelet gates:

[+0.1362, +0.1569, +0.1544, +0.1571, -0.1280]

Mean gate:

+0.0953 ± 0.1252.

Parameter overhead:

- +897 parameters;
- +0.0252%.

Frozen development gate:

- Macro mean positive: PASS;
- Macro positive >=3/5: PASS;
- Hard mean nonnegative: PASS;
- Worst mean nonnegative: PASS.

Decision:

> **WR-HBP PASSES DEVELOPMENT SCREENING.**

Current interpretation:

1. direct W0 replacement harms HBP;
2. tested preprocessing-to-raw transfer mechanisms do not consistently preserve
   the desired Macro/Hard/Worst behavior;
3. preserving raw RGB and adding a tiny explicit L1 wavelet-detail residual is
   the first tested preprocessing/HBP combination to satisfy all frozen
   development criteria.

This remains post-primary exploratory evidence on reused folds with seed 42.
Outer test remains untouched.



### Physical-descriptor audit preflight correction

The first notebook execution stopped during unit tests before class-wise
descriptor analysis. Red-channel Otsu failed a synthetic robustness case because
a very-dark internal spot could become the selected foreground while the pale
bean body was omitted.

The label-free bean mask was therefore revised, before seeing Coffee17 audit
outcomes, to use border-estimated background RGB distance followed by Otsu and
largest-component morphology.

Revised scientific pin:

`ab1f2d6728bd7bf78599da73069229d40271447e`

The failed attempt produced no Coffee17 physical-descriptor result, used no
outer-test materialization, and executed no neural training.



### WR-PDR deterministic preflight correction

The initial WR-PDR Kaggle run stopped during fold-1 base training because
PyTorch's CUDA `median(dim=...)` used by VisuShrink is not available under
strict deterministic algorithms.

No fold result was completed.

The correction keeps the VisuShrink formula unchanged and computes only the
median statistic on CPU when strict deterministic CUDA is active, then returns
the threshold to CUDA.

Revised scientific pin:

`33f27cb451a4b19cd90accacf55df4f79bd783df`

This is an execution-compatibility correction made before any WR-PDR outcome was
observed.



### WR-PDR second deterministic preflight correction

A second attempt stopped before completing fold 1 because CUDA
`adaptive_avg_pool2d_backward` in HBP alignment is unsupported by strict
deterministic algorithms.

The frozen HBP grids have exact integer ratios (56->7 and 14->7), so strict
deterministic execution now uses the equivalent reshape + non-overlapping block
mean. The legacy adaptive-average-pooling implementation remains unchanged
outside strict deterministic mode.

The notebook preflight now executes a complete WR-HBP CUDA forward/backward
unit test before fold training.

Revised scientific pin:

`d4c7ff0daf09e2b6f81364bca557496058e2ef27`

No WR-PDR fold result had completed before this correction.


---

## 19. WR-PDR-HBP V1 result: FAIL

Full result:

`docs/results/WR_PDR_HBP_V1_RESULTS.md`

Five strict-deterministic matched folds, seed 42, outer test untouched.

| Metric | WR-HBP | WR-PDR-HBP | Delta |
|---|---:|---:|---:|
| Accuracy | 91.75% | 90.52% | -1.24 pp |
| Balanced Accuracy | 91.91% | 90.54% | -1.37 pp |
| Macro-F1 | 91.77% | 90.33% | -1.44 pp |
| Hard-F1 | 86.41% | 83.35% | -3.05 pp |
| Worst-F1 | 66.60% | 59.93% | -6.67 pp |

Macro-F1 improved in 0/5 folds.
Paired outcomes: 0 rescue, 6 damage, net -6 correct.

Decision:

> **STOP WR-PDR-HBP.** Do not tune physical-descriptor residual weights or
> descriptor subsets on the reused development folds.

---

## 20. WR-HBP GCE V1 result: FAIL

Canonical result:

`docs/results/WR_HBP_GCE_V1_RESULTS.md`

Matched WR-HBP CE versus GCE:

| Metric | WR-HBP CE | WR-HBP GCE | Delta |
|---|---:|---:|---:|
| Accuracy | 91.75% | 91.13% | -0.62 pp |
| Balanced Accuracy | 91.91% | 90.91% | -1.01 pp |
| Macro-F1 | 91.77% | 90.71% | -1.06 pp |
| Hard-F1 | 86.41% | 85.60% | -0.80 pp |
| Worst-F1 | 66.60% | 63.00% | -3.60 pp |

Macro-F1 improved in 0/5 folds; Hard-F1 improved in 1/5 folds.
Paired outcomes: 5 rescue, 8 damage, net -3 correct.

Frozen gate: **FAIL**.

Decision:

> **STOP GCE on WR-HBP.** Loss substitution did not resolve the persistent hard
> errors.

---

## 21. WR-HBP SAR V1 result: FAIL / negligible effect

Canonical result:

`docs/results/WR_HBP_SELF_ASSESSMENT_V1_RESULTS.md`

| Metric | WR-HBP base | WR-HBP SAR | Delta |
|---|---:|---:|---:|
| Accuracy | 91.75% | 91.96% | +0.21 pp |
| Balanced Accuracy | 91.91% | 92.15% | +0.24 pp |
| Macro-F1 | 91.77% | 91.97% | +0.20 pp |
| Hard-F1 | 86.41% | 86.83% | +0.42 pp |
| Worst-F1 | 66.60% | 66.60% | 0.00 pp |

The improvement occurred in only 1/5 folds.
Paired outcomes across 485 observations: 1 rescue, 0 damage.
Worst-F1 did not improve.

Frozen screening gate: **FAIL** because Hard-F1 was not positive in at least
3/5 folds.

Decision:

> **STOP SAR V1.** The effect is too small and fold-local to support the
> proposed reassessment mechanism.

---

## 22. Frozen representation geometry audit: DINOv2 does not support H_R

Full result:

`docs/results/COFFEE17_REPRESENTATION_GEOMETRY_V1_RESULTS.md`

Five locked development folds compared frozen ImageNet MobileNetV3-Large with
frozen DINOv2-S/14 using 5-NN and a fixed-C linear probe.

Linear probe:

| Metric | MobileNetV3 | DINOv2-S/14 | Delta |
|---|---:|---:|---:|
| Macro-F1 | 63.86% | 60.52% | -3.34 pp |
| Hard-F1 | 57.63% | 54.56% | -3.07 pp |

DINOv2 Macro-F1 improved in only 1/5 folds; Hard-F1 improved in 0/5 folds.
Audited hard-pair confusion total increased from 40 to 43.

5-NN Macro-F1 delta was -5.81 pp and negative in 5/5 folds.

Of 124 validation observations for which both MobileNet decoders were wrong,
91 were also wrong under both DINOv2 decoders.

Frozen gate:

`NO_CLEAR_H_R_SUPPORT`

Decision:

> Do not proceed to DINOv2 distillation by default. A stronger frozen global
> representation did not solve the persistent Coffee17 hard errors in this
> audit.

---

## 23. DCL-style local destruction/construction V1 result: FAIL

Full result:

`docs/results/COFFEE17_DCL_LOCAL_LEARNING_V1_RESULTS.md`

Scientific code used by all folds:

`4dfacc01de28191c7bb11a00353a6bfc303adf5f`

All five folds passed the strict CUDA smoke test. The HBP inference architecture
was identical between control and treatment; outer test was untouched.

| Metric | HBP-CE | HBP-DCL | Delta |
|---|---:|---:|---:|
| Accuracy | 91.55% | 85.36% | -6.19 pp |
| Balanced Accuracy | 91.72% | 84.53% | -7.18 pp |
| Macro-F1 | 91.47% | 83.31% | -8.17 pp |
| Hard-F1 | 87.39% | 75.12% | -12.27 pp |
| Worst-F1 | 65.64% | 18.00% | -47.64 pp |

Macro-F1 and Hard-F1 improved in 0/5 folds.

Paired predictions:
- rescue: 11;
- damage: 41;
- net: -30 correct.

Preregistered hard-pair confusion total increased:

25 -> 43.

Largest pair increases:
- Withered <-> Immature: 4 -> 13;
- Partial Sour <-> Full Sour: 6 -> 16.

Frozen gate:

`NO_CLEAR_DCL_SUPPORT`

Decision:

> **STOP DCL-style destruction/construction.** Do not tune grid size, auxiliary
> weights, or closely related destructive-local variants on these reused folds.

This result rejects the tested destructive-local mechanism; it does not prove
that every local-evidence approach is invalid.

---

## 24. Anti-loop registry for older local-evidence experiments

The repository already contains implementations/configs for three local-evidence
directions whose numerical outcomes are not currently present in the canonical
result documents:

- E1 HBP local MoE:
  `configs/coffee17/E1_mobilenetv3_hbp_local_moe_source.yaml`;
- M1s spatially preserved HBP:
  `configs/coffee17/M1s_mobilenetv3_sp_hbp_source.yaml`;
- O1 HBP object crop:
  `configs/coffee17/O1_mobilenetv3_hbp_object_crop_source.yaml`.

These are **not to be treated as untested** merely because their old result
artifact is missing. Their status is
`BLOCKED_RESULT_MISSING` in
`docs/experiments/COFFEE17_EXPERIMENT_LEDGER.md`.

In particular, proposals such as "global/local max expert", "preserve spatial
HBP", "object-centric crop", or trivial variants such as Top-K pooling must first
be checked against these older experiments before any new training is
authorized.
