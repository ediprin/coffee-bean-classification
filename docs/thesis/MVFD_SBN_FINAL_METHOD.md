# Frozen Thesis Method: MVFD-SBN

## Status

**Frozen candidate method.**

Method development is closed on the current Coffee17 validation protocol. No
additional teacher weighting, routing, attention, ensemble teacher, loss sweep,
or architecture stacking should be introduced without a new independent
evaluation basis.

Frozen scientific implementation:
- method: `MVFD-SBN`
- scientific code commit: `fdae1fff10348f78ca67bb00948ae5768e9c98d9`
- config: `configs/mvfd_sbn/MVFD_SBN_ALL4.yaml`
- result artifact SHA-256:
  `d2f432be885a9c507fdca0cbeb02d8d3dbef70d2080f306c8d84d84bb6ffecf6`

## Research question

The final method does **not** claim that one fixed preprocessing transform is
universally superior to raw RGB.

The supported research question is:

> Can deterministic color/contrast/frequency preprocessing views provide
> complementary privileged representations during training, and can that
> information be transferred into a single lightweight raw-RGB deployment
> model?

The experimental sequence supports this framing:
1. fixed preprocessing alone did not provide a convincing aggregate
   replacement for raw RGB;
2. C0/F0/W0 showed class-dependent complementary information;
3. naive shared multi-view training suffered from normalization mismatch;
4. Selective BatchNorm removed the transformed-view optimization collapse;
5. auxiliary transformed-view classification alone did not improve the R0
   deployment model;
6. explicit transformed-view feature distillation produced the first
   non-trivial positive aggregate shift over the matched R0 control;
7. confidence-aware teacher weighting did not improve over the simpler
   equal-mean teacher.

## Backbone and deployment target

Backbone:

`MobileNetV3-Large -> GAP -> Linear(17)`

The backbone is held fixed throughout the final causal comparisons. This keeps
the experiment focused on the contribution of preprocessing-derived
representations rather than architecture changes.

At inference there is only one raw-RGB path:

```
raw RGB
   |
   v
MobileNetV3-Large
   |
   v
GAP
   |
   v
17-class linear classifier
```

There is no C0/F0/W0 preprocessing, auxiliary classifier, teacher, fusion,
routing, or extra forward pass at deployment.

## Training views

For each training image `x`:

- **R0**: raw RGB
- **C0**: Lab-L luminance CLAHE
- **F0**: luminance + angular/frequency preprocessing
- **W0**: Haar-wavelet + VisuShrink reconstruction

All views share the same MobileNetV3-Large feature extractor.

Let:

`e_R = GAP(f(R0(x)))`

`e_C = GAP(f(C0(x)))`

`e_F = GAP(f(F0(x)))`

`e_W = GAP(f(W0(x)))`

The primary R0 classifier is `h_R`; C0/F0/W0 use separate training-only
linear classifiers `h_C`, `h_F`, and `h_W`.

## Selective Batch Normalization

R0 follows ordinary training-mode BatchNorm and is the only view allowed to
update persistent running mean/variance.

For C0/F0/W0:
- the current transformed mini-batch statistics are used;
- persistent running statistics are not updated;
- BatchNorm affine parameters remain shared and trainable.

Thus deployment statistics remain aligned with the raw-RGB R0 path.

## Equal-mean privileged teacher

The transformed-view teacher is training-only:

`t = stopgrad((e_C + e_F + e_W) / 3)`

The equal mean is retained because the later confidence-aware refinement failed
to improve the frozen MVFD result and produced weights that remained nearly
uniform.

## Objective

Primary classification:

`L_R = CE(h_R(e_R), y)`

Auxiliary transformed-view classification:

`L_aux = CE(h_C(e_C), y) + CE(h_F(e_F), y) + CE(h_W(e_W), y)`

Feature distillation:

`L_feat = mean_i ||e_R_i - t_i||_2^2`

Final objective:

`L = L_R + 0.05 L_aux + 0.007 L_feat`

The explicit knowledge-transfer direction is:

```
C0 ----\
F0 ----- > equal-mean transformed-view feature teacher ---> R0 feature
W0 ----/
```

The teacher is detached in the feature-distillation term.

## Training contract

Frozen settings:
- image size: 224
- batch size: 32
- epochs: 50
- optimizer: AdamW
- learning rate: 0.0003
- weight decay: 0.0001
- scheduler: cosine
- label smoothing: 0.1
- augmentation mode: preprocessing-study
- rotation angles: 0, 45, 90, 135, 180, 225, 270 degrees
- object crop: disabled
- EMA: disabled
- checkpoint selection: R0-primary validation Macro-F1 only

No transformed-view metric may select the deployment checkpoint.

## Frozen validation result

Matched R0 control:

- Accuracy: 90.93%
- Balanced Accuracy: 90.56%
- Macro-F1: 90.65%
- Hard-F1: 87.01%
- Worst-F1: 64.55%

MVFD-SBN:

- Accuracy: 91.55%
- Balanced Accuracy: 92.09%
- Macro-F1: 91.77%
- Hard-F1: 85.26%
- Worst-F1: 67.88%

Delta MVFD-SBN vs matched R0:

- Accuracy: +0.619 pp
- Balanced Accuracy: +1.527 pp
- Macro-F1: +1.115 pp
- Hard-F1: -1.750 pp
- Worst-F1: +3.333 pp

Macro-F1 improves in 4/5 folds.

The Hard-F1 decrease is an explicit limitation and must not be hidden.

## Causal ablation

Removing only feature distillation gives:

`L_control = L_R + 0.05 L_aux`

AUXCE-SBN control Macro-F1:

`89.94%`

Full MVFD-SBN Macro-F1:

`91.77%`

Difference:

`+1.826 pp`

MVFD-SBN exceeds the AUXCE-SBN control in Macro-F1 on all 5 folds.

Representation diagnostics also change in the intended direction:

- R0-teacher cosine: approximately 0.953 -> 0.990
- R0-teacher L2 distance: approximately 6.43 -> 2.43
- raw feature discrepancy: approximately 43.57 -> 6.24

This supports explicit transformed-view feature transfer as an active
contribution rather than auxiliary classification alone.

## Rejected refinement

CA-MVFD-SBN replaced the equal mean by sample-wise CE-based confidence weights.

It did not improve MVFD-SBN:

- Macro-F1: 91.38% vs 91.77%
- delta: -0.384 pp
- positive folds vs equal-mean MVFD: 1/5

The learned-free confidence weights were effectively near uniform:
- C0: 0.3361
- F0: 0.3288
- W0: 0.3351
- mean maximum weight: 0.3490
- entropy: 1.09598 vs theoretical 3-view maximum ln(3)=1.09861

Therefore reliability weighting is not retained.

## Thesis claim boundary

Supported wording:

> Deterministic preprocessing views provide complementary training-time
> representations, and explicit feature-level transfer from these transformed
> views can improve the aggregate raw-RGB representation of a lightweight
> MobileNetV3-Large classifier under the frozen Coffee17 validation protocol.

Unsupported wording:
- preprocessing is universally better than raw RGB;
- MVFD-SBN is confirmed superior on an untouched test population;
- the method improves every difficult class;
- reliability weighting is beneficial;
- the current result is a state-of-the-art claim.

The current Coffee17 method-development evidence is post-primary exploratory.
A genuinely independent evaluation is required for a clean confirmatory
performance claim.
