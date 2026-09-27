# Coffee17 MVSC-MVFD-SBN v1

## Status

Post-primary exploratory extension of frozen MVFD-SBN.

The preceding ML-MVFD-SBN v1 experiment added elementwise spatial squared-L2
transfer at an intermediate MobileNetV3 stage and failed its frozen screening
gate. MVSC-MVFD-SBN keeps the successful frozen MVFD-SBN final-GAP transfer
unchanged and replaces only that failed intermediate objective.

## Parent method retained

Frozen MVFD-SBN remains:

- one shared MobileNetV3-Large backbone;
- R0 raw-RGB primary path;
- C0/F0/W0 deterministic training-only transformed views;
- Selective BatchNorm;
- separate auxiliary classifiers during training;
- equal-mean C0/F0/W0 teacher at the final GAP embedding;
- final squared-L2 feature-distillation coefficient 0.007;
- auxiliary CE coefficient 0.05 for each transformed view;
- checkpoint selection only from R0 validation Macro-F1;
- R0-only inference.

Final MVFD term:

```
e_R = GAP(F_d^R)
t_d = stopgrad((e_C + e_F + e_W) / 3)
L_final = ||e_R - t_d||_2^2
```

## New intermediate objective

MobileNetV3 exposes `out_indices=[3,4]`. The final stage remains the
deployment feature. The earlier exposed map is used only during training.

For each view, pool the intermediate map spatially and L2-normalize:

```
z_v = normalize(GAP(F_m^v))
```

R0 is the only contrastive anchor. C0/F0/W0 are detached keys:

```
K = stopgrad([z_C; z_F; z_W])
```

For R0 anchor i, every transformed key with the same ground-truth class is a
positive, including the three transformed versions of the same image. Keys
with a different class are negatives.

```
P(i) = {k : y_k = y_i}
s(i,k) = z_R_i^T K_k / tau

L_MVSC = mean_i [
  -1/|P(i)| sum_{p in P(i)}
   log exp(s(i,p)) / sum_a exp(s(i,a))
]
```

The temperature is fixed to `tau=0.07`, following the standard supervised
contrastive formulation of Khosla et al. (NeurIPS 2020). It is not tuned on
Coffee17.

The coefficient is fixed at `0.05`. This is a bounded exploratory
regularization weight, matched to the already-frozen per-view auxiliary CE
coefficient rather than selected by a Coffee17 sweep. It is not claimed
optimal.

## Objective

```
L = CE_R
  + 0.05 [CE_C + CE_F + CE_W]
  + 0.007 L_final
  + 0.05 L_MVSC
```

There is no intermediate elementwise feature matching.

## Why this mechanism

Coffee17 contains visually similar defect classes. Coffee-specific prior work
supports preserving and exploiting intermediate/multiscale representations,
while prior coffee work by Yang et al. (ICCE-TW 2021) already used
activation-based middle-layer knowledge transfer. Therefore this experiment
does not repeat direct activation matching. It instead asks whether transformed
training views can provide class-discriminative relational supervision to the
R0 intermediate representation.

The generic basis is supervised contrastive learning: same-class
representations are pulled together and different-class representations are
separated. The adaptation here is deliberately asymmetric: R0 anchors learn
from detached C0/F0/W0 keys so the extra objective directly serves the
deployment representation.

## Experimental controls

Unchanged from the frozen Coffee17 development authority:

- 979 original images / 17 classes;
- exact deterministic five-fold identity split;
- seed 42;
- 97 validation images per fold;
- matched R0 validation hashes;
- matched frozen-MVFD validation hashes;
- matched initial primary-model fingerprint;
- 50 epochs;
- AdamW lr 0.0003;
- weight decay 0.0001;
- cosine scheduler;
- label smoothing 0.1;
- preprocessing-study augmentation;
- outer test not accessed.

No projector, attention module, FPN, bilinear head, extra inference branch, or
new deployment parameter is added.

## Diagnostics

At every epoch log:

- R0/C0/F0/W0 CE;
- final MVFD feature loss and weighted contribution;
- MVSC loss and weighted contribution;
- mean number of positive and negative keys per R0 anchor;
- mean positive and negative cosine similarity;
- positive-minus-negative cosine gap;
- final-teacher cosine/L2;
- R0 validation Macro-F1, Hard-F1 and Worst-F1.

## Frozen screening gate

Primary comparison remains frozen MVFD-SBN:

1. mean Macro-F1 improves;
2. Macro-F1 improves in at least 3/5 folds;
3. mean Hard-F1 improves;
4. mean Worst-F1 does not drop by more than 1 percentage point.

Passing is exploratory evidence only because the same development folds have
already been reused.

## Claim boundary

If the gate passes, the supported claim is limited to:

> On the reused Coffee17 development protocol, R0-anchor multi-view supervised
> contrastive regularization at an intermediate feature stage adds useful
> class-discriminative training signal beyond frozen final MVFD-SBN while
> preserving raw-RGB-only deployment.

It does not prove state of the art, independent generalization, or that one
specific visual cue is the causal reason for the gain.
