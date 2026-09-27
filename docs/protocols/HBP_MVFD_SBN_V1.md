# Coffee17 HBP-MVFD-SBN v1

## Research question

This experiment asks one bounded question:

> On the exact five Coffee17 development folds already used by MVFD-SBN,
> does transformed-view feature distillation add useful information when the
> deployable representation is Hierarchical Bilinear Pooling (HBP), rather
> than ordinary GAP?

The experiment is motivated by two established internal results that must not
be conflated:

1. HBP has shown its clearest benefit on the fine-grained Coffee17 task and
   much smaller or negative benefit after labels are made coarser or on
   simpler coffee datasets.
2. MVFD-SBN improves aggregate R0 performance with single-RGB deployment, but
   its existing implementation distills only a final GAP embedding.

HBP-MVFD-SBN tests whether these mechanisms are complementary.

## Why HBP, not another intermediate loss

The failed ML-MVFD-SBN v1 experiment directly matched an intermediate spatial
feature map and substantially degraded Macro-F1, Hard-F1 and Worst-F1. The
next experiment therefore does not add another intermediate matching loss.

Instead, HBP itself forms the deployable multi-level representation from three
MobileNetV3 feature depths:

```
F1, F3, F4
  -> learned 1x1 projections
  -> spatial alignment to the deepest grid
  -> pairwise interactions (F1*F3, F1*F4, F3*F4)
  -> signed square root + L2 normalization
  -> concatenation
  -> linear 17-class classifier
```

The exact repository implementation is retained. No new attention, FPN,
projector, local expert, or inference branch is introduced.

## Matched factorial slice

For the exact same reconstructed Coffee17 fold and seed 42:

| Code | Deployable representation | Training views | MVFD |
|---|---|---|---|
| G0 | GAP | R0 | no |
| GM | GAP | R0/C0/F0/W0 | yes |
| H0 | HBP | R0 | no |
| HM | HBP | R0/C0/F0/W0 | yes |

G0 is the already-frozen matched R0 reference. GM is the earlier frozen
MVFD-SBN experiment and is not reconstructed or hard-coded in this branch.
This branch trains only H0 and HM. Their primary paired comparison is valid
without GM.

A complete two-factor interaction,

```
(HM - H0) - (GM - G0)
```

may only be computed after the exact frozen GM artifact is supplied. The
runner deliberately refuses to recreate that value from memory.

## HBP architecture

Locked model:

- backbone: MobileNetV3-Large;
- input: 224;
- HBP out_indices: [1, 3, 4];
- projection_dim: 512;
- three normalized pairwise bilinear blocks;
- concatenated embedding dimension: 1536;
- linear 17-class classifier.

The matched H0 and HM arms are initialized from the exact same seed-42 HBP
state fingerprint.

## MVFD on the HBP embedding

For each training sample:

```
z_R = HBP(f(R0(x)))
z_C = HBP(f(C0(x)))
z_F = HBP(f(F0(x)))
z_W = HBP(f(W0(x)))
```

Training-only teacher:

```
t = stopgrad((z_C + z_F + z_W) / 3)
```

Feature transfer:

```
L_feat = mean_i ||z_R_i - t_i||_2^2
```

Total HM objective:

```
L_HM =
    CE(h_R(z_R), y)
  + 0.05 * [CE(h_C(z_C), y) + CE(h_F(z_F), y) + CE(h_W(z_W), y)]
  + 0.007 * L_feat
```

The coefficients are inherited unchanged from frozen MVFD-SBN. They are not
retuned on Coffee17.

H0 uses only:

```
L_H0 = CE(h_R(z_R), y)
```

## Selective BatchNorm

HM retains the frozen SBN contract:

- R0 is the only view allowed to update persistent running statistics;
- C0/F0/W0 use current mini-batch statistics without persisting them;
- affine BN parameters remain shared and trainable.

This contract also applies to BatchNorm layers inside the HBP projection
blocks because they are part of the same base model.

## Deployment

H0 and HM have the same deployed architecture:

```
raw RGB -> MobileNetV3-Large -> HBP -> Linear17
```

For HM, C0/F0/W0, auxiliary classifiers, teacher construction and
distillation loss are discarded at inference.

## Locked controls

Unchanged:

- Coffee17 original identities: 979 images / 17 classes;
- exact five development folds from the preprocessing-study authority;
- validation identity hashes must match the frozen reference;
- seed 42;
- 50 epochs;
- AdamW, lr 3e-4, weight decay 1e-4;
- cosine schedule;
- label smoothing 0.1;
- deterministic training rotations from the preprocessing-study protocol;
- object crop off;
- EMA off;
- best checkpoint selected only by R0 validation Macro-F1;
- outer test not accessed.

## Primary endpoint

Primary comparison:

```
HM - H0
```

Metrics:

- Macro-F1;
- Hard-F1;
- Worst-F1;
- Accuracy and balanced accuracy as secondary metrics.

Exploratory screening gate:

1. mean Macro-F1 delta > 0;
2. positive Macro-F1 delta in at least 3/5 folds;
3. mean Hard-F1 delta > 0;
4. mean Worst-F1 delta >= -0.01.

Passing is exploratory evidence only because the same development folds have
already been reused.

## Secondary analysis

The package also reports:

```
H0 - G0
HM - G0
```

This allows checking whether HBP itself behaves similarly on the current
five-fold development authority before interpreting HM.

The full interaction with GM is deferred until the exact earlier MVFD-SBN
artifact is available.

## Granularity follow-up

The previously observed Fine-17 versus Coarse-9 behavior is treated as a
mechanistic hypothesis, not as a reason to tune this run.

If HM passes the Fine-17 gate, a separate locked follow-up may compare the same
method under the existing Fine-17 and Coarse-9 label mappings. The relevant
question is whether the incremental HBP/MVFD benefit becomes smaller when
visually similar defect subclasses are merged.

No Coarse-9 tuning is allowed before the Fine-17 result is frozen.

## Causal follow-up

If HM passes, a separate HBP+AUXCE-SBN arm should be run before a final method
claim. This isolates the contribution of explicit HBP-embedding distillation
from the contribution of transformed-view auxiliary CE alone.

## Claim boundary

A Fine-17 pass supports only:

> On the reused Coffee17 development folds, transformed preprocessing-view
> distillation adds useful training signal to an HBP-based deployable
> representation under raw-RGB-only inference.

It does not by itself prove:

- that HBP and MVFD have a positive statistical interaction versus frozen GM;
- that the gain is caused by texture, color, contour, or any specific visual
  cue;
- that the effect generalizes to Coarse-9 or another dataset;
- state of the art.
