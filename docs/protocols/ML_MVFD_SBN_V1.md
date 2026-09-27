# Coffee17 ML-MVFD-SBN v1

## Status

Exploratory extension of the frozen MVFD-SBN method.

This experiment does **not** replace MVFD-SBN and does not reopen the final
method claim. It tests one additional mechanistic hypothesis on the same
development folds: whether retaining the original final-GAP feature transfer
while also transferring an intermediate spatial feature map can reduce the
hard-class trade-off observed in frozen MVFD-SBN.

## Parent method retained exactly

Frozen MVFD-SBN components retained:

- one shared MobileNetV3-Large backbone;
- R0 raw-RGB primary path;
- C0/F0/W0 deterministic transformed training views;
- Selective BatchNorm;
- separate training-only auxiliary classifiers;
- equal-mean transformed-view final GAP teacher;
- final feature-distillation coefficient 0.007;
- auxiliary CE coefficient 0.05 per transformed view;
- R0-only checkpoint selection;
- raw-RGB-only deployment.

Frozen final teacher:

`e_R = GAP(F_d^R)`

`t_d = stopgrad((e_C + e_F + e_W) / 3)`

`L_final = ||e_R - t_d||_2^2`

## Extension

MobileNetV3-Large now exposes the penultimate selected feature stage together
with the final stage:

`out_indices = [3, 4]`

The final stage remains the input to GAP, so the deployment classifier is
unchanged.

For the intermediate feature map:

`F_m^R, F_m^C, F_m^F, F_m^W`

construct:

`t_m = stopgrad((F_m^C + F_m^F + F_m^W) / 3)`

The intermediate loss preserves spatial correspondence:

`L_mid = mean_i sum_c mean_hw (F_m^R - t_m)^2`

All four views are geometry-aligned because C0/F0/W0 are deterministic
photometric/frequency transforms of the same already-augmented training image.

## Objective

`L = CE_R`
`  + 0.05 [CE_C + CE_F + CE_W]`
`  + 0.007 L_final`
`  + 0.007 L_mid`

The original MVFD final feature loss remains unchanged. The only new mechanism
is the intermediate spatial feature-transfer term.

The v1 intermediate coefficient is fixed to 0.007 without a Coffee17 sweep.
It is deliberately not tuned on the five reused development folds. Its purpose
is a bounded first test, not a claim that 0.007 is optimal for intermediate
features.

## Why this is an extension rather than a replacement

The frozen MVFD result already supports explicit transformed-view -> R0
transfer at the final embedding. ML-MVFD-SBN keeps that active mechanism and
asks whether an earlier spatially resolved representation contributes
additional information.

No attention, routing, residual teacher, learned weighting, bilinear head, or
extra inference module is added.

## Experimental controls

The experiment must reproduce the exact Coffee17 development authority:

- 979 original images / 17 classes;
- deterministic five-fold identity split;
- seed 42;
- 97 validation images per fold;
- matched R0 validation hashes;
- frozen MVFD validation hashes;
- matched initial model-state fingerprint;
- 50 epochs;
- AdamW lr 0.0003;
- weight decay 0.0001;
- cosine scheduler;
- label smoothing 0.1;
- preprocessing-study augmentation;
- best checkpoint selected only by R0 validation Macro-F1;
- outer test not accessed.

The run must abort if exposing `out_indices=[3,4]` changes the primary initial
model-state fingerprint relative to the matched R0 reference.

## Diagnostics

Log at every epoch:

- R0/C0/F0/W0 CE;
- final feature loss and weighted contribution;
- intermediate feature loss and weighted contribution;
- R0-final-teacher cosine and L2;
- R0-mid-teacher cosine and L2;
- R0 vs auxiliary top-1 disagreement;
- R0 validation Macro-F1, Hard-F1 and Worst-F1.

## Frozen screening gate

Relative to frozen MVFD-SBN:

1. mean Macro-F1 must improve;
2. Macro-F1 must improve on at least 3/5 folds;
3. mean Hard-F1 must improve;
4. mean Worst-F1 may not decrease by more than 1 percentage point.

PASS only authorizes further independent confirmation planning. It does not
turn the reused development folds into confirmatory evidence.

## Claim boundary

Supported if the gate passes:

> On the reused Coffee17 development protocol, adding intermediate spatial
> transformed-view feature transfer to frozen MVFD-SBN improves the selected
> aggregate and hard-class validation metrics while preserving raw-RGB-only
> deployment.

Unsupported:

- intermediate features are proven to contain coffee-defect texture;
- downsampling is proven to be the causal source of all Coffee17 errors;
- the extension is independently confirmed;
- the method is state of the art.

If the gate fails, report the failure and keep frozen MVFD-SBN unchanged.
