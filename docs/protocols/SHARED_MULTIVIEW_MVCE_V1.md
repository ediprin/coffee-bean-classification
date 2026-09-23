# Coffee17 shared-weight multi-view CE — v1

## Purpose

This is a post-primary exploratory method-development experiment. It asks a
single question: can one RGB-deployed MobileNetV3-Large absorb useful
R0/C0/F0/W0 preprocessing variation during training without teacher models or
multiple inference branches?

## Data authority

Frozen primary checkpoints are no longer required. The repository contains a
compact validation authority reconstructed from the saved KD analysis artifact.
For every fold, the reconstructed validation split must match:

- exactly 97 validation images;
- the frozen SHA-256 over sorted `class/filename<TAB>class` rows;
- seed 42 and the original deterministic fold-generation code.

The initial model-state fingerprint must also equal the fingerprint preserved in
the saved KD artifact.

No outer-test images are materialized or evaluated.

## Model and views

A single MobileNetV3-Large + GAP model is shared across all training views:

- R0 raw RGB;
- C0 Lab-L CLAHE;
- F0 luminance AFAB-2;
- W0 Haar4 VisuShrink reconstruction.

The geometric augmentation is performed once by the original deterministic
training loader. All four preprocessing frontends therefore receive the exact
same rotated/resized base image.

## Objective

For a sample `(x,y)`:

`L_raw = CE(f(R0(x)), y)`

`L_aux = [CE(f(C0(x)),y) + CE(f(F0(x)),y) + CE(f(W0(x)),y)] / 3`

`L = 0.5 * L_raw + 0.5 * L_aux`

There is no teacher, pseudo-label, learned view weight, class-dependent gate,
consistency loss, or feature-fusion module in v1.

## BatchNorm contract

Deployment uses raw RGB only. To prevent auxiliary preprocessing distributions
from overwriting deployment-domain BatchNorm running statistics:

- the R0 forward is the only forward allowed to update BN running mean/variance;
- C0/F0/W0 forwards use BN in evaluation-statistics mode;
- shared convolutional/classifier weights and BN affine parameters still
  receive gradients from all four views.

This adds no inference parameters.

## Training recipe

The original R0 recipe is retained:

- seed 42;
- 224x224;
- batch 32;
- deterministic rotations [0,45,90,135,180,225,270];
- AdamW, lr 3e-4, weight decay 1e-4;
- cross entropy with label smoothing 0.1;
- cosine schedule;
- 50 epochs;
- best epoch selected by validation Macro-F1.

## Inference

Only one path remains:

`raw RGB -> R0 normalization -> MobileNetV3-Large -> GAP -> 17 classes`

C0/F0/W0 are training-only and are discarded at deployment.
