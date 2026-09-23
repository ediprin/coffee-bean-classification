# Coffee17 MVFD-SBN v1

## Purpose

MVFD-SBN is the first explicit transformed-view -> R0 transfer experiment.

The preceding AT-SBN experiment kept F0 optimization healthy but produced
essentially no R0-only Macro-F1 gain over the matched R0 control. Its
self-distillation direction was R0 -> auxiliary, and the auxiliary predictions
became very close to R0 on training samples. MVFD-SBN reverses the information
flow at feature level.

## Published basis

Dong et al. (Neurocomputing 2026) construct a training-only multi-view teacher
feature and minimize squared-L2 distance between student and teacher features.
Their Eq. 8 is a squared-L2 feature consistency loss, and they report a single
lambda_kd = 0.007 across all evaluated datasets after selecting it on a held-out
validation split.

Black & Souvenir (WACV 2024) independently show that cross-view training can
improve single-view inference and that one-way multi-view -> single-view
distillation can perform comparably to their full mutual-distillation variant.

This Coffee17 adaptation does not reproduce either paper literally:
- our views are deterministic preprocessing views of the same bean;
- the teacher is an equal mean of C0/F0/W0 GAP embeddings;
- we do not introduce Dong et al.'s SGFP/CVPA/RGA modules;
- we retain the Coffee17 SBN mechanism already validated experimentally.

The purpose is one controlled question: does an explicit transformed-view
feature target improve the R0 deployment representation?

## Architecture

Shared:
- one MobileNetV3-Large feature extractor;
- GAP representation;
- shared BN affine parameters.

Heads:
- primary R0 linear classifier;
- training-only C0/F0/W0 linear classifiers.

Selective BatchNorm:
- R0 alone updates persistent running mean/variance;
- C0/F0/W0 use their current mini-batch statistics without changing deployment
  running statistics.

## Teacher construction

For a training sample:

e_R = GAP(f(R0(x)))
e_C = GAP(f(C0(x)))
e_F = GAP(f(F0(x)))
e_W = GAP(f(W0(x)))

The training-only teacher is:

t = stopgrad((e_C + e_F + e_W) / 3)

Equal aggregation is deliberately used in v1. Earlier Coffee17 experiments
showed that validation-learned weighting is unstable on the approximately
97-image validation folds. No reliability MLP or class routing is introduced.

## Objective

Primary CE:

L_R = CE(h_R(e_R), y)

Auxiliary CE:

L_aux = CE(h_C(e_C),y) + CE(h_F(e_F),y) + CE(h_W(e_W),y)

Feature distillation:

L_feat = mean_i || e_R_i - t_i ||_2^2

Total:

L = L_R + 0.05 L_aux + 0.007 L_feat

The 0.05 auxiliary CE weight is retained from the preceding auxiliary-training
protocol. The 0.007 feature-distillation coefficient is the fixed coefficient
reported by Dong et al. for their squared-L2 consistency term. It is a direct
published-value transfer, not a Coffee17 validation-tuned value.

There is no output-level KD and no late classifier merging.

## Directionality

The teacher prototype is detached. Therefore L_feat updates the R0 student
feature toward the transformed-view prototype but does not pull C0/F0/W0 toward
R0 through the feature-distillation term.

The explicit feature-transfer direction is:

C0/F0/W0 -> detached multi-view prototype -> R0

Auxiliary CE still trains the shared feature extractor and the auxiliary heads.

## Model selection

Only R0-primary validation Macro-F1 selects the best checkpoint.

Auxiliary validation metrics and feature-similarity diagnostics never select a
checkpoint.

## Diagnostics

Every epoch logs:
- primary CE;
- C0/F0/W0 CE;
- raw feature-distillation loss and weighted contribution;
- R0 and teacher feature norms;
- R0-to-teacher cosine similarity and L2 distance;
- R0-to-each-view cosine similarity and L2 distance;
- R0-vs-auxiliary top-1 disagreement;
- R0 validation Macro-F1, Hard-F1 and Worst-F1.

The final package also stores auxiliary validation metrics at the R0-selected
best checkpoint.

## Inference

At deployment, discard:
- C0/F0/W0 preprocessing;
- all auxiliary classifiers;
- teacher construction;
- feature-distillation loss.

Inference remains:

raw RGB -> R0 preprocessing -> MobileNetV3-Large -> GAP -> primary R0 classifier

No extra inference parameter or forward pass is retained.

## Scope

This remains post-primary exploratory development. The previously opened outer
OOF/test population must not be used for tuning or model selection.
