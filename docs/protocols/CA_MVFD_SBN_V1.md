# Coffee17 CA-MVFD-SBN v1

## Purpose

CA-MVFD-SBN tests one specific limitation of the frozen MVFD-SBN method:
the teacher prototype currently treats C0, F0 and W0 equally for every sample.

The existing Coffee17 evidence shows that preprocessing effects are strongly
class- and sample-dependent. The equal-mean teacher may therefore dilute a
useful transformed view when another transformed view is unreliable.

This experiment changes only teacher aggregation. Backbone, views, SBN,
auxiliary CE weights, feature-distillation coefficient, optimizer, schedule,
model selection and R0-only inference remain frozen.

## Published basis

Dong et al. (Neurocomputing 2026) explicitly state that simple multi-view
averaging can dilute discriminative cues when view quality differs, motivating
reliability-guided aggregation.

Zhang, Chen and Wang (ICASSP 2022), Confidence-Aware Multi-Teacher Knowledge
Distillation (CA-MKD), assign sample-wise teacher reliability from the
cross-entropy between each teacher prediction and the ground-truth label.
Their weighting rule is:

w_k = 1/(K-1) * [1 - exp(CE_k) / sum_j exp(CE_j)]

They also extend confidence-aware weighting to intermediate feature transfer
and report that simple average weighting underperforms their confidence-aware
approach.

CA-MVFD-SBN adapts this principle from multiple teacher networks to multiple
deterministic preprocessing views of one shared lightweight backbone.

## Teacher construction

For transformed view v in {C0,F0,W0}, obtain:
- GAP embedding e_v
- auxiliary logits z_v

For each training sample:

CE_v = CE(z_v, y)

Define the source-inspired confidence weight:

beta_v =
  1/(K-1) *
  [1 - exp(CE_v) / sum_u exp(CE_u)]

with K=3.

Implementation uses softmax over the CE values for numerical stability:

badness = softmax([CE_C, CE_F, CE_W])

beta = (1 - badness) / (K - 1)

Thus:
- beta_C + beta_F + beta_W = 1
- a transformed view with larger ground-truth CE receives smaller weight
- no learned routing network is added
- no validation-derived view weights are used

Both beta and the resulting teacher are stop-gradient for the feature-
distillation path.

Teacher:

t = stopgrad(beta_C e_C + beta_F e_F + beta_W e_W)

## Objective

Exactly as frozen MVFD-SBN except teacher aggregation:

L =
  CE_R
  + 0.05 [CE_C + CE_F + CE_W]
  + 0.007 mean_i ||e_R_i - t_i||_2^2

There is no hyperparameter sweep.

## Diagnostics

Every epoch additionally records:
- mean beta_C, beta_F, beta_W
- mean maximum teacher weight
- teacher-weight entropy
- class-conditional mean C0/F0/W0 weights
- R0-teacher cosine and L2 distance
- per-view CE and R0-vs-view disagreement

The purpose is not only to measure performance but to verify whether the
aggregation actually behaves sample- and class-dependently.

## Primary comparison

CA-MVFD-SBN is compared against the already frozen equal-mean MVFD-SBN result.

Interpretation:
- if CA-MVFD-SBN improves consistently, equal averaging was a meaningful
  bottleneck;
- if it ties or degrades, the frozen equal-mean teacher remains preferable and
  no further reliability module should be added.

## Inference

Unchanged from MVFD-SBN:

raw RGB -> MobileNetV3-Large -> GAP -> primary 17-class classifier

C0/F0/W0, auxiliary heads, confidence weighting and feature teacher are all
training-only and discarded at deployment.

## Scope

This is post-primary exploratory method development. The outer test remains
closed during this experiment.
