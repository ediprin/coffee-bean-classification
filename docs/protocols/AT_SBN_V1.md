# Coffee17 AT-SBN v1

## Purpose

AT-SBN is the next controlled experiment after MVCE-SBN.

MVCE-SBN proved that selective BatchNorm removes the severe F0 optimization
collapse, but R0-only inference still remained approximately tied with the
matched R0 control. The unresolved question is therefore cross-view transfer:
can transformed-view supervision improve the shared representation used by the
raw R0 deployment path?

## Published basis

The closest published analogue is Zhang et al., CVPR 2020,
*Auxiliary Training: Towards Accurate and Robust Models*.

Their released implementation uses:
- one primary clean classifier;
- separate auxiliary classifiers for transformed inputs;
- selective BatchNorm behavior;
- auxiliary CE weight 0.05 per transformed view;
- clean-to-auxiliary self-distillation weight 0.05 per transformed view;
- detached clean logits as the soft target;
- a late squared-L2 classifier-weight merging penalty;
- auxiliary classifiers discarded at inference.

Their released CIFAR code defaults to 210 epochs and activates the weight
merging term when the zero-based epoch index reaches 180, i.e. from the 181st
epoch. AT-SBN maps that late-training fraction to the existing frozen 50-epoch
Coffee17 recipe: the first merge-active epoch is 44. This proportional mapping
is an adaptation, not a claim that the original paper used 50 epochs.

We deliberately do not copy the paper's auxiliary attention/bottleneck module
in v1. The immediate Coffee17 test isolates classifier decoupling,
self-distillation and late merging on top of the already validated SBN
mechanism.

## Architecture

Shared deployable components:
- MobileNetV3-Large feature extractor;
- GAP;
- primary R0 linear classifier.

Training-only components:
- one linear classifier for C0;
- one linear classifier for F0;
- one linear classifier for W0.

The auxiliary classifiers are created after the primary base model and their
initialization is RNG-isolated. Therefore the primary model-state fingerprint
remains exactly paired with the frozen matched R0 control.

## Selective BatchNorm

R0:
- normal BN training behavior;
- only R0 updates persistent running mean/variance.

C0/F0/W0:
- each forward uses its own current mini-batch mean/variance;
- transformed views do not modify persistent running statistics;
- shared BN affine parameters remain trainable.

This is the SBN mechanism that removed the previous F0 loss collapse.

## Objective

For the primary view:

z_R = h_R(f(R0(x)))

For each auxiliary view v in {C0,F0,W0}:

z_v = h_v(f(v(x)))

Primary classification:

L_R = CE(z_R, y)

Auxiliary classification:

L_aux = sum_v CE(z_v, y)

Zhang-style clean-to-auxiliary self-distillation at temperature 1:

L_KD = sum_v H(softmax(stopgrad(z_R)), softmax(z_v))

For epochs 1-43:

L = L_R + 0.05 L_aux + 0.05 L_KD

From epoch 44 onward:

L_merge = sum_v [
  ||W_v - W_R||_2^2 + ||b_v - b_R||_2^2
]

L = L_R + 0.05 L_aux + 0.05 L_KD + L_merge

The baseline Coffee17 label smoothing of 0.1 is retained in the hard-label CE
terms. No extra dropout is added to the auxiliary classifiers.

## Model selection

Only R0-primary validation Macro-F1 selects the best checkpoint.

Auxiliary validation metrics, disagreement, JS divergence and classifier
distances are diagnostics only and never select a checkpoint.

## Inference

All auxiliary classifiers and transformed inputs are discarded.

Deployment remains:

raw RGB -> R0 preprocessing -> MobileNetV3-Large -> GAP -> primary linear head

Therefore AT-SBN adds:
- zero inference parameters;
- zero inference preprocessing branches;
- zero extra inference forward passes.

## Diagnostic outputs

Every epoch logs:
- primary CE;
- C0/F0/W0 CE;
- C0/F0/W0 self-distillation loss;
- R0-vs-aux top-1 disagreement;
- R0-vs-aux Jensen-Shannon divergence;
- classifier weight distances;
- merge-stage state;
- R0-primary validation metrics.

The final package also contains auxiliary-view validation metrics at the
R0-selected best checkpoint.

## Scope

This remains post-primary exploratory method development. The previously
opened outer OOF/test population must not be used for model selection or
method tuning.
