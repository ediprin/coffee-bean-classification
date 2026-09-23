# Coffee17 MVCE-SBN v1

## Empirical trigger

The first shared-weight multi-view experiment used R0 BatchNorm running
statistics for auxiliary C0/F0/W0 forwards. It underperformed a matched R0
control by 1.72 pp mean validation Macro-F1. At the best validation epoch,
the F0 training CE remained near random-17-class CE on folds 2, 3 and 5
(2.641, 2.773 and 2.832), whereas C0/W0 were around 0.6-0.8.

This v1 changes only normalization behavior. It does not tune view weights,
drop F0, add a consistency loss, add attention, or alter the backbone.

## Literature basis

Zhang et al. (CVPR 2020), *Auxiliary Training: Towards Accurate and Robust
Models*, define selective batch normalization: clean and corrupted examples
are normalized with their own current batch statistics, while running
statistics used for deployment are updated only from clean examples. Their
CIFAR100 ablation reports 79.47% accuracy for the complete method versus
76.37% without selective BN.

Xie et al. (CVPR 2020), *Adversarial Examples Improve Image Recognition*,
similarly separate normalization for shifted auxiliary examples, share
non-BN network weights, and discard auxiliary normalization at inference.
They also report a fine-grained variant with additional BN separation for
AutoAugment-transformed images.

Liu et al. (Neural Networks 2024), *Spectral Decomposition and Transformation
for Cross-domain Few-shot Learning*, explicitly use separated batch
normalization for original and frequency-domain augmented images because the
augmented distribution differs substantially from the original.

These works do not establish that the same method must work on Coffee17.
They motivate a targeted test of the normalization mismatch exposed by our
own failed MVCE experiment.

## Training objective

The input views and objective remain unchanged:

L_raw = CE(f(R0(x)), y)

L_aux = [CE(f(C0(x)),y) + CE(f(F0(x)),y) + CE(f(W0(x)),y)] / 3

L = 0.5 * L_raw + 0.5 * L_aux

## Selective BN contract

For R0, every BatchNorm layer behaves normally during training and updates
running mean/variance.

For each auxiliary view v in {C0,F0,W0}, BatchNorm uses the current
view-specific mini-batch mean/variance:

h_hat_v = gamma * (h_v - mu_B,v) / sqrt(var_B,v + eps) + beta

but auxiliary forwards do not update the persistent running mean/variance.

The affine parameters gamma and beta are shared across all views and receive
gradients from all views.

At inference, only raw RGB R0 is used, with the R0 running statistics.
Therefore there are no extra inference parameters or forward passes.

## Scope

Five folds, seed 42, 50 epochs, same MobileNetV3-Large + GAP, same optimizer,
same preprocessing transforms and same deterministic rotations. The old outer
test remains unopened for this method. The matched R0 controls are frozen from
the immediately preceding experiment artifact.
