# Compact MPN-COV fail-fast protocol

## Research question

Does covariance pooling with matrix-square-root normalization provide a more
useful second-order representation than GAP and HBP for Coffee17 when the
backbone and training protocol are held fixed?

This is the final bounded architecture-refinement experiment. It is not a
claim that MPN-COV is new. The method follows Li et al., *Towards Faster
Training of Global Covariance Pooling Networks by Iterative Matrix Square Root
Normalization* (CVPR 2018), and its predecessor Li et al., *Is Second-order
Information Helpful for Large-scale Visual Recognition?* (ICCV 2017).

Primary references:

- paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html
- official implementation: https://github.com/jiangtaoxie/fast-MPN-COV
- MPN-COV predecessor: https://arxiv.org/abs/1904.06836

## Frozen models

All models use ImageNet-pretrained EfficientNetV2-B0, 224 x 224 inputs,
cross-entropy, identical data splits, augmentations, optimizer defaults, and
checkpoint selection.

| Code | Pooling head | Representation |
|---|---|---:|
| COV0 | GAP | first-order baseline |
| COV1 | HBP, projection 512 | prior second-order baseline |
| COV2 | compact iSQRT-COV, reduction 128 | 8,256 dimensions |

COV2 follows this fixed sequence:

1. deepest EfficientNetV2 feature map;
2. learned 1 x 1 Conv-BN-ReLU reduction to 128 channels;
3. spatially centered covariance with divisor `H*W`;
4. trace normalization;
5. five Newton-Schulz iterations;
6. square-root trace post-compensation;
7. upper-triangular vectorization;
8. dropout and linear classifier.

The official implementation commonly reduces to 256 channels. The fixed 128
channel choice is an explicitly named compact adaptation for this small
dataset and T4 compute budget, not a reproduction of the official network.
Matrix iterations run in float32 even when the surrounding model uses mixed
precision. `epsilon=1e-5` is only a numerical trace guard.

## Data and evaluation

Use the deduplicated Coffee17 grouped split (`965` clean identities), with
validation for screening. No test result may be inspected at this stage.
Primary metrics are Macro-F1, Hard-F1, and Worst-class F1. Accuracy is
secondary.

Stage 1 uses validation seed 42 only. COV2 passes only if all criteria hold
relative to COV0:

- Macro-F1 delta > 0;
- Hard-F1 delta > 0;
- Worst-class F1 delta >= -1 percentage point.

If Stage 1 fails, architecture refinement stops. No tuning of reduction
dimension, iteration count, epsilon, learning rate, or selected classes is
allowed after seeing the result.

If Stage 1 passes, confirm on validation seeds 123 and 2026 with the exact same
configuration. Test evaluation is permitted only after that confirmation is
decided and recorded.

## Scope of the claim

A positive result supports only that compact matrix-power-normalized
covariance pooling improves this controlled Coffee17 setting. A negative
result is also conclusive for this protocol: it does not disprove MPN-COV in
general, but it ends this architecture branch for the thesis.
