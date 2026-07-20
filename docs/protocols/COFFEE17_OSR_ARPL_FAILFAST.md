# Coffee17 OSR ARPLoss fail-fast

## Scope

This experiment tests whether a training-time open-set objective improves the
frozen EfficientNetV2-B0 + GAP + CE + MSP baseline. It evaluates only the
`near` and `medium` Coffee17 OSR v1 tiers at seed 123. Unknown test samples are
never used for training, validation, checkpoint selection, or threshold fitting.

## Method identity

The candidate implements the official repository's `ARPLoss` option without
the optional confusing-sample (`--cs`) GAN. It must therefore be reported as:

> ARPLoss without confusing-sample adversarial enhancement (ARPL-no-CS)

It must not be reported as ARPL+CS. The implemented components are:

1. one learnable reciprocal point per known class;
2. class logits equal to normalized squared-L2 distance minus dot product;
3. cross-entropy on reciprocal-point logits;
4. learnable-radius margin-ranking regularization with weight 0.1 and margin 1;
5. maximum ARPL logit as the official knownness score.

Primary sources:

- Chen et al., *Learning Open Set Network with Discriminative Reciprocal
  Points*, ECCV 2020:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/591_ECCV_2020_paper.php
- Chen et al., *Adversarial Reciprocal Points Learning for Open Set
  Recognition*, TPAMI 2021, DOI 10.1109/TPAMI.2021.3106743.
- Official PyTorch implementation: https://github.com/gary23ai/ARPL

## Frozen comparison

- Baseline: EfficientNetV2-B0 + GAP + CE, MSP score.
- Candidate: EfficientNetV2-B0 + GAP + ARPLoss-no-CS, maximum ARPL logit.
- Tiers: near and medium.
- Screening seed: 123.
- Checkpoint selection: known-validation Macro-F1 only.
- Operational threshold: known-validation only, target known acceptance 95%.

A tier passes only when all conditions hold:

- AUROC gain is at least 2 percentage points;
- OSCR gain is positive;
- known Macro-F1 loss is no greater than 1 percentage point.

Only passing tiers may be confirmed on seeds 42 and 2026. If neither tier
passes, ARPL-no-CS stops and ARPL+CS is not inferred to have failed because it
is a different, generative method.
