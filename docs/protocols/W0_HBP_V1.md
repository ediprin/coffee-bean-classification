# Coffee17 W0-HBP V1 Protocol

## Purpose

Test whether the already-frozen W0 preprocessing becomes useful when paired directly with the already-established HBP representation.

Primary comparison:

`W0 -> MobileNetV3-Large -> HBP -> Linear17`

versus the previously completed fold-matched control:

`R0 -> MobileNetV3-Large -> HBP -> Linear17`

Only the W0-HBP arm is newly trained. The completed HBP-R0 control must be reused from the prior HBP-MVFD-SBN analysis package and must not be retrained.

## Duplication audit

The repository was audited before freezing this protocol.

Already completed:

- R0 + GAP preprocessing study;
- W0 + GAP preprocessing study;
- R0 + HBP;
- HBP + MVFD-SBN with R0 as deployment input and C0/F0/W0 as training-only auxiliary views;
- W0-RAS scalar auxiliary supervision;
- W0-AT spatial attention transfer.

Not found as a completed experiment:

- standalone W0 input feeding MobileNetV3-Large + HBP under the matched Coffee17 preprocessing-study folds.

The old config `M5w01_mobilenetv3_hbp_lmmd_w01.yaml` is an LMMD experiment with adaptation weight 0.1; `w01` there does not mean W0 preprocessing.

Therefore W0-HBP V1 is not a rerun of an existing completed arm.

## Scientific question

> Does wavelet/VisuShrink preprocessing provide additional value when the classifier uses multi-stage second-order HBP representation rather than ordinary raw-RGB HBP?

This experiment does not claim a causal interaction between wavelet preprocessing and HBP. It is a matched empirical test of whether W0 improves the established HBP control.

## Frozen W0 frontend

Use the exact preprocessing-study W0 definition:

- channel-wise RGB Haar DWT;
- four decomposition levels;
- finest-detail noise estimate `median(|d|)/0.6745`;
- universal VisuShrink threshold;
- soft threshold detail coefficients;
- inverse DWT reconstruction;
- epsilon `1e-8`.

No wavelet parameter search is authorized.

## Frozen HBP model

- MobileNetV3-Large;
- out indices `[1, 3, 4]`;
- 1x1 projection + BN + ReLU to 512 channels per selected stage;
- earlier maps pooled to deepest spatial size;
- pairwise products `(0,1)`, `(0,2)`, `(1,2)`;
- spatial mean;
- signed square-root + L2 normalization;
- concatenated 1536-D HBP embedding;
- linear 17-class classifier.

## Training protocol

Exactly match the completed HBP-R0 development control:

- seed 42;
- image size 224;
- batch size 32;
- preprocessing-study train rotations `[0,45,90,135,180,225,270]`;
- 50 epochs;
- AdamW;
- learning rate `3e-4`;
- weight decay `1e-4`;
- cross entropy with label smoothing `0.1`;
- cosine scheduler;
- no EMA;
- checkpoint selection by validation Macro-F1.

## Reference reuse

The HBP-R0 control comes from the completed `hbp-mvfd-sbn-analysis-package.zip`.

For each fold, W0-HBP must verify against that package:

- validation identity-label SHA256;
- validation count;
- seed;
- HBP architecture contract;
- HBP initial model-state SHA256;
- frozen HBP-R0 metrics.

If the prior package is absent or inconsistent, abort. Do not regenerate the HBP-R0 control inside this experiment.

## Screening gate

Compare W0-HBP against the completed fold-matched HBP-R0 control.

All conditions must hold:

1. mean Macro-F1 delta > 0;
2. Macro-F1 improves in at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0.

If the gate fails, stop W0-HBP V1. No post-hoc wavelet-level, threshold, HBP-stage, or resolution search is authorized on these reused development folds.

## Inference

W0 remains part of the deployed preprocessing path:

`RGB image -> W0 Haar+VisuShrink reconstruction -> MobileNetV3-Large -> HBP -> Linear17`

This is intentional: preprocessing has a substantive role in the final pipeline. Efficiency must therefore include W0 preprocessing cost if this arm passes screening.

## Evaluation boundary

This is post-primary exploratory method development on reused Coffee17 development folds.

The outer-test OOF is not accessed.
