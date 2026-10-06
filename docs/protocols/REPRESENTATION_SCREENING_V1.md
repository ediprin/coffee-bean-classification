# Coffee17 Representation Screening V1

Status: **frozen Phase-B development screening**

Outer test: **must not be accessed**.

This phase complements the frequency-attention screen with paper-grounded
representation, recalibration, loss, wavelet dual-domain and Fourier operators.

## Arms

| ID | Architecture / objective | Primary anchor |
|---|---|---|
| B0 | MobileNetV3-Large -> GAP -> Linear17 | B0 |
| B1 | MobileNetV3-Large -> HBP -> Linear17 | B1 |
| R1 | MobileNetV3-Large -> MFR(deep) -> GAP | B0 |
| R2 | MobileNetV3-Large -> MFR(deep) -> HBP | B1 |
| R3 | MobileNetV3-Large -> LRBP-I style scorer | B1 |
| R4 | MobileNetV3 stages -> adaptive weighted fusion -> GAP | B1 |
| R5 | MobileNetV3-Large -> HBP, CE + Focal fusion loss | B1 |
| W7 | MobileNetV3 deep feature -> spatial + Haar-DWT dual-domain fusion | B0 |
| F1 | MobileNetV3 deep feature -> one late FFC local/global block -> GAP | B0 |
| F2 | MobileNetV3 deep feature -> GFNet-style learnable global filter -> GAP | B0 |

## R1/R2 — Mixed Feature Recalibration

Huang et al. (2022) define average- and max-pooling recalibration paths with:

- lambda = 8;
- alpha = 0.8;
- rho = 0.2.

The source paper's printed W5 matrix shape is dimensionally inconsistent with
its own concatenation equation and Fig. 6. V1 follows the computational graph
shown in Fig. 6: the two average-path reduced descriptors are concatenated and
mapped from 2N/lambda back to N.

Only the deepest feature is recalibrated in V1. R1/R2 therefore isolate transfer
of the MFR mechanism before any stage-location search.

## R3 — LRBP-I adaptation

The scorer follows Kong & Fowlkes' low-rank Frobenius formulation:

- shared feature projection P: C -> m;
- m = 100;
- total classifier rank r = 8;
- each class has positive and negative rank-r/2 factors;
- class score is the positive squared Frobenius response minus the negative
  squared Frobenius response.

A one-vs-all hinge objective plus the paper-motivated Frobenius regularizer is
used. The source paper initializes P using activation-statistics/PCA. V1 does
not use development-data PCA; P is learned end-to-end from standard neural
initialization so no extra Coffee17 data-dependent selection is introduced.
Therefore R3 is an **LRBP-I architecture adaptation**, not a literal
reproduction of the original VGG/PCA training recipe.

## R4 — Adaptive multi-stage fusion

This is a MobileNet adaptation of the mechanism demonstrated by MFSwin:

1. take three backbone stages;
2. project them to a common dimension (384);
3. align them to the deepest spatial grid;
4. build a concatenated global descriptor;
5. softmax three learned stage weights;
6. weighted-sum the projected feature maps;
7. refine and GAP classify.

R4 is deliberately an alternative to HBP rather than MFSwin + HBP stacking.

## R5 — Fusion loss

The HBP architecture is exactly B1 at initialization. Only the objective changes:

L = 0.7 * CE(label_smoothing=0.1) + 0.3 * Focal(gamma=2).

This is a Coffee17 protocol adaptation of the CE/Focal fusion idea reported by
Swin-HSSAM. It is treated as a low-cost objective screen, not a main
architecture contribution.

## W7 — lightweight DWT dual-domain arm

DWTFormer demonstrates that spatial and frequency representations can be
complementary, but the full model is too heavy for the lightweight Coffee17
scope. V1 therefore tests the mechanism only:

- spatial 1x1 projection of the deepest feature;
- Haar DWT into LL/LH/HL/HH;
- concatenate the four bands and project them;
- upsample the frequency feature to the spatial grid;
- concatenate spatial and frequency features;
- 1x1 fusion -> GAP -> classifier.

This is explicitly a **DWTFormer-inspired lightweight adaptation**, not a
reproduction of MPVT/MFA/DFF-DCA.

## F1 — late FFC block

A single late block follows FFC's central local/global design:

- split feature channels 50:50;
- local path uses spatial convolution;
- global path uses FFT -> spectral 1x1 conv -> IFFT;
- cross-path 1x1 projections exchange information;
- concatenate and classify by GAP.

This avoids replacing every MobileNet convolution during the first screen.

## F2 — late global filter

The deepest 7x7 MobileNet feature receives a GFNet-style learnable complex
frequency filter:

X = FFT(x)
X' = K .* X
x' = IFFT(X')

The filtered feature is added residually to the original late feature and then
GAP-classified. This is a CNN late-block adaptation of GFNet, not the complete
GFNet transformer-style architecture.

## Why WaveCNet is not an arm here

WaveCNet's defining intervention is replacement of backbone downsampling
operations by DWT. A faithful MobileNetV3 adaptation requires surgery to
multiple stride/depthwise blocks and would no longer be a one-factor late
representation screen. A fake "DWT before GAP" arm would not test WaveCNet.

Therefore WaveCNet remains a literature/control reference and is intentionally
not mislabeled as a directly reproduced arm in V1.

## Frozen common protocol

- Coffee17 original: 979 images / 17 classes;
- existing clean preprocessing-study 5-fold reconstruction;
- seed 42;
- input 224x224;
- batch 32;
- rotations [0,45,90,135,180,225,270];
- ImageNet-pretrained MobileNetV3-Large;
- 50 epochs;
- AdamW, lr 3e-4, weight decay 1e-4;
- cosine schedule;
- no EMA;
- checkpoint selection by validation Macro-F1;
- no outer-test access.

Except for R3 and R5, all candidates use the same CE + label smoothing 0.1 as
the established development protocol.

## Reporting

For every arm:

- Accuracy;
- Balanced Accuracy;
- Macro-F1;
- Hard-F1;
- Worst-F1;
- positive Macro-F1 fold count;
- paired rescue/damage counts;
- parameters.

All five folds must complete. No candidate is stopped from a single bad fold.

## Literature basis

- Huang et al. (2022), *Deep learning based soybean seed classification*.
- Bi et al. (2022), improved Swin / MFSwin maize seed recognition.
- Kong & Fowlkes (2017), *Low-Rank Bilinear Pooling for Fine-Grained
  Classification*, CVPR.
- Jiao et al. (2025), *Swin-HSSAM: A green coffee bean grading method by Swin
  transformer*.
- Xiang et al. (2025), *DWTFormer: a frequency-spatial features fusion model
  for tomato leaf disease identification*.
- Chi et al. (2020), *Fast Fourier Convolution*, NeurIPS.
- Rao et al. (2021), *Global Filter Networks for Image Classification*,
  NeurIPS.
