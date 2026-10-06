# Coffee17 Master Representation Screening V1

Status: **frozen for paper-grounded five-fold development screening**

Outer test: **not authorized**.

## Purpose

Train all currently defensible candidate mechanisms one-by-one under the same
Coffee17 development folds so method selection is based on matched empirical
evidence rather than intuition.

The master runner trains the common GAP and HBP anchors once per fold and then
reuses them for paired comparisons. This avoids retraining the same anchors for
each family.

## Arms

| ID | Method | Primary anchor |
|---|---|---|
| B0 | MobileNetV3-Large + GAP | B0 |
| B1 | MobileNetV3-Large + HBP | B1 |
| B2 | WR-HBP: luminance Haar/VisuShrink shallow residual + HBP | B1 |
| R1 | MFR(deep) + GAP | B0 |
| R2 | MFR(deep) + HBP | B1 |
| R3 | LRBP-I/co-decomposition-style low-rank bilinear scorer | B1 |
| R4 | MFSwin-style adaptive multi-stage fusion | B1 |
| R5 | HBP + 0.7 CE / 0.3 Focal fusion objective | B1 |
| W1 | WCANet-style wavelet channel attention + GAP | B0 |
| W2 | WCANet-style wavelet channel attention + HBP | B1 |
| W3 | FdaNet-style adaptive subband weighting + GAP | B0 |
| W4 | FdaNet-style adaptive subband weighting + HBP | B1 |
| W5 | FcaNet-LF K=2 DCT attention + GAP | B0 |
| W6 | FcaNet-LF K=2 DCT attention + HBP | B1 |
| W7 | lightweight DWT spatial-frequency dual-domain fusion | B0 |
| F1 | one late Fast Fourier Convolution local/global block + GAP | B0 |
| F2 | one late GFNet-style learnable global filter + GAP | B0 |

## Why there is no W8/WaveCNet arm

WaveCNet's defining intervention replaces backbone downsampling operations with
DWT. A faithful MobileNetV3 version requires modifying several stride/depthwise
blocks. A simple DWT before GAP would not reproduce WaveCNet and would be a
misleading label. It is therefore retained as literature evidence, not forced
into this matched late-representation screen.

## Source-method boundaries

Some arms are direct mechanism transfers; others are explicitly adaptations:

- B2 follows the already frozen WR-HBP protocol.
- W1/W3/W5 transfer the source attention mechanism to the deepest MobileNet
  feature.
- W2/W4/W6 apply the same mechanism only to HBP's deepest selected feature;
  this is a controlled Coffee17 adaptation.
- R1/R2 transfer MFR with lambda=8, alpha=0.8, rho=0.2.
- R3 follows the LRBP-I Frobenius scoring structure with m=100 and r=8, but
  does not perform Coffee17 PCA initialization; the shared projection is
  learned end-to-end.
- R4 transfers adaptive multi-stage weighting to MobileNet stages rather than
  copying the full Swin backbone.
- R5 transfers the CE/Focal objective idea while keeping the established
  Coffee17 label smoothing in its CE term.
- W7 tests DWTFormer's central spatial/frequency complementarity principle with
  a lightweight CNN fusion block, not the full transformer.
- F1 inserts one FFC-style late block rather than replacing every MobileNet
  convolution.
- F2 inserts one GFNet-style global filter on the fixed 7x7 late feature rather
  than replacing the full backbone.

No result from an adapted arm may be described as a literal reproduction of the
source paper.

## Frozen protocol

All arms share:

- Coffee17 original: 979 images / 17 classes;
- existing clean preprocessing-study five-fold reconstruction;
- seed 42 only;
- 224x224 input;
- batch size 32;
- rotations [0,45,90,135,180,225,270];
- ImageNet-pretrained MobileNetV3-Large;
- 50 epochs;
- AdamW;
- lr 3e-4;
- weight decay 1e-4;
- cosine schedule;
- no EMA;
- checkpoint selection by validation Macro-F1;
- no outer-test access.

Default classification objective is CE with label smoothing 0.1. R3 and R5 are
method-specific objective exceptions documented above.

## Initialization matching

Where candidate and anchor share encoder/pool/classifier structure, the runner
verifies tensor-for-tensor equality at initialization.

- W1/W3/W5 and R1 match B0 shared cores.
- W2/W4/W6 and R2 match B1 shared cores.
- R5 is exactly B1 at initialization.
- B2 uses the WR-HBP shared-core and zero-gate preflight.
- R3/R4/W7/F1/F2 verify equal pretrained encoder initialization where their
  head differs structurally.

## Full-run policy

All five folds are completed for every requested arm. A negative first fold
does not stop a method.

Per candidate report:

- Accuracy;
- Balanced Accuracy;
- Macro-F1;
- Hard-F1;
- Worst-F1;
- paired deltas to its anchor;
- positive Macro-F1 fold count;
- rescue/damage/net-correct;
- parameter count;
- WR-HBP gate where applicable.

Development promotion signal remains:

1. mean Macro-F1 delta > 0;
2. Macro-F1 positive in >=3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0.

For efficiency-oriented R3, a small score loss may still be retained for a
separate Pareto analysis; it does not automatically become the accuracy winner.

## Literature basis

- Yu et al. (2018), HBP, ECCV.
- Kong & Fowlkes (2017), LRBP, CVPR.
- Huang et al. (2022), SNet/MFR.
- Bi et al. (2022), MFSwin.
- Jiao et al. (2025), Swin-HSSAM.
- Zhang et al. (2025), WCANet.
- Zhang et al. (2024), FdaNet/HaNet.
- Qin et al. (2021), FcaNet, ICCV.
- Xiang et al. (2025), DWTFormer.
- Li et al. (2020), WaveCNet, CVPR.
- Chi et al. (2020), Fast Fourier Convolution, NeurIPS.
- Rao et al. (2021), GFNet, NeurIPS.
