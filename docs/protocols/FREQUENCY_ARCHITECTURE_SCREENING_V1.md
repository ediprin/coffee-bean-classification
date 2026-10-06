# Coffee17 Frequency Architecture Screening V1

Status: **frozen Phase-A development screening**

Base scientific lineage: `codex/wr-hbp-final-confirmation-v1` at
`2248a6d4b1881cae59b4ecb8028ada4611e3230b`.

Outer test: **must not be accessed**.

## Research question

When frequency information is integrated *inside* the feature architecture rather
than used as input preprocessing, which paper-grounded mechanism transfers best
to Coffee17 under the same MobileNetV3-Large training protocol?

This phase deliberately compares three mechanisms before any module stacking:

1. **WCA** — Haar-DWT-derived channel attention (Zhang et al., KBS 2025).
2. **Fda** — sample-dependent LL/LH/HL/HH weighting followed by Haar IWT
   (Zhang et al., CEA 2024).
3. **Fca-LF2** — two-component low-frequency DCT channel attention
   (Qin et al., ICCV 2021).

## Arms

| ID | Architecture | Matched anchor |
|---|---|---|
| B0 | MobileNetV3-Large -> GAP -> Linear17 | B0 |
| B1 | MobileNetV3-Large -> HBP -> Linear17 | B1 |
| W1 | MobileNetV3-Large -> WCA(deep) -> GAP -> Linear17 | B0 |
| W2 | MobileNetV3-Large -> WCA(deep) -> HBP -> Linear17 | B1 |
| W3 | MobileNetV3-Large -> Fda(deep) -> GAP -> Linear17 | B0 |
| W4 | MobileNetV3-Large -> Fda(deep) -> HBP -> Linear17 | B1 |
| W5 | MobileNetV3-Large -> Fca-LF2(deep) -> GAP -> Linear17 | B0 |
| W6 | MobileNetV3-Large -> Fca-LF2(deep) -> HBP -> Linear17 | B1 |

For HBP variants, only the deepest selected feature is modified in V1. The
shallow and middle features are unchanged. This is an explicit Coffee17
adaptation used to isolate one factor; it is not claimed to be the exact
placement used in the source papers.

## WCA implementation contract

For feature map `X`:

`(LL,LH,HL,HH) = HaarDWT(X)`

`Y_i = GAP(F_i)`

`Y = sum_i Y_i`

`a = sigmoid(FC(Y))`

`X' = a * X`

The four wavelet bands construct the descriptor; the original spatial feature
map is preserved and only channel-reweighted.

## Fda implementation contract

`(LL,LH,HL,HH) = HaarDWT(X)`

Each band is compressed by a learned 1x1 projection and GAP. The four scalars
are passed through a two-layer FC mapper to produce four sample-dependent
sigmoid weights. Each subband is weighted independently and Haar-IWT restores
the spatial feature map.

Odd feature grids are replicate-padded by at most one row/column only inside
the DWT/IWT pair and cropped back to the original size after reconstruction.

## Fca implementation contract

V1 uses **FcaNet-LF with K=2**, because the source paper reports K=2 as the
best low-frequency setting and it avoids a Coffee17-specific frequency search
on already reused development folds.

The channel tensor is split into two groups and compressed with DCT (0,0) and
(0,1), followed by the standard channel-attention FC mapping.

## Frozen training protocol

All arms use:

- Coffee17 original, 979 images / 17 classes;
- the existing clean preprocessing-study 5-fold reconstruction;
- seed 42;
- 224x224 input;
- batch size 32;
- rotations [0,45,90,135,180,225,270];
- MobileNetV3-Large ImageNet initialization;
- HBP out indices [1,3,4] and projection dim 512 where applicable;
- linear 17-class classifier;
- 50 epochs;
- AdamW;
- learning rate 3e-4;
- weight decay 1e-4;
- cross-entropy with label smoothing 0.1;
- cosine scheduler;
- no EMA;
- checkpoint selection by validation Macro-F1.

No per-method hyperparameter tuning is authorized in this phase.

## Initialization integrity

For W1/W3/W5, the encoder/pool/classifier shared core must initialize exactly
as B0. For W2/W4/W6, it must initialize exactly as B1.

Candidate-only modules are constructed after the shared core and the CPU RNG
state is restored afterwards. The runner verifies tensor equality of the
shared core before training.

## Metrics

Primary:

- Macro-F1.

Safety:

- Hard-F1;
- Worst-F1.

Additional:

- Accuracy;
- Balanced Accuracy;
- positive-delta fold count;
- paired rescue/damage/net-correct counts;
- parameter count.

## Decision policy

All five folds are completed for every arm; no early stopping of a candidate
based on one negative fold.

Phase-A promotion signal:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 positive on at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0.

These are development-screening criteria only, not an outer-confirmatory claim.

## Literature basis

- Zhang et al. (2025), *Fine-grained recognition of citrus varieties via
  wavelet channel attention network*, Knowledge-Based Systems.
- Zhang et al. (2024), *Hybrid attention network for citrus disease
  identification*, Computers and Electronics in Agriculture.
- Qin et al. (2021), *FcaNet: Frequency Channel Attention Networks*, ICCV.
- Yu et al. (2018), *Hierarchical Bilinear Pooling for Fine-Grained Visual
  Recognition*, ECCV.

WaveCNet, DWTFormer, FFC and GFNet remain literature-backed Phase-B candidates
but are intentionally not mixed into this first architecture screen until WCA,
Fda and Fca have been measured under the matched Coffee17 protocol.
