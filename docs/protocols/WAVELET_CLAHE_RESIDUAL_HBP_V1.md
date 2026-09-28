# WRC-HBP V1 — WR-HBP + CLAHE Contrast Residual

Status: **frozen for 5-fold seed42 development screening**

Branch: `codex/wavelet-clahe-residual-hbp-v1`

## 1. Research question

Can the completed WR-HBP model be improved by adding one orthogonal
low-frequency/color-contrast cue derived from the already-audited C0 CLAHE
representation, while preserving the successful RGB + wavelet pathway?

Matched comparison:

- control: `WR-HBP`
- candidate: `WRC-HBP = WR-HBP + zero-gated CLAHE contrast residual`

## 2. Evidence leading to this experiment

Completed WR-HBP screening showed:

- Macro-F1: 91.02% -> 91.77% (+0.75 pp);
- Macro-F1 positive in 5/5 folds;
- Hard-F1: +0.11 pp;
- Worst-F1: +1.42 pp;
- development gate PASS.

Post-hoc paired-probability analysis showed that the largest remaining
weakness was concentrated around the sour boundary, especially Partial Sour
versus Full Sour.

Earlier C0 preprocessing analysis showed positive class-level effects for
Partial Sour and Full Sour. This suggests an orthogonal contrast/color cue may
complement the successful wavelet-detail residual.

The repository duplication audit found no completed experiment that adds a
CLAHE-derived contrast residual on top of WR-HBP.

## 3. Candidate architecture

The WR-HBP path is unchanged:

`RGB -> MobileNetV3-Large -> (F_s, F_m, F_d)`

`Y -> Haar L1 -> [LH,HL,HH] -> VisuShrink -> P_w -> shallow residual`

The added C0 path uses the exact frozen C0 frontend:

`RGB -> CIELAB -> CLAHE(L, clip=2.0, grid=8x8) -> RGB_C0`

The contrast signal is the signed Rec.709 luminance delta:

`Delta_L = Y(RGB_C0) - Y(RGB)`

A tiny contrast projection uses:

`AvgPool4 -> Conv3x3(1->8) -> BN -> SiLU -> Conv1x1(8->C_s) -> BN`

The final shallow feature is:

`F_s' = F_s + tanh(alpha_w) P_w(D_L1) + tanh(alpha_c) P_c(Delta_L)`

where:

- `alpha_w` is the ordinary WR-HBP wavelet gate;
- `alpha_c` is a new learnable scalar initialized to exactly zero.

HBP remains unchanged:

`z = HBP(F_s', F_m, F_d)`

## 4. Initialization integrity

WRC-HBP calls the WR-HBP constructor first. The contrast-only branch is created
afterwards and the CPU RNG state is restored.

Before training the runner must verify:

- tensor-for-tensor equality of every WR-HBP shared parameter/buffer;
- identical shared WR-HBP SHA-256 fingerprint;
- `tanh(alpha_c)=0`;
- identical initial WR-HBP and WRC-HBP logits to max absolute difference
  <= 1e-7 on a deterministic tensor.

Therefore WRC-HBP starts functionally from the same WR-HBP control.

## 5. Frozen training protocol

Both arms:

- Coffee17 preprocessing-study development folds;
- 5 folds;
- seed 42;
- input 224;
- batch 32;
- deterministic rotations [0,45,90,135,180,225,270];
- MobileNetV3-Large;
- HBP out indices [1,3,4];
- projection dim 512;
- linear 17-class classifier;
- AdamW;
- lr 3e-4;
- weight decay 1e-4;
- CE + label smoothing 0.1;
- cosine scheduler;
- 50 epochs;
- no EMA;
- checkpoint selection by validation Macro-F1.

Only WRC-HBP has the new contrast branch.

## 6. Metrics

Primary:

- Macro-F1.

Safety:

- Hard-F1;
- Worst-F1.

Also report:

- Accuracy;
- Balanced Accuracy;
- positive-delta fold counts;
- rescue/damage;
- per-class F1 deltas;
- hard-group deltas;
- wavelet gate;
- contrast gate;
- parameter overhead.

## 7. Frozen screening gate

WRC-HBP passes only if all are true:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 improves in at least 3/5 folds;
3. mean paired Hard-F1 delta >= 0;
4. mean paired Worst-F1 delta >= 0.

If the gate fails:

> stop WRC-HBP. Do not tune CLAHE clip limit, grid size, contrast
> representation, branch width, injection stage, gate initialization, wavelet
> branch, HBP configuration, loss, or resolution on these reused folds.

## 8. Evaluation boundary

This is post-primary exploratory development on reused Coffee17 development
folds.

Outer test is not accessed.
