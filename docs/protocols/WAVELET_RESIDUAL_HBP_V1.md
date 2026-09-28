# WR-HBP V1 — RGB + L1 Wavelet Residual + HBP

Status: **frozen for 5-fold seed42 development screening**

Branch: `codex/wavelet-residual-hbp-v1`

## 1. Research question

Can explicit wavelet-detail information improve the established RGB-HBP model
when RGB is preserved as the main representation and the wavelet signal is
introduced only as a small residual at the shallow feature stage?

Matched comparison:

- control: `R0 -> MobileNetV3-Large -> HBP -> Linear17`
- candidate: `R0 RGB main path + L1 wavelet-detail residual -> HBP -> Linear17`

The candidate is called **WR-HBP**.

## 2. Why this experiment follows the completed evidence

Completed Coffee17 evidence already showed:

1. W0 as a standalone GAP input had only a small positive Macro-F1 point
   estimate and did not improve aggregate Hard-F1.
2. R0+W0 late probability fusion improved Macro-F1 relative to R0, showing
   useful complementarity.
3. Direct W0-HBP failed: Macro-F1 -1.23 pp, Hard-F1 -2.13 pp.
4. W0-RAS scalar transfer failed.
5. W0-AT spatial-attention transfer failed.
6. ML-MVFD direct intermediate tensor matching failed strongly.
7. HBP+MVFD failed on Macro/Hard.
8. The W0 cue audit found a coherent association around VisuShrink-retained
   L1 high-frequency detail; RGB-channel retained ratios were highly redundant.

The unresolved combination is therefore:

> retain the full RGB path and expose the wavelet detail explicitly as an
> additional inference-time signal, without replacing RGB and without forcing
> the RGB representation to imitate a transformed teacher.

Repository duplicate audit found no completed wavelet side-branch, wavelet
feature injection, or direct RGB+wavelet residual HBP experiment.

## 3. Candidate architecture

Main path:

`x_RGB -> ImageNet normalization -> MobileNetV3-Large -> (F_s, F_m, F_d)`

Wavelet path is computed from the same augmented RGB image before ImageNet
normalization:

`Y = 0.2126 R + 0.7152 G + 0.0722 B`

One-level Haar DWT:

`D_L1 = [LH, HL, HH]`

VisuShrink threshold:

`sigma = median(|D_L1|) / 0.6745`

`tau = sigma * sqrt(2 log N)`

Soft threshold:

`D_ret = sign(D_L1) * max(|D_L1| - tau, 0)`

Tiny projection branch:

`P(D_ret) = BN(Conv1x1(SiLU(BN(Conv3x3_stride2(D_ret))))))`

The branch uses 16 hidden channels.

Residual injection:

`F_s' = F_s + tanh(alpha) * P(D_ret)`

where `alpha` is a learnable scalar initialized to exactly zero.

HBP then receives:

`HBP(F_s', F_m, F_d)`

The HBP definition, projection dimension, classifier, loss, optimizer, and
training schedule remain unchanged.

## 4. Why luminance-only wavelet detail

The main MobileNet path retains all RGB color information.

The wavelet branch is intentionally restricted to luminance texture because:

- the Coffee17 W0 cue audit found the strongest coherent signal in retained
  high-frequency detail;
- retained-ratio descriptors across RGB channels were highly redundant;
- the purpose of this branch is to add explicit texture evidence, not duplicate
  the RGB representation.

This is a fixed design choice. No RGB-vs-luminance branch search is authorized
on the reused development folds.

## 5. Initialization integrity

The WR-HBP model constructs the shared RGB-HBP core in the same order as the
R0-HBP control:

1. encoder;
2. HBP;
3. dropout;
4. linear classifier.

The candidate-only wavelet branch is constructed afterwards, and the CPU RNG
state is restored after its initialization.

Before training, the runner must verify:

- tensor-for-tensor equality of all shared encoder/HBP/classifier parameters;
- identical shared-core SHA-256 fingerprint;
- `tanh(alpha) = 0`;
- control and candidate logits agree to max absolute error <= 1e-7 on a
  deterministic preflight tensor.

Thus WR-HBP begins functionally from the same RGB-HBP core.

## 6. Frozen training protocol

Both arms use:

- Coffee17 clean preprocessing-study folds;
- 5 folds;
- seed 42;
- image size 224;
- batch size 32;
- rotations [0,45,90,135,180,225,270];
- MobileNetV3-Large ImageNet initialization;
- HBP out indices [1,3,4];
- HBP projection dim 512;
- linear 17-class classifier;
- 50 epochs;
- AdamW;
- lr 3e-4;
- weight decay 1e-4;
- CE with label smoothing 0.1;
- cosine scheduler;
- no EMA;
- checkpoint selection by validation Macro-F1.

Only the candidate contains the fixed wavelet residual branch.

## 7. Metrics

Primary:

- Macro-F1.

Safety metrics:

- Hard-F1;
- Worst-F1.

Also report:

- Accuracy;
- Balanced Accuracy;
- positive-delta fold counts;
- rescue/damage counts;
- per-class F1 deltas;
- learned wavelet gate at the selected checkpoint;
- candidate parameter overhead.

## 8. Frozen screening gate

WR-HBP passes only if all conditions hold:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 improves in at least 3/5 folds;
3. mean paired Hard-F1 delta >= 0;
4. mean paired Worst-F1 delta >= 0.

If the gate fails:

> stop WR-HBP. Do not tune hidden channels, wavelet bands, wavelet level,
> thresholding rule, injection stage, gate initialization, HBP stages, loss,
> or image resolution on these reused development folds.

## 9. Evaluation boundary

This experiment is post-primary exploratory development.

Outer test is not accessed.

## 10. Literature basis

The design is supported at the mechanism level by prior work showing that
wavelet decompositions can expose spatial-frequency/texture information for
image classification and by bilinear pooling literature showing the value of
second-order interactions for fine-grained recognition.

The Coffee17-specific choice of L1 retained detail is driven by the project's
completed cue audit, not by a claim that a particular wavelet branch is already
established as superior for Coffee17.

Relevant project/library references include:

- Kong & Fowlkes (2017), *Low-Rank Bilinear Pooling for Fine-Grained
  Classification*.
- Darmawan et al. (2026), *Preliminary Research of Automated Coffee Bean
  Roasting Level Classification for Quality Control in Manufacturing Using
  CNN and Wavelet*.
- Coffee17 preprocessing/HBP decision records in this repository.
