# WR-HBP + Localized Top-k Self-Assessment Residual V1

Status: **frozen development protocol; training pending**.

## Why this experiment exists

This protocol is not chosen from aggregate accuracy alone. It is a direct
response to the persistent-error audit recorded in
`docs/analysis/PERSISTENT_ERROR_AUDIT_WR_HBP_V1.md`.

Strict WR-HBP makes 40 errors over 485 development fold-observations. Of those,
35 remain wrong under both WR-PDR and GCE-LS, and 31/35 retain the same wrong
class under all three methods. The repeated failures include:

- Withered -> Immature;
- Severe Insect Damage -> Slight Insect Damage;
- Cut -> Slight Insect Damage;
- Partial Sour -> Full Sour;
- Slight Insect Damage -> Fade;
- Slight Insect Damage -> Severe Insect Damage.

The physical-descriptor audit also shows heterogeneous failure mechanisms:
some persistent samples look wrong under global geometry, some under luminance,
and some are not explained by the frozen global descriptors at all.

Therefore V1 tests a different mechanism:

> preserve the successful global WR-HBP classifier exactly, freeze it, and
> learn only a small localized residual that reassesses the classes already
> considered plausible by the base model.

## Literature basis

The top-k reassessment concept is based on:

Tuong Do, Huy Tran, Erman Tjiputra, Quang D. Tran, Anh Nguyen,
*Fine-Grained Visual Classification using Self Assessment Classifier*,
IEEE CAI 2024; arXiv:2205.10529.

SAC explicitly targets ambiguity among the top-k predicted fine-grained
classes and combines image-region features with the candidate classes. Its
released implementation uses top-k = 5.

This Coffee17 experiment is **not an exact SAC reproduction**. It keeps the
specific principle required by our error audit while respecting the lightweight
single-forward WR-HBP setting:

- dynamic top-5 classes from the frozen WR-HBP logits;
- class-conditioned spatial attention over an existing MobileNetV3 mid-level
  feature map;
- a small reassessment head;
- additive residuals only on the current top-5 logits;
- no second image forward, crop branch, language/GRU label encoder, or dropping
  branch.

## Why top-5 is pre-registered

The strict WR-HBP persistent errors have the following true-class ranks:

- rank 2: 20/35;
- rank 3: 6/35;
- rank 4: 3/35;
- rank 5: 1/35;
- rank 6: 3/35;
- rank 7: 2/35.

Thus the ground-truth class is already inside WR-HBP's top-5 for **30/35
(85.7%)** persistent errors. This empirically supports reassessment rather than
globally replacing the classifier. Top-5 also matches the released SAC setup.
There is no k sweep.

## Architecture

Stage 1 is the frozen strict-deterministic WR-HBP recipe.

Stage 2 freezes every WR-HBP parameter and reuses its mid-level spatial feature
map `F_m`. Let the frozen base logits be `z` and their top-5 class indices be
`S`.

A 1x1 projection maps each spatial feature vector to 128 dimensions. Each of
the 17 classes has a learned 128-D embedding. For each candidate class
`c in S`, dot-product attention over spatial locations gives a
class-conditioned local representation.

The reassessment MLP receives:

- attended local visual feature;
- candidate-class embedding;
- frozen WR-HBP probability for that candidate.

It outputs one residual scalar per top-5 candidate. The final logits are:

```
z_final[c] = z_base[c] + r[c],  c in top5
z_final[c] = z_base[c],         c outside top5
```

The residual output layer is initialized to exactly zero, so before stage-2
training the candidate is functionally identical to the frozen WR-HBP
checkpoint.

## Frozen settings

Base WR-HBP:

- MobileNetV3-Large;
- HBP out_indices [1,3,4];
- projection 512;
- luminance L1 Haar LH/HL/HH + VisuShrink wavelet residual;
- 224 input;
- 50 epochs;
- AdamW 3e-4, weight decay 1e-4;
- label smoothing 0.1;
- cosine schedule;
- seed 42;
- strict deterministic CUDA path.

Self-assessment residual:

- base frozen, including BN statistics;
- top-k = 5;
- feature_index = 1 (mid-level map);
- projection/class embedding = 128;
- MLP hidden = 128;
- zero-initialized residual output;
- 20 epochs;
- AdamW 3e-4, weight decay 1e-4;
- same train-only augmentation loader;
- CE with the same label smoothing 0.1;
- checkpoint selected by validation Macro-F1;
- no hyperparameter sweep.

The 20-epoch head-only stage is pre-registered before seeing any SAR result and
is treated as an exploratory fail-fast budget, not an optimized schedule.

## Leakage boundary

The persistent-error audit determined the *mechanism* being tested but is not
fed into the loss or sampler. The reassessment head does not receive validation
confusion pairs, sample IDs, or hard-class weights. Its candidate set is
generated dynamically from the frozen base logits for each input.

The listed persistent confusion pairs are diagnostics only.

Outer test remains locked.

## Strict deterministic contract

The experiment inherits all deterministic corrections already established by
WR-PDR/GCE:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- deterministic PyTorch algorithms;
- cuDNN benchmark off and deterministic on;
- TF32 off;
- deterministic CUDA VisuShrink median path;
- deterministic exact-bin HBP alignment.

Before stage 2, zero-residual preflight must verify the candidate logits match
the frozen WR-HBP logits within 1e-7.

## Frozen gate

PASS requires all conditions:

1. mean Hard-F1 delta > 0;
2. Hard-F1 improves in at least 3/5 folds;
3. mean Macro-F1 delta >= 0;
4. mean Worst-F1 delta >= 0;
5. paired rescues >= paired damages;
6. total audited persistent-pair confusion does not increase.

Failure means STOP. No top-k, feature-stage, embedding-width, epoch, fusion, or
loss tuning is authorized on these development folds.

## Interpretation

A PASS would support the specific claim that localized reassessment of the
base model's ambiguous candidates adds useful information beyond global HBP and
wavelet detail.

A FAIL would be evidence that the remaining persistent Coffee17 errors are not
rescued by this class-conditional local reassessment mechanism under the frozen
dataset and model protocol.
