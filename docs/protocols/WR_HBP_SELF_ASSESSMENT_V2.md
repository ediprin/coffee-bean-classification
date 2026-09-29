# WR-HBP + Localized Top-k Self-Assessment Residual V2

Status: **frozen implementation-correction protocol; training pending**.

## Reason for V2

SAR V1 completed five strict-deterministic folds but its intended localization
mechanism collapsed to almost perfectly uniform spatial attention.

The root cause was identified directly from the V1 implementation and saved
diagnostics: spatial vectors and class vectors were both L2-normalized, then
their cosine similarity was divided by sqrt(d). At d=128 this shrank already
bounded cosine scores by another factor of 11.31.

V1 validation attention entropy was approximately 5.27809 nats, essentially
identical to ln(196)=5.278114659 for a uniform 14x14 map.

V2 is therefore a new protocol that corrects this implementation defect. It is
not a hyperparameter rescue sweep.

## Frozen correction

Everything from V1 is unchanged except the attention-logit scale.

For normalized spatial feature s_l and normalized class vector e_c:

```
a(c,l) = sqrt(d) * <s_l, e_c>
A(c,:) = softmax_l(a(c,l))
```

with d=128.

The sqrt(d) multiplier is fixed analytically and is not selected from
validation. No temperature sweep is allowed.

## Model

- strict WR-HBP base;
- MobileNetV3-Large + HBP + luminance L1 wavelet residual;
- base model frozen during reassessment;
- top-k = 5;
- mid-level feature index = 1;
- spatial projection = 128;
- class embedding = 128;
- hidden dimension = 128;
- additive residual only on current top-5 logits;
- zero-initialized residual output;
- 20 head-only epochs;
- AdamW lr 3e-4, weight decay 1e-4;
- CE label smoothing 0.1;
- same five Coffee17 development folds and seed 42;
- outer test locked.

## Required diagnostics

Per fold save:

- base and reassessed predictions;
- true-class rank in base top-5;
- base/final true probability and margin;
- residual magnitude;
- attention entropy;
- top-5 candidate classes and residuals;
- rescue/damage counts;
- audited persistent-pair confusion changes.

A regression test must ensure the normalized-cosine attention scale is
sqrt(d), not 1/sqrt(d).

## Frozen gate

PASS requires all:

1. mean Hard-F1 delta > 0;
2. Hard-F1 positive in >=3/5 folds;
3. mean Macro-F1 delta >= 0;
4. mean Worst-F1 delta >= 0;
5. rescues >= damages;
6. audited persistent-pair confusion total does not increase.

Failure means STOP. No additional temperature, top-k, feature-stage, loss,
epoch, or width tuning is permitted on these reused development folds.
