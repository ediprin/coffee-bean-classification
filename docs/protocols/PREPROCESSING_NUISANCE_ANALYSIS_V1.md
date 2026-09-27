# Coffee17 Preprocessing Nuisance Analysis v1

## Question

This analysis tests a narrower claim than "camera invariance".

The question is:

> Do C0/F0/W0 preserve Coffee17 defect-discriminative representations under
> controlled acquisition-like perturbations better than raw RGB (R0)?

The analysis does **not** claim that Fourier or wavelet processing removes all
camera, illumination, shadow, compression, or sensor effects.

## Why this is needed

The deterministic preprocessing study established that:
- no fixed transformed view consistently replaces R0;
- transformed views provide class-dependent complementary information.

MVFD-SBN then showed that transformed-view feature transfer improves aggregate
R0 validation performance.

What remains untested is whether any transformed representation is also more
stable to acquisition nuisance while retaining class separability.

## Design

Use the frozen MVFD-SBN model and its exact shared MobileNetV3-Large backbone.

For each validation image and each view P in {R0,C0,F0,W0}:

1. compute the clean view embedding z_P(x);
2. apply a frozen nuisance T before the preprocessing frontend;
3. compute z_P(T(x));
4. compare clean and perturbed embeddings;
5. evaluate the view's existing classifier without any retraining.

Because all four views use the same frozen shared backbone, representation
differences are not confounded by using different backbone architectures.

## Synthetic nuisance proxies

Frozen transforms:

- brightness_low: multiply RGB by 0.80
- brightness_high: multiply RGB by 1.20
- gamma_dark: gamma 1.25
- gamma_bright: gamma 0.80
- white_balance_warm: channel gains [1.10, 1.00, 0.90]
- white_balance_cool: channel gains [0.90, 1.00, 1.10]
- shadow_left: smooth horizontal multiplicative mask 0.65 -> 1.00
- shadow_right: smooth horizontal multiplicative mask 1.00 -> 0.65
- gaussian_blur: 5x5, sigma 1.2
- gaussian_noise: sigma 0.03, deterministic seed
- jpeg_q50: JPEG quality 50

All outputs are clipped to [0,1].

These are acquisition-like proxies, not literal physical camera simulation.

## Metrics

### 1. Feature stability

Cosine similarity:

S_cos(P,T) = mean_i cos(z_P(x_i), z_P(T(x_i)))

Higher is more stable.

Normalized L2 shift:

D_L2(P,T) = mean_i ||norm(z_P(x_i)) - norm(z_P(T(x_i)))||_2

Lower is more stable.

### 2. Class separability

On L2-normalized embeddings:

Fisher(P) = between-class scatter / within-class scatter

For each nuisance report:

Retention_sep(P,T) = Fisher(P,T) / Fisher(P,clean)

Higher retention means the nuisance damages class structure less.

### 3. Classification retention

Each view uses its already-trained classifier:
- R0 -> primary classifier
- C0/F0/W0 -> their frozen training-only auxiliary classifiers

Report clean and nuisance Macro-F1, Balanced Accuracy, and their absolute drop.

This is a diagnostic of the transformed representations, not a deployment
recommendation for the auxiliary views.

## Evaluation population

Use only the same five development validation folds used for MVFD-SBN
mechanistic analysis.

No outer-test images are accessed.

Fold observations overlap and must not be interpreted as independent samples.

## Interpretation rule

Evidence for nuisance robustness requires both:

1. smaller representation shift than R0 under a nuisance; and
2. equal or better retention of class separability/classification.

A transform that is merely insensitive because it removes useful information
does not count as robust.

## Claim boundary

If F0 or W0 is more stable under some nuisance, the supported wording is:

> The transformed representation is more stable than R0 to the tested
> acquisition-like perturbation while retaining more defect-discriminative
> structure under the frozen Coffee17 validation protocol.

Do not write:
- camera invariant;
- illumination invariant;
- intrinsic coffee texture isolated;
- robust to arbitrary real-world cameras.

Actual camera invariance requires same-bean capture under multiple physical
cameras and lighting conditions.
