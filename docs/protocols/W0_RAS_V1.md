# W0-RAS V1 Protocol

## Purpose

Test one minimal training-only transfer mechanism derived from the completed Coffee17
frequency/texture cue audit.

Working question:

> Can a raw-RGB MobileNetV3-Large learn the W0 fine-detail retention cue during
> training and retain the ordinary single-model raw-RGB inference path?

This is a **development-fold screening experiment**. The already-opened Coffee17
outer-test OOF population is not used for W0-RAS training, checkpoint selection,
or screening.

## Prior evidence motivating the treatment

The completed cue audit identified the most coherent W0 signal as relative
retention of fine Haar detail after VisuShrink soft-thresholding. The strongest
nonredundant candidate family was level-1 HH retained-energy ratio. RGB variants
were highly redundant, so V1 uses one scalar target: the mean of R/G/B L1-HH
retained-energy ratios.

The cue audit was used to select this target before W0-RAS training. Therefore
Coffee17 W0-RAS results are exploratory method-development evidence, not an
independent confirmation of the cue-selection hypothesis.

## Matched base model

The classification path is the frozen preprocessing-study R0 recipe:

- MobileNetV3-Large (mobilenetv3_large_100)
- GAP head
- linear 17-class classifier
- image size 224
- seed 42
- 50 epochs
- AdamW, lr 3e-4, weight decay 1e-4
- cosine scheduler
- cross entropy with label smoothing 0.1
- deterministic train rotations from the preprocessing-study loader
- raw RGB preprocessing at deployment

The auxiliary mechanism must not modify the deployment model.

## W0 target

For a training image tensor x in RGB [0,1], apply one level of the exact Haar
DWT implementation already frozen in the preprocessing study.

For each channel c in {R,G,B}, let HH_c denote the level-1 HH coefficients.
VisuShrink uses the complete level-1 detail tensor D_c = {LH_c, HL_c, HH_c}:

sigma_c = median(|D_c|) / 0.6745

tau_c = sigma_c * sqrt(2 log N)

where N is the number of finest-level detail coefficients used by the existing
visushrink_threshold implementation.

Soft threshold:

soft(v, tau) = sign(v) * max(|v| - tau, 0)

Pre- and post-threshold HH energies:

E_pre,c = mean(HH_c^2)

E_post,c = mean(soft(HH_c, tau_c)^2)

Per-channel retained-energy ratio:

r_c = E_post,c / (E_pre,c + eps)

Scalar target:

r(x) = (r_R + r_G + r_B) / 3

The implementation must call the existing haar_dwt2,
visushrink_threshold, and soft_threshold functions. No new wavelet
implementation is allowed.

## Fold-safe target standardization

For each development fold, compute mu_train and sigma_train only from that
fold's **training split**, using unrotated resized 224x224 images.

The training target is generated from the exact augmented tensor presented to
the classifier and standardized as:

r_tilde(x) = (r(x) - mu_train) / (sigma_train + eps)

No validation sample contributes to mu_train or sigma_train.

## Auxiliary head

Let z(x) be the ordinary GAP embedding returned by the raw-RGB model.

Training-only scalar head:

r_hat(x) = w^T z(x) + b

The auxiliary head is initialized after the base model while preserving the
global RNG state so adding the head does not shift the base model/dropout RNG
stream relative to the matched R0 recipe.

The head is discarded at deployment.

## Loss

L = L_CE + lambda_RAS * L_RAS

with

L_RAS = mean((r_hat - r_tilde)^2)

and frozen:

lambda_RAS = 0.05

No tuning of lambda is authorized in V1.

## Loss-scale fail-fast gate

During epoch 1 record:

rho = (lambda_RAS * mean(L_RAS)) / mean(L_CE)

If rho > 1.0, abort the run as a loss-contract failure. This gate prevents a
repeat of the intermediate-transfer scale failure observed previously.

The gate does not authorize changing lambda inside V1.

## Checkpoint selection

Checkpoint selection remains validation Macro-F1, matching the R0 preprocessing
study. The auxiliary regression loss is not used for checkpoint selection.

## Screening population

Run the same five Coffee17 development folds at seed 42.

Compare against the already-completed matched R0 development-fold runs from the
frozen preprocessing study. The comparison is valid only when clean-content,
fold-manifest, base configuration, and initial model-state contracts match.

Primary reported metrics:

- Macro-F1
- Hard-F1
- Worst-F1
- Accuracy
- Balanced Accuracy

## Promotion rule

V1 is promoted to a later confirmation step only if all of the following hold
on the five development folds:

1. mean Macro-F1 delta versus matched R0 > 0;
2. Macro-F1 delta is positive on at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0;
5. no loss-contract failure occurs.

This is intentionally strict because the project has repeatedly observed
aggregate gains that merely redistribute errors among difficult classes.

Failure does not authorize architecture stacking, lambda tuning, or selecting a
different wavelet band from the same audit post hoc.

## Inference contract

Final inference after dropping the auxiliary head is unchanged:

raw RGB -> MobileNetV3-Large -> GAP -> Linear17

Deployment parameter count, FLOPs, and latency are therefore identical to the
R0 GAP model, aside from any measurement noise.

## Explicit exclusions

W0-RAS V1 does not include:

- W0 image input at inference;
- a second backbone;
- multi-view CE;
- output KD;
- MVFD;
- intermediate spatial tensor MSE;
- attention;
- HBP;
- multi-level wavelet targets;
- LH/HL targets;
- confidence weighting;
- lambda search;
- outer-test OOF access.
