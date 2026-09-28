# WR-PDR-HBP V1 — Physical-Descriptor Logit Residual on Frozen WR-HBP

Status: **frozen development protocol; training pending**

Branch:

`codex/physical-logit-residual-wr-hbp-v1`

## 1. Research question

Can the physically supported information discovered by the Coffee17 descriptor audit
improve WR-HBP without perturbing the successful RGB + wavelet feature
representation?

Matched comparison inside each fold:

- control: exact frozen `WR-HBP` validation logits;
- candidate: `WR-PDR-HBP = WR-HBP logits + masked physical residual`.

The candidate uses the **same exact trained WR-HBP checkpoint** as the control.
Only a tiny post-hoc linear residual is fitted.

## 2. Why this design

Completed evidence established:

- WR-HBP passed its original development screen and improved Macro-F1 in 5/5
  folds, but later retraining exposed CUDA run-to-run drift;
- WRC-HBP showed that injecting a globally useful contrast cue into the learned
  feature map can improve Sour locally while damaging other classes;
- the no-training Coffee17 Physical Descriptor Audit found strong, consistent
  information for:
  - Full vs Partial Sour: color-distribution / extent;
  - Full vs Partial Black: luminance extent;
  - Broken/Cut/Immature/Shell/Withered: global geometry;
- the current dark-component proxy for Severe vs Slight Insect Damage was not
  supported and is excluded.

The residual is therefore applied **after** WR-HBP classification logits. WR-HBP
features, HBP, wavelet branch, and classifier remain unchanged during residual
fitting.

## 3. Strict deterministic WR-HBP base

Each fold trains one WR-HBP base using the original frozen WR-HBP training recipe:

- Coffee17 preprocessing-development fold;
- seed 42;
- input 224;
- batch 32;
- deterministic rotation schedule [0,45,90,135,180,225,270];
- MobileNetV3-Large;
- HBP [1,3,4], projection 512;
- L1 luminance Haar/VisuShrink wavelet residual, hidden 16;
- AdamW lr 3e-4, weight decay 1e-4;
- CE label smoothing 0.1;
- cosine scheduler;
- 50 epochs;
- checkpoint selected by validation Macro-F1.

This experiment additionally requires:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `torch.use_deterministic_algorithms(True)`;
- cuDNN benchmark disabled;
- cuDNN deterministic enabled;
- TF32 disabled for CUDA matmul and cuDNN.

If a required operation has no deterministic CUDA implementation, the run must
raise rather than silently fall back to a nondeterministic path.

### Deterministic VisuShrink implementation note

The first strict-deterministic Kaggle attempt stopped before completing fold 1
because PyTorch's CUDA implementation of `median(dim=...)` is not deterministic.
The failure occurred inside the VisuShrink noise estimator, before a model result
was produced.

The frozen correction keeps the original VisuShrink equation unchanged:

`sigma = median(abs(detail)) / 0.6745`.

Only that median statistic is computed on CPU when strict deterministic CUDA is
enabled, then the threshold tensor is returned to the original CUDA device for
soft-thresholding. The wavelet branch remains no-grad preprocessing, so this does
not alter the trainable graph.

This correction was made in response to an execution-compatibility failure, not
to any observed classification outcome.



### Deterministic HBP spatial-alignment implementation note

A second strict-deterministic preflight stopped during the first backward pass
because PyTorch's CUDA `adaptive_avg_pool2d_backward` used for HBP stage
alignment has no deterministic implementation.

For the frozen MobileNetV3 HBP grids, the spatial ratios are exact integer
partitions (56->7 and 14->7). Under strict deterministic execution, HBP therefore
uses an equivalent non-overlapping block mean implemented as reshape + mean.
When deterministic algorithms are disabled, the legacy
`adaptive_avg_pool2d` path remains unchanged.

The deterministic path is validated against adaptive average pooling for the
exact-bin case and the Kaggle preflight now executes a complete WR-HBP CUDA
forward + backward smoke test before any fold training.

This correction was made after an execution failure and before any completed
WR-PDR fold result was observed.


## 4. Frozen physical feature vector

Only information supported by the completed descriptor audit is retained.

### Global geometry

- area fraction;
- perimeter / sqrt(area);
- circularity;
- eccentricity;
- solidity;
- major/minor-axis aspect ratio.

`extent` is excluded because it is more dependent on image-axis orientation.

### Sour cue

- mean L*;
- warm fraction `a*>5, b*>10`.

These are color-distribution / extent proxies and are not called a validated
sour-region mask.

### Black cue

- mean L*;
- fraction `L*<50`.

The luminance fraction is not called a calibrated black-defect segmentation mask.

### Explicit exclusion

The dark-component / insect topology family is disabled because the physical
descriptor audit did not support it.

## 5. Masked class-feature structure

The residual is class-specific and intentionally sparse.

Geometry features may correct logits only for:

- Broken;
- Cut;
- Immature;
- Shell;
- Withered.

Sour features may correct logits only for:

- Full Sour;
- Partial Sour.

Black features may correct logits only for:

- Full Black;
- Partial Black.

All residual weights for the remaining class-feature combinations are exactly
zero.

Total active residual parameters:

`5*6 + 2*2 + 2*2 = 38`.

## 6. Residual formulation

Let the frozen WR-HBP logits be:

`l_WR(x) in R^17`.

Let the nine-dimensional physical descriptor vector be:

`m(x)`.

Descriptors are standardized using **source/train statistics only**:

`m_tilde_j = (m_j - mu_j_train) / sigma_j_train`.

The candidate logits are:

`l_PDR(x) = l_WR(x) + W_PDR m_tilde(x)`.

Only the 38 pre-authorized entries of `W_PDR` are free parameters.

Initialization:

`W_PDR = 0`.

Therefore before fitting:

`l_PDR(x) = l_WR(x)`

exactly.

## 7. Residual fitting

The WR-HBP checkpoint is frozen.

Residual fitting uses only the fold's original, non-augmented `source/train`
images:

- base WR-HBP logits are computed with the frozen best checkpoint;
- physical descriptors are extracted label-free;
- class labels are used only by the residual classification loss;
- optimizer: deterministic CPU float64 L-BFGS-B;
- objective: cross-entropy with label smoothing 0.1 + L2;
- L2 coefficient: 0.01;
- maximum optimizer iterations: 500;
- no hyperparameter sweep.

The problem has only 38 active parameters.

## 8. Descriptor QC policy

No failed descriptor mask is silently accepted as ordinary data.

For a train image whose descriptor mask fails the frozen QC:

- exclude that row from residual fitting;
- keep the image in the already-trained WR-HBP base training history unchanged.

For a validation image whose descriptor mask fails QC:

- apply zero physical residual;
- preserve its exact WR-HBP logits.

All QC-fail identities are recorded.

## 9. Paired evaluation

For each fold, candidate and control use the **same exact WR-HBP validation
logits**. The only difference is the fitted physical residual.

Report:

- Accuracy;
- Balanced Accuracy;
- Macro-F1;
- Hard-F1;
- Worst-F1;
- positive-delta fold counts;
- rescue/damage;
- per-class F1 deltas;
- hard-group deltas;
- fitted residual weights;
- descriptor-QC failures;
- base best epoch and wavelet gate;
- deterministic-runtime settings.

## 10. Frozen screening gate

WR-PDR-HBP passes only if all are true:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 improves in at least 3/5 folds;
3. mean paired Hard-F1 delta >= 0;
4. mean paired Worst-F1 delta >= 0.

If the gate fails:

> stop WR-PDR-HBP. Do not tune descriptor thresholds, feature list, class mask,
> L2, optimizer, physical feature standardization, WR-HBP, or decision rule on
> these reused folds.

## 11. Claim boundary

This is post-primary exploratory development.

The physical descriptors were selected after the completed Coffee17 development
audit, so these folds are reused development evidence.

A PASS can justify promotion to a separately frozen confirmation step. It is not
an independent final superiority claim.

Outer test remains untouched.
