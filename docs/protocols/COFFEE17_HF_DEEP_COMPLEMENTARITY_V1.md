# Coffee17 HF–Deep Complementarity V1

Status: **registered before outcome observation**

Purpose: test one narrow question before any further architecture search:

> Do explicit color/texture/shape descriptors add discriminative information to
> the already-established Coffee17 HBP representation?

This is a **representation complementarity screening**, not a new backbone
search and not a claim that handcrafted/deep fusion is novel in coffee.

## Evidence basis

Tulsi et al. (2026) extract 71 handcrafted descriptors spanning shape, RGB/
CIELAB color, GLCM, LBP, Law's Texture Energy, DWT, and Gabor features, and
compare handcrafted features with learned image embeddings. Their paper reports
that image embeddings outperform handcrafted-only models overall, while
handcrafted color/texture/multiscale descriptors retain interpretable signal.

The 71 feature names are frozen to the Appendix list in Tulsi et al. The present
implementation is **Tulsi-aligned rather than a bit-identical reproduction**:

- Tulsi's supplied paper states red-channel Otsu segmentation. Coffee17 V1 uses
  the project's already-audited, label-free border-background RGB-distance Otsu
  mask because the red-channel version had already shown a pale-bean/dark-spot
  failure mode before this experiment.
- The supplied paper references supplementary details for exact descriptor
  kernels/settings that are not contained in the provided PDF. Therefore the
  frozen implementation records its exact GLCM/LBP/Laws/DWT/Gabor conventions
  in code and must not be described as an exact software reproduction.
- The Appendix name `Average_Blue` is interpreted as CIELAB b* because the
  same list separately contains RGB `Average_B` and also CIELAB L*/a*.

## Data contract

- Coffee17 original images only: 979 identities / 17 classes.
- Existing strict five-fold development split, seed 42.
- Fold materialization contains only `source/train` and `source/val`; outer
  test must not be materialized.
- HBP training augmentation is the canonical deterministic rotation schedule
  `[0,45,90,135,180,225,270]`.
- Probe representation extraction uses each **original image once**. No rotated
  copies are used to fit MRMR, scaling, or the downstream probe.
- Train/validation identity overlap must be zero.

## HBP representation

The learned representation is the existing MobileNetV3-Large + HBP:

- ImageNet pretrained MobileNetV3-Large;
- HBP stages `[1,3,4]`;
- projection dimension 512;
- 1536-D HBP embedding;
- raw RGB 224x224;
- AdamW 3e-4, weight decay 1e-4;
- CE with label smoothing 0.1;
- cosine schedule, 50 epochs;
- best checkpoint selected by validation Macro-F1.

A compatible external HBP checkpoint may be supplied to the runner. Otherwise
the runner trains this core itself. The downstream three-arm comparison always
uses the **same frozen HBP checkpoint within a fold**.

## Handcrafted representation

The exact 71-D ordered feature contract is implemented in
`src/bilinear_lmmd/features/tulsi_handcrafted.py` and contains:

- 6 mean RGB/CIELAB color features;
- 14 GLCM/Haralick-style features;
- 6 LBP energy/entropy features at (R,P) = (1,8), (2,16), (3,24);
- 6 Laws texture-energy features;
- 18 bior3.3 DWT detail mean/std features over three levels;
- 16 Gabor magnitude mean/std features over four orientations and two
  frequencies;
- 5 shape features.

MRMR is fitted **only on the training identities of each fold** and selects
exactly 20 descriptors. Selection is frozen to a deterministic MI relevance
minus mean MI redundancy criterion with random state 42.

No validation labels participate in feature selection.

## Three arms

The downstream classifier is identical for all arms: L2-regularized
multinomial logistic regression, `C=1`, `lbfgs`, `max_iter=5000`, no
validation tuning.

### HF20

1. extract 71 descriptors;
2. MRMR-20 fitted on train;
3. StandardScaler fitted on train-selected features;
4. row-wise L2 normalization;
5. logistic probe.

### HBP-EMB

1. extract the frozen HBP embedding from each original image;
2. row-wise L2 normalization;
3. the same logistic probe.

### HF20 + HBP-EMB

Concatenate the two unit-norm modality blocks and divide by sqrt(2), so neither
block is given an arbitrary scale advantage:

[
z_i = rac{1}{sqrt 2}
      [N(h_i);N(S(q_i))].
]

The same logistic probe is then fitted.

## Metrics

Per fold:

- Accuracy;
- Balanced Accuracy;
- Macro-F1;
- Hard-F1;
- Worst-F1;
- rescue/damage of fusion against HBP-EMB;
- audited confusion counts for the six frozen difficult pairs.

Frozen difficult pairs:

1. Withered ↔ Immature
2. Severe Insect Damage ↔ Slight Insect Damage
3. Cut ↔ Slight Insect Damage
4. Partial Sour ↔ Full Sour
5. Slight Insect Damage ↔ Fade
6. Full Black ↔ Partial Black

The selected MRMR feature names are also recorded per fold. Their frequency is
reported descriptively; it is not used to tune V1.

## Frozen decision gate

`SUPPORT_HF_COMPLEMENTARITY` only if **all** are true across five folds:

1. mean Fusion − HBP-EMB Macro-F1 > 0;
2. Macro-F1 delta > 0 in at least 4/5 folds;
3. mean Hard-F1 delta > 0;
4. total audited hard-pair confusions decrease;
5. mean Worst-F1 delta >= 0.

Otherwise:

`NO_CLEAR_HF_COMPLEMENTARITY`

If V1 fails, stop this HF-fusion route on the reused development folds. Do not
rescue it by trying RealMLP, FT-Transformer, different MRMR K, different
regularization C, feature-family pruning, or learned fusion on the same folds.

If V1 passes, it establishes only that explicit descriptors are complementary
to the HBP representation under this development protocol. Any later model
development must be registered separately before accessing new outcome data.

## Outer test

Outer test access is forbidden in V1.
