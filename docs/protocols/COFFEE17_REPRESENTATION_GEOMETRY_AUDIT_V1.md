# Coffee17 Representation Geometry Audit V1

## Purpose

Single decision experiment before any new Coffee17 architecture work.

Question: are the persistent hard-class errors mainly caused by the supervised ImageNet MobileNetV3 representation, or do they remain under a substantially different self-supervised representation?

## Locked comparison

- Dataset: Coffee17 original, 979 images / 17 classes.
- Evaluation: the existing five deterministic development folds, seed 42.
- Outer test: never materialized into the experiment runtime and never evaluated.
- Input: original raw RGB, resized to 224x224. No HBP, WR, wavelet, CLAHE, object crop, learned preprocessing, or train-time augmentation.
- Encoder A: `mobilenetv3_large_100`, ImageNet supervised weights, frozen.
- Encoder B: `vit_small_patch14_dinov2.lvd142m`, DINOv2 self-supervised weights, frozen.
- No encoder fine-tuning.
- Embeddings: L2-normalized.
- Decoders only:
  - cosine 5-NN;
  - multinomial logistic-regression linear probe with fixed `C=1.0`.
- No hyperparameter search.

## Primary metrics

For every fold and decoder:

- Macro-F1
- Hard-F1
- Worst-class F1
- Accuracy / balanced accuracy for context

The hard-class score is computed from the union of classes in the preregistered hard pairs.

## Preregistered hard pairs

1. Withered vs Immature
2. Severe Insect Damage vs Slight Insect Damage
3. Cut vs Slight Insect Damage
4. Partial Sour vs Full Sour
5. Slight Insect Damage vs Fade
6. Full Black vs Partial Black

For each pair the audit records bidirectional confusion counts and representation geometry.

## Geometry diagnostics

Using training-fold embeddings only, compute class centroids and for each hard pair:

- cosine distance between class centroids;
- mean within-class cosine distance for each class;
- inter-class / pooled-intra-class distance ratio.

This geometry is diagnostic only. Nearest-centroid is not used as a classifier.

For every validation sample, record:

- 5-NN same-class fraction;
- own-centroid cosine similarity;
- nearest rival class and similarity;
- own-vs-rival centroid margin;
- kNN prediction;
- linear-probe prediction.

## Decision gate

The representation hypothesis `H_R` receives support only if the DINOv2 linear probe simultaneously satisfies all of:

1. mean Macro-F1 delta vs MobileNetV3 > 0;
2. Macro-F1 delta > 0 in at least 4/5 folds;
3. mean Hard-F1 delta > 0;
4. Hard-F1 delta > 0 in at least 4/5 folds;
5. total confusion across the six preregistered hard pairs is lower than MobileNetV3.

Decision labels:

- `SUPPORT_H_R_REPRESENTATION`: all five conditions pass.
- `NO_CLEAR_H_R_SUPPORT`: at least one condition fails.

Magnitude is still reported; the gate is directional/consistency evidence, not a claim of practical significance.

## Interpretation after the run

- `SUPPORT_H_R_REPRESENTATION`: proceed to a representation-transfer route (teacher adaptation/distillation to a lightweight deployment model), without returning to HBP/WR.
- `NO_CLEAR_H_R_SUPPORT`: do not distill DINOv2 by default; inspect cross-representation persistent errors and move to local fine-grained learning only if the failures remain localized/subtle.

No outer-test result is used to choose the branch.
