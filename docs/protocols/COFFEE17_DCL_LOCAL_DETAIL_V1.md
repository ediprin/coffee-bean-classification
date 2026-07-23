# Coffee17 DCL local-detail protocol v1

## Status

**Completed with three-seed DCL0 FAIL; all DCL variants stopped.** Test
Coffee17 remains locked. DCL0 passed the seed-123 screen but failed the
predefined aggregate gate over seeds 42, 123, and 2026. DCL1 confirmation was
therefore not authorized; DCL2 had already failed to improve over DCL1.

## Research question

Can a training-only local-detail objective improve Coffee17 classification
without adding deployment cost, and only if that premise holds, does
confusion-aware contrastive learning add value beyond ordinary supervised
contrastive learning?

This question is different from earlier failed studies:

- `hbp_moe` added a local max-pooling inference expert;
- `CP1/CP2` used two different same-class images and global embeddings;
- progressive multi-granularity used jigsaw branches and multi-stage
  classification;
- DCL uses the same augmented image before and after controlled local region
  confusion, plus explicit swap and layout reconstruction objectives.

## Primary literature and implementation audit

The first stage follows Chen et al., *Destruction and Construction Learning
for Fine-Grained Image Recognition*, CVPR 2019, and was checked against the
authors' public implementation `JDAI-CV/DCL`.

The official mechanism contains:

1. a common augmented image and a region-confused view;
2. classification CE on both views;
3. a swap classifier distinguishing original and confused images;
4. a 1x1 feature-map head predicting the region layout with L1 loss;
5. only the ordinary backbone-GAP-classifier path at inference.

The contrastive objective is based on Khosla et al., *Supervised Contrastive
Learning*, NeurIPS 2020. Confusion-aware negative weighting reuses the already
audited Coffee17 CP2 formulation, but is conditional on DCL0 and is not assumed
to work.

## Controlled models

| Code | Shared model | Training-only objective |
|---|---|---|
| BE2G | EfficientNetV2-B0 + GAP + linear | CE |
| BE2H | EfficientNetV2-B0 + HBP + linear | CE |
| DCL0 | EfficientNetV2-B0 + GAP + linear | CE + swap CE + layout L1 |
| DCL1 | DCL0 | DCL + vanilla SupCon |
| DCL2 | DCL0 | DCL + confusion-aware SupCon |

All candidates use ImageNet-pretrained `tf_efficientnetv2_b0.in1k`, input 224,
the same clean grouped fold, optimizer, schedule, augmentation, label
smoothing, epoch count, and checkpoint selection by validation Macro-F1.

The DCL auxiliary heads are training-only. The deployment path for DCL0/1/2
is:

```text
image -> EfficientNetV2-B0 -> final feature -> GAP -> linear classifier
```

## Region confusion and explicit deviations

The Coffee17 adaptation uses a 7x7 grid. Neighboring rows and columns are
locally permuted, preserving DCL's Region Confusion Mechanism intent. The
original and confused images share the same stochastic rotation, crop, flip,
and color augmentation.

Differences from the official implementation are explicit:

- input is 224 rather than 448;
- backbone is EfficientNetV2-B0 rather than ResNet-50;
- no fixed 10-pixel border crop is applied because Coffee17 already receives
  the frozen 224 standard transform;
- the region permutation is tracked exactly instead of inferred by matching
  patch brightness, eliminating ambiguous layout targets;
- the final layout map is resized to 7x7 only if the selected backbone feature
  does not already have that spatial size.

Therefore DCL0 is a **controlled adaptation**, not a claim of exact numerical
reproduction.

## Objective

For original/confused paired views:

```text
L_DCL = CE_class + 1.0 * CE_swap + 1.0 * L1_layout
```

Conditional second stage:

```text
DCL1: L = L_DCL + 0.2 * SupCon
DCL2: L = L_DCL + 0.2 * confusion-aware SupCon
```

DCL2 builds its symmetric confusion matrix from train predictions only, uses a
five-epoch ordinary-SupCon warm-up, then applies:

```text
w(a,b) = 1 + 2.0 * C_train[a,b]
```

Validation/test labels never define training pairs or weights.

## Fail-fast sequence

### Stage 1: DCL

- Train only `DCL0`, seed 123.
- Compare against existing BE2G and BE2H validation reports.
- `DCL0_final` passes only if DCL0 versus BE2G has:
  - positive Macro-F1 delta;
  - positive Hard-F1 delta;
  - Worst-F1 delta at least -1 point.

If it fails, stop. DCL1, DCL2, seed 42/2026, and test are prohibited.

### Stage 2: contrastive

Only after Stage 1 passes:

- train DCL1 and DCL2 on seed 123;
- DCL2 final passes only if it independently passes against:
  - BE2H;
  - DCL0;
  - DCL1.

This isolates the DCL local-detail premise, ordinary contrastive contribution,
and confusion-aware contribution rather than attributing a combined result to
all components.

### Confirmation amendment after seed-123 screening

The seed-123 results triggered the following decision before any 42/2026
training:

- DCL0 passed against BE2G and BE2H;
- DCL1 passed against BE2G, BE2H, and DCL0;
- DCL2 produced the same reported Macro/Hard/Worst metrics as DCL1 and failed
  the strict DCL1 comparison.

Therefore DCL2 is stopped. Confirmation is sequential:

1. aggregate DCL0 over seeds 42, 123, and 2026;
2. stop if the original DCL0 gate fails on that aggregate;
3. only when DCL0 still passes, aggregate DCL1 over the same three seeds;
4. DCL1 final requires independent PASS against BE2H and DCL0.

The amendment reduces training cost and prevents the already non-informative
DCL2 weighting from consuming additional seeds. Test remains locked.

The resulting three-seed DCL0 deltas against GAP were Macro `+1.17 ± 1.21`,
Hard `-0.11 ± 2.88`, and Worst `-6.46 ± 16.72` percentage points. The gate
failed; no further DCL training is authorized under v1.

## Persistence

Every epoch writes `last.pt`, `best.pt`, `history.json`,
`resolved_config.json`, and `artifact_manifest.json`. With
`--artifact-required`, the run is uploaded to Hugging Face every configured
interval and aborts when persistence fails.

## Claim boundary

- A passing DCL0 supports a paper-grounded local-detail adaptation.
- A passing three-seed DCL1 supports DCL plus ordinary SupCon under this
  Coffee17 protocol.
- DCL2 cannot support a confusion-aware claim because it did not improve over
  DCL1 during the authorized screening.
- Neither result alone proves global algorithmic novelty.
- Novelty wording requires a separate literature comparison after empirical
  isolation.
- A one-seed validation result is screening, not thesis-final evidence.
