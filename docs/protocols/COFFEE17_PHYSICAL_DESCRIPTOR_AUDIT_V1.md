# Coffee17 Physical Descriptor Audit V1

Status: **frozen exploratory audit; no neural training**

Branch:

`codex/coffee17-physical-descriptor-audit-v1`

## Research question

Before adding another learned branch to WR-HBP, test whether physically motivated,
label-free descriptors actually separate the relevant Coffee17 defect groups on the
development data.

The descriptor extractor never sees class labels. Labels are used only after extraction
for statistical grouping.

## Evidence basis

The audit is grounded in prior coffee literature and defect definitions:

- García et al. (2019): surface area, roundness, eccentricity, and damaged-area
  relation are useful for coffee-bean defect inspection; HSV/LUV were used for
  color-related defect segmentation.
- Santos et al. (2020): area and color descriptors were among the most important
  features for defect classification.
- Chang & Huang (2021): Partial Sour, Immature, and Withered are strongly
  view-dependent; insect damage can appear on both sides.
- Chang & Liu (2024): Withered wrinkles, incomplete Cut contours, and
  Immature/Withered confusion motivate explicit geometry analysis.
- SCAA defect handbook: Partial Sour and Partial Black are defined by affected
  area being less than one half; Severe vs Slight Insect Damage differs by
  perforation count.
- Tulsi et al. (2026): label-free coffee-bean segmentation via red-channel Otsu
  plus morphology provides precedent for handcrafted morphology extraction.

## Data boundary

The audit uses the same Coffee17 clean/provenance machinery already used in the
preprocessing study.

For each fold separately:

- materialize only that fold's `train + val` development data;
- do not materialize `source/test`;
- extract descriptors from original, non-augmented images;
- analyze the fold independently.

The five fold-specific audits are summarized only as consistency evidence. Because
development sets overlap across folds, fold consistency is not treated as five
independent statistical replications.

No neural model is trained and no model checkpoint is accessed.

## Label-free bean mask

Coffee17 uses a controlled white background and standardized single-bean images.
The bean mask is extracted with:

1. red-channel Otsu threshold;
2. remove very small components;
3. morphological closing;
4. retain the largest connected component;
5. fill interior holes for bean-shape measurement.

Mask QC is recorded for every image. No failed image may be silently dropped.

## Frozen descriptor families

### Bean geometry

- area fraction;
- perimeter / sqrt(area);
- circularity = 4*pi*A/P^2;
- eccentricity;
- solidity;
- major/minor-axis aspect ratio;
- extent.

### CIELAB distribution

For L*, a*, b* inside the bean mask:

- mean;
- standard deviation;
- q10, q25, q50, q75, q90.

### Fixed luminance coverage curve

Fractions of bean pixels satisfying:

- L* < 20, 30, 40, 50, 60, 70.

These are generic luminance coverage proxies. They are not called validated
black-defect masks.

### Fixed warm-color coverage proxies

Fractions satisfying fixed positive a*/b* conditions:

- a*>0, b*>10;
- a*>5, b*>10;
- a*>5, b*>15;
- a*>10, b*>15;
- a*>10, b*>20.

These are generic warm/yellow-red coverage proxies motivated by the physical sour
description. They are not called validated sour-region masks because García et al.
do not provide a directly reusable numeric HSV/LUV sour threshold in the audited
paper text.

### Dark-component topology proxies

Within an eroded bean interior, connected components are extracted at robust
luminance thresholds:

`L* < median(L*) - k * 1.4826*MAD(L*)`

for k = 2.0, 2.5, 3.0.

For each k report:

- dark component count;
- total component area fraction;
- largest component area fraction.

These are topology proxies, not exact insect-hole counts.

## Frozen statistical questions

1. **Full Sour vs Partial Sour**
   - CIELAB distribution;
   - warm-color coverage proxies.

2. **Full Black vs Partial Black**
   - CIELAB distribution;
   - luminance coverage curve.

3. **Severe vs Slight Insect Damage**
   - dark-component topology proxies.

4. **Shape family**
   - Broken, Cut, Immature, Shell, Withered;
   - Kruskal-Wallis on each geometry descriptor;
   - pairwise Mann-Whitney comparisons.

Within each hypothesis family, p-values are corrected with Benjamini-Hochberg FDR.

Effect size is Cliff's delta. A fold-level feature is flagged as medium-effect-and-FDR
when:

- q < 0.05; and
- |Cliff's delta| >= 0.33.

Across five overlapping development folds, a descriptor is marked only as an
**exploratory consistent candidate** when:

- direction agrees in at least 4/5 folds;
- median |Cliff's delta| >= 0.33;
- medium-effect-and-FDR condition holds in at least 3/5 folds.

This consistency flag is descriptive development evidence, not an independent
confirmatory test.

## Decision rule after the audit

No learned morphology branch is implemented from intuition alone.

A downstream residual cue is considered only for physical information that:

- has direct coffee-domain justification; and
- shows consistent Coffee17 development separation in this frozen audit.

If the physically motivated descriptors do not separate the target classes, stop
the morphology/coverage route without neural training.

## Prohibited claims

Do not claim:

- direct sour-area measurement from the warm-color proxies;
- direct black-area measurement from the L* coverage curve;
- exact insect-hole counts from dark components;
- SCAA-defined separation between Coffee17 Broken and Cut;
- five statistically independent replications from the overlapping folds.
