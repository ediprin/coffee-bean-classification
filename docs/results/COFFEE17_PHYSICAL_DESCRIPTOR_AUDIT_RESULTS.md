# Coffee17 Physical Descriptor Audit V1 — Results

Date analyzed: **2026-09-28**

Source artifact:

- `coffee17-physical-descriptor-audit-v1-analysis-package.zip`
- SHA-256: `6b3bf6a77e0a00ed76dc93db573d8a56dda2b67eb43a6573619ce37ec568cedc`

Scientific commit used by the notebook:

`ab1f2d6728bd7bf78599da73069229d40271447e`

## Integrity

Package metadata confirms:

- outer test materialized: **false**;
- model accessed: **false**;
- neural training executed: **false**.

Coffee17 provenance:

- raw images: 979;
- clean images after exact-duplicate/conflict handling: 965;
- same-class duplicates removed: 12;
- cross-class conflicting exact duplicates quarantined: 2.

Each fold was analyzed only on its train+val development materialization:

- fold 1: 766 images;
- fold 2: 769;
- fold 3: 772;
- fold 4: 774;
- fold 5: 779.

The five development sets overlap, so fold consistency is descriptive evidence
rather than five independent statistical replications.

## Mask QC

One image, `Dry Cherry/Dry Cherry_08.jpg`, touched the image border and failed
the conservative mask-QC rule in folds 1, 2, 3, and 5. It was not silently
dropped.

This sample is outside all four frozen target comparisons, so it does not alter
the Sour, Black, Insect, or selected shape-family conclusions below.

## 1. Full Sour vs Partial Sour — strong support for explicit color/extent information

The audit found **17 exploratory consistent candidate descriptors**.

Strongest stable signals include:

| Descriptor | Typical direction Full Sour - Partial Sour | Median Cliff's delta | FDR + medium effect folds |
|---|---:|---:|---:|
| L* q90 | lower | -0.730 | 5/5 |
| L* q75 | lower | -0.726 | 5/5 |
| L* mean | lower | -0.679 | 5/5 |
| warm fraction a*>5,b*>10 | higher | +0.645 | 5/5 |
| warm fraction a*>10,b*>15 | higher | +0.630 | 5/5 |
| a* q25 | higher | +0.626 | 5/5 |
| a* mean | higher | +0.614 | 5/5 |

Across folds, Full Sour median L* mean is approximately **47.8-48.9**, while
Partial Sour is approximately **60.7-61.1**.

For the fixed warm-color proxy `a*>5,b*>10`, Full Sour median fraction is
approximately **0.825-0.874**, while Partial Sour is approximately
**0.559-0.603**.

Interpretation:

> Coffee17 contains a strong and highly consistent color-distribution/extent
> signal separating Full Sour from Partial Sour.

Claim boundary:

- the warm proxy is not a validated sour-region segmentation;
- this result does not directly prove that the measured warm fraction equals the
  SCAA physical sour-area fraction.

It does establish that the missing information hypothesis is supported at the
level of label-free color distribution.

## 2. Full Black vs Partial Black — extremely strong explicit extent signal

The audit found **26 exploratory consistent candidate descriptors**.

The strongest result is the fixed luminance coverage curve.

For `L* < 50`:

- Full Black median coverage: approximately **0.975-0.979**;
- Partial Black median coverage: approximately **0.319-0.340**;
- median Cliff's delta: **+0.950**;
- FDR + medium effect: **5/5 folds**.

Mean L* is also sharply separated:

- Full Black median L* mean: approximately **28.6-31.0**;
- Partial Black median L* mean: approximately **55.1-56.3**;
- median Cliff's delta: **-0.922**;
- FDR + medium effect: **5/5 folds**.

Interpretation:

> Explicit dark-area / luminance-extent information is strongly present in
> Coffee17 and is especially well aligned with Full Black vs Partial Black.

The fixed L* thresholds remain generic luminance proxies rather than a
calibrated black-defect segmentation mask.

## 3. Severe vs Slight Insect Damage — topology proxy not supported

The frozen dark-component family produced **zero exploratory consistent
candidate descriptors**.

For every tested component-count/area proxy:

- no feature reached BH-FDR q < 0.05 in any fold;
- no feature satisfied the frozen medium-effect-and-FDR rule.

The direction also does not support a simple visible-hole-count interpretation:
for example, the median Cliff's delta for the k=3 dark-component count is about
**-0.279** for Severe minus Slight.

Interpretation:

> The current single-view dark-component proxy does not recover the SCAA
> Severe-vs-Slight perforation distinction in Coffee17.

Possible reasons include opposite-side holes being invisible, dark texture
being an imperfect hole proxy, and within-class visual variability. This route
is **not promoted** from the present audit.

## 4. Global geometry — strongly supported

All seven frozen geometry descriptors show significant class-group differences
for Broken/Cut/Immature/Shell/Withered in **5/5 folds**:

- area fraction;
- perimeter / sqrt(area);
- circularity;
- eccentricity;
- solidity;
- aspect ratio;
- extent.

Several pairwise separations are especially strong and directionally stable.

Examples:

- Broken vs Cut:
  - aspect ratio / eccentricity median Cliff's delta: **+0.706**;
  - circularity: **-0.436**.
- Cut vs Withered:
  - area fraction: **+0.569**;
  - solidity: **-0.444**.
- Immature vs Withered:
  - area fraction: **-0.526**;
  - aspect ratio / eccentricity: **-0.515**.
- Shell vs Withered:
  - aspect ratio / eccentricity: **+0.839**;
  - area fraction: **-0.739**.
- Broken vs Immature:
  - aspect ratio / eccentricity: **+0.907**;
  - circularity: **-0.901**.

Interpretation:

> Coffee17 contains substantial explicit global-geometry information that is
> complementary in form to WR-HBP's wavelet micro-texture cue.

## Frozen audit decision

The pre-model hypothesis is **partially supported**.

Promoted physical information:

1. **Full/Partial Sour:** color-distribution / extent cues;
2. **Full/Partial Black:** dark/luminance extent cues;
3. **Broken/Cut/Immature/Shell/Withered:** global bean geometry.

Not promoted:

4. **Severe/Slight Insect:** current dark-component topology proxy.

## Implication for WR-HBP

The audit supports the prior diagnosis that WR-HBP's strongest added cue is
micro-texture, while some fine-grained boundaries contain useful information in
orthogonal physical variables:

`RGB semantic + wavelet micro-texture + explicit extent/geometry evidence`.

This audit does **not** establish that fusing those descriptors into WR-HBP will
improve Macro-F1. It establishes only that the candidate information exists in
Coffee17 and is statistically separable before neural training.

Any downstream model experiment should therefore restrict itself to the
supported descriptor families and should not include the failed insect topology
proxy.
