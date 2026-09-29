# Persistent Error Audit — strict WR-HBP, WR-PDR, and GCE-LS

Status: diagnostic audit after WR-PDR and GCE-LS results.

Source packages:

- WR-HBP/GCE-LS SHA256:
  `9fbcb438deeb45559dac0b6e881aa729f7819c83a9ea6bd238d1fbcf78fd589a`
- WR-PDR SHA256:
  `29ff8f7694ddb2e8940fb1b9bd800a712545d057091b86a63d0f14a72df88255`
- physical descriptor audit SHA256:
  `6b3bf6a77e0a00ed76dc93db573d8a56dda2b67eb43a6573619ce37ec568cedc`

The strict WR-HBP predictions in the GCE package and WR-PDR package are
bit-for-bit identical for all 485 validation observations, including all
probabilities.

## Persistence

Strict WR-HBP makes 40 errors over 485 fold-observations.

- 35/40 WR-HBP errors remain wrong under both WR-PDR and GCE-LS: **87.5%**.
- 31/35 persistent errors keep the exact same wrong class under all three
  methods.
- The dominant persistent WR-HBP confusion directions are:

| Actual | Predicted | Count |
|---|---|---:|
| Withered | Immature | 5 |
| Severe Insect Damage | Slight Insect Damage | 4 |
| Cut | Slight Insect Damage | 3 |
| Partial Sour | Full Sour | 3 |
| Slight Insect Damage | Fade | 3 |
| Slight Insect Damage | Severe Insect Damage | 3 |

Seven raw images appear as validation samples in two independent development
folds and are wrong in all six corresponding predictions across WR-HBP,
WR-PDR, and GCE-LS:

- `Broken_55.jpg`
- `Cut_59.jpg`
- `Severe Insect Damange_49.jpg`
- `Slight Insect Damage_08.jpg`
- `Slight Insect Damage_11.jpg`
- `Withered_06.jpg`
- `Withered_37.jpg`

Their mean strict WR-HBP true-class probabilities are only 4.44%, 8.23%,
10.02%, 10.60%, 12.65%, 9.69%, and 14.22%, respectively. These are not
borderline one-off mistakes; the same images remain systematically on the
wrong side of the decision boundary across independently trained folds.

## Physical-descriptor diagnostic

The frozen physical descriptors explain part, but not all, of these failures.

- `Broken_55.jpg`: global geometry (circularity, perimeter/area, aspect ratio,
  area fraction) is substantially closer to the wrong Immature/Withered side
  than to the Broken class center. A global shape correction therefore cannot
  reliably recover the local fracture cue.
- `Cut_59.jpg`: warm-color fraction and several geometry descriptors are
  closer to Slight Insect Damage, while luminance is closer to Cut. The
  discriminative evidence is not a single global color statistic.
- `Severe Insect Damange_49.jpg`: several global geometry descriptors are
  closer to Slight Insect Damage even though contour descriptors retain some
  Severe signal. Severity likely depends on localized damage topology rather
  than whole-bean geometry.
- `Slight Insect Damage_08.jpg`: this is an extreme physical outlier inside
  its labeled class; L* mean is 78.90 with very little dark area and it is
  repeatedly predicted as Fade. Its solidity is also an extreme within-class
  outlier.
- `Slight Insect Damage_11.jpg`: global descriptors are not strongly
  outlying, yet it is repeatedly predicted as Severe Insect Damage. This is a
  direct example where global hand-crafted descriptors do not explain the
  failure.
- `Withered_06.jpg` and `Withered_37.jpg`: one is pulled toward Immature
  mainly by luminance/dark-area cues and the other mainly by geometry. Even
  within one confusion direction, the failure mechanism is heterogeneous.

## Consequence for the next experiment

The observed failure pattern argues against another global loss-only or
global-descriptor residual as the next step. The next mechanism must preserve
the already-strong global WR-HBP decision while adding **localized,
class-conditional evidence for ambiguous top-k classes**.

This audit is the required design input for the next experiment. Any new
candidate must state explicitly how it addresses these persistent cases and
must not globally replace the successful WR-HBP softmax structure.
