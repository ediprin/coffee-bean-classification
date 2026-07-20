# Coffee17 fine-grained open-set protocol v1

Status: **frozen before model evaluation**.

## Research question

Can a closed-set coffee-defect classifier reject a semantically unseen defect
without sacrificing recognition of the known defect classes? The first stage
tests the problem and standard baselines. It does not test HBP and does not
claim a new open-set algorithm.

## Controlled semantic splits

The source is one already deduplicated Coffee17 grouped fold. Every split holds
out exactly three of the seventeen classes, leaving fourteen known classes.

| Tier | Unknown classes | Operational relationship to known classes |
|---|---|---|
| Near | Partial Black; Partial Sour; Slight Insect Damage | The corresponding Full/Severe sibling remains known. |
| Medium | Withered; Floater; Shell | A related mechanism/morphology remains known, but not a direct severity sibling. |
| Far | Husk; Parchment; Fungus Damage | Covering residue is removed as a family and fungus is a removed singleton mechanism. |

These tiers are an operational research grouping, not an official SCA/SNI
taxonomy and not a WordNet hop distance. They may be reviewed before the first
run, but must not be changed after any OSR result is observed. The machine
readable source of truth is
`configs/osr/coffee17_osr_v1.yaml`.

No image from an unknown class may enter known train, known validation,
checkpoint selection, score calibration, prototype construction, or OpenMax
fitting. The original grouped train/validation/test identity assignment is
preserved.

## Controlled evaluation population

The full test set is reported as a diagnostic. The primary comparison uses a
deterministic balanced test manifest containing seven images per class:

```text
14 known classes x 7 = 98 known images
 3 unknown classes x 7 = 21 unknown images
```

Thus every semantic tier has the same openness, known/unknown ratio, and class
support. The manifest is selected without looking at model predictions.

## Baselines

Exactly one `EfficientNetV2-B0 + GAP + linear classifier + cross-entropy` model
is trained per split. The same checkpoint produces five knownness scores:

1. MSP: maximum softmax probability;
2. MLS: maximum unnormalised logit;
3. Energy: `T * logsumexp(logits / T)` as a knownness score;
4. Prototype: maximum cosine similarity to a class prototype;
5. OpenMax: one minus the recalibrated unknown probability, using the common
   Euclidean-cosine (`eucos`) activation distance.

Prototype and OpenMax statistics use only correctly classified known training
images. OpenMax fits a Weibull tail to class-mean activation distances.

## Metrics

Higher scores always mean “more likely known”. For every score and threshold
`t`:

```text
CCR(t) = # correctly classified known samples with score > t / N_known
FPR(t) = # unknown samples with score > t / N_unknown
OSCR   = area under CCR against FPR
```

The additional research metric macro-OSCR replaces each count with a mean of
per-class rates:

```text
macro_CCR(t) = mean_k [# correct-and-accepted known class k / N_known,k]
macro_FPR(t) = mean_u [# accepted unknown class u / N_unknown,u]
macro_OSCR   = area under macro_CCR against macro_FPR
```

Macro-OSCR is not the standard OSCR and is always labelled as an additional
class-balanced metric. Also report known Macro-F1, AUROC, AUPR-IN, AUPR-OUT,
FPR95, per-class unknown acceptance, and the balanced-manifest versions.

Curve metrics do not select a threshold. The operational threshold is the
fifth percentile of the known-validation score (target known acceptance 95%).
It is frozen before test evaluation; unknown test data never select it.

## Fail-fast decision

Stage 1 is three splits times seed 123: three training runs total. HBP is not
run. A later dual-statistic experiment is allowed only if near unknown is
meaningfully difficult and the baseline table leaves measurable headroom.
Any later proposed method must improve OSCR by at least two percentage points
over the strongest GAP score, improve AUROC on at least two of three tiers,
and reduce known Macro-F1 by no more than one point. A passing screen must then
be confirmed with multiple seeds.

External CBD/USK images belong to a separate cross-dataset OOD protocol and
must not be mixed into this semantic OSR result.

On Colab, prepared image folders should remain under `/content`, while
checkpoints and reports are written to Google Drive. This avoids copying three
OSR dataset views to Drive without sacrificing resumable training.

All methods use the same ImageNet-pretrained backbone. This controls the
comparison but does not prove that the held-out visual concepts were absent
from pretraining; that exposure is reported as a limitation.

## Primary references

- Lang et al., *From Coarse to Fine-Grained Open-Set Recognition*, CVPR 2024:
  semantic similarity controls OSR difficulty and MLS is a required strong
  baseline.
- Bendale and Boult, *Towards Open Set Deep Networks*, CVPR 2016: OpenMax and
  activation-vector Weibull recalibration.
- Liu et al., *Energy-based Out-of-distribution Detection*, NeurIPS 2020:
  post-hoc energy scoring.
- Palechor et al., *Large-Scale Open-Set Classification Protocols for
  ImageNet*, WACV 2023: OSCR reports known-class correctness jointly with
  unknown acceptance.
