# Coffee17 Frequency–Texture Cue Audit V1 — Evidence Summary

## Status

Completed train-free analysis on the exact 965 clean Coffee17 OOF identities from the
frozen preprocessing study.

This artifact is **exploratory method-development evidence**. The same Coffee17 OOF
population was already observed during preprocessing analysis, so these findings are
not an independent confirmation set.

## Input integrity

- samples: 965
- classes: 17
- F0 rescue/harm status:
  - both correct: 789
  - rescued: 43
  - harmed: 50
  - both wrong: 83
- W0 rescue/harm status:
  - both correct: 794
  - rescued: 46
  - harmed: 45
  - both wrong: 80

## Analysis model

For each extracted cue q, the continuous outcome was preprocessing improvement in
true-class probability relative to R0.

For W0:

delta_p = W0_true_prob - R0_true_prob

Each cue was standardized, then fitted with:

delta_p ~ z(cue) + class + fold + R0_true_prob

using OLS with HC3 robust covariance.

Multiple testing used Benjamini-Hochberg FDR across the 111 W0 cue features.

This model is used as an exploratory association screen. It does not establish a
causal effect of an individual frequency component.

## W0 result

15 / 111 W0 features passed q < 0.05.

Strongest threshold effects:

| feature | beta per 1 SD | q |
| --- | ---: | ---: |
| threshold G | -0.04443 | 0.00465 |
| threshold R | -0.04206 | 0.00465 |
| threshold B | -0.04044 | 0.00465 |

Strongest retained-energy effects:

| feature | beta per 1 SD | q |
| --- | ---: | ---: |
| L1-HH R retained ratio | +0.04492 | 0.00647 |
| L1-HH G retained ratio | +0.04475 | 0.00647 |
| L1-HH B retained ratio | +0.04466 | 0.00647 |

Other significant retained-ratio features occurred mainly in L1/L2 detail bands,
plus a small number of L4 blue-channel detail ratios.

Neither raw pre-threshold energy nor post-threshold energy formed the main
significant family. The coherent signal was the **relative amount of detail energy
surviving VisuShrink**.

## Redundancy

The three L1-HH RGB retained ratios are nearly redundant:

- R vs G correlation: approximately 0.996
- R vs B correlation: approximately 0.994

Threshold and retention are also strongly inversely related. For example:

- threshold R vs L1-HH R retained ratio: approximately -0.956
- threshold G vs L1-HH G retained ratio: approximately -0.961

Therefore V1 must not create three separate RGB supervision heads. The predeclared
minimal target is the mean RGB L1-HH retained-energy ratio.

## Class information

The mean-RGB L1-HH retained ratio has substantial between-class structure on the
965 OOF identities (one-way eta-squared approximately 0.63). This supports its use
as a fine-grained auxiliary descriptor, while still requiring downstream
classification testing.

## F0 result

The corresponding F0 angular-frequency cue screen did not produce a stable
FDR-significant family explaining improvement over R0. F0 can contain class
structure, but this audit did not provide a defensible basis for selecting one
angular component for selective transfer.

Therefore W0-RAS V1 uses W0 retention only.

## Mechanistic interpretation

The selected candidate is:

> fine-detail survivability after deterministic Haar + VisuShrink processing

The V1 hypothesis is narrower than "high frequency helps". It asks whether a raw
GAP embedding benefits from being trained to encode one scalar describing how much
level-1 HH detail survives W0 denoising.

## Frozen next experiment

See:

docs/protocols/W0_RAS_V1.md

The treatment uses one training-only linear regression head on the ordinary raw-RGB
GAP embedding and a fixed auxiliary loss weight of 0.05. The head is discarded at
inference.
