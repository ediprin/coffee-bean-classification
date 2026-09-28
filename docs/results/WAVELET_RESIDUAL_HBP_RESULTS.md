# WR-HBP V1 Results — PASS

Date analyzed: **2026-09-28**

Source artifact:

- `wavelet-residual-hbp-analysis-package.zip`
- SHA-256: `3fff1a877e22787b2135810afd39ad490a22e2eb43bb83d336ba4ade6c922ee9`

Scientific commit used by all five folds:

`01c9212965bc9040ef151204b9404d564f523a0f`

Protocol:

`coffee17-wavelet-residual-hbp-v1`

## Integrity

All five matched folds completed with seed 42.

For every fold:

- R0-HBP and WR-HBP used the same validation identities/labels;
- the shared RGB-HBP core had the same initial SHA-256:
  `6d425c6c149005afde1224ed74ccfe4eb84c95e574d21e25a0e728578d79f9cc`;
- initial control/candidate logits were exactly equal before training;
- R0-HBP was retrained as the matched control;
- WR-HBP was trained as the candidate;
- outer test was not accessed.

## Aggregate result

| Metric | R0-HBP | WR-HBP | Delta WR-R0 |
|---|---:|---:|---:|
| Accuracy | 91.34% ± 2.37 | 91.75% ± 2.82 | **+0.41 ± 0.56 pp** |
| Balanced Accuracy | 91.45% ± 2.82 | 91.93% ± 3.11 | **+0.48 ± 0.50 pp** |
| Macro-F1 | 91.02% ± 2.66 | **91.77% ± 2.74** | **+0.75 ± 0.46 pp** |
| Hard-F1 | 86.12% ± 5.07 | **86.24% ± 5.35** | **+0.11 ± 1.34 pp** |
| Worst-F1 | 65.33% ± 8.00 | **66.76% ± 9.55** | **+1.42 ± 2.88 pp** |

## Per-fold Macro-F1

| Fold | R0-HBP | WR-HBP | Delta |
|---|---:|---:|---:|
| 1 | 92.23% | 92.23% | **+0.006 pp** |
| 2 | 92.77% | 93.90% | **+1.13 pp** |
| 3 | 93.44% | 94.55% | **+1.11 pp** |
| 4 | 87.01% | 87.84% | **+0.83 pp** |
| 5 | 89.66% | 90.31% | **+0.66 pp** |

Macro-F1 improved in **5/5 folds**.

Positive-delta folds:

- Accuracy: 2/5;
- Balanced Accuracy: 3/5;
- Macro-F1: **5/5**;
- Hard-F1: 3/5;
- Worst-F1: 2/5, with two ties.

## Paired prediction outcomes

Across 485 paired validation observations:

- WR-HBP rescue: **7**;
- WR-HBP damage: **5**;
- both correct: 438;
- both wrong: 35;
- net: **+2 correct predictions**.

## Hard-group behavior

Mean WR-HBP minus R0-HBP:

- sour/black: **-1.18 pp**;
- shape/withered: **+1.32 pp**;
- insect damage: **+0.24 pp**.

The aggregate Hard-F1 improvement is small. Shape/withered and insect damage
improve on mean, while sour/black decreases.

## Per-class behavior

Largest mean gains:

- Fade: **+8.62 pp**;
- Withered: **+5.61 pp**;
- Parchment: +1.82 pp;
- Fungus Damage: +1.32 pp.

Largest mean losses:

- Full Sour: **-2.06 pp**;
- Cut: **-1.79 pp**;
- Partial Sour: **-1.47 pp**.

Several classes have zero mean change because control/candidate predictions were
identical for those classes on the small validation folds.

## Learned wavelet gate

Selected-checkpoint `tanh(alpha)` values:

- fold 1: +0.1362;
- fold 2: +0.1569;
- fold 3: +0.1544;
- fold 4: +0.1571;
- fold 5: -0.1280.

Mean: **+0.0953 ± 0.1252**.

The branch therefore did not remain at its zero initialization. Four folds
learned a similar positive contribution; fold 5 learned the opposite sign.

## Efficiency

Trainable parameter count:

- R0-HBP: 3,562,305;
- WR-HBP: 3,563,202;
- overhead: **897 parameters = 0.0252%**.

The additional representational cost is therefore negligible in parameter
count. End-to-end latency including the Haar/VisuShrink side branch still needs
to be measured separately before an efficiency claim.

## Frozen screening gate

Required:

1. mean Macro-F1 delta > 0;
2. Macro-F1 positive in at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0.

Observed:

- mean Macro-F1 delta: **+0.746 pp — PASS**;
- Macro positive folds: **5/5 — PASS**;
- mean Hard-F1 delta: **+0.114 pp — PASS**;
- mean Worst-F1 delta: **+1.424 pp — PASS**.

Final development gate:

> **PASS**

## Scientific conclusion

WR-HBP is the first preprocessing/HBP combination in the current development
chain to pass the complete frozen gate while preserving raw RGB as the main
representation.

The result supports the specific mechanism tested here:

> explicit L1 luminance Haar/VisuShrink detail can provide useful residual
> information to the shallow RGB feature before HBP without replacing RGB or
> forcing RGB to imitate a transformed representation.

The strongest evidence is the **5/5 positive Macro-F1 direction**, the positive
mean Hard/Worst deltas, and the extremely small parameter overhead.

Claim boundary:

- this is still post-primary exploratory development;
- the folds were already reused during method development;
- only seed 42 was used here;
- outer test remains untouched.

Therefore this result is sufficient to **promote WR-HBP past development
screening**, not sufficient by itself for an independent final-confirmatory
superiority claim.



## Additional post-hoc diagnostic: what WR-HBP is actually changing

A deeper paired-probability analysis of the uploaded five-fold package shows
that the gain is broader than the +2 net top-1 decisions alone suggest.

Across all 485 paired validation observations:

- mean true-class probability delta (WR-HBP - R0-HBP): **+1.06 pp**;
- median true-class probability delta: **+0.58 pp**;
- true-class probability increased on **276/485** observations;
- mean true-class-vs-best-rival margin delta: **+1.57 pp**;
- the true-class margin increased on **281/485** observations.

This indicates that WR-HBP often strengthens the correct class evidence without
necessarily changing the final argmax.

### Classes with the strongest mean true-class probability increase

- Withered: **+3.40 pp**; margin **+5.81 pp**;
- Fungus Damage: **+3.25 pp**; margin **+4.56 pp**;
- Fade: **+2.66 pp**; margin **+5.65 pp**;
- Slight Insect Damage: **+2.12 pp**; margin **+3.54 pp**;
- Full Sour: **+1.86 pp**; margin **+3.33 pp**.

### Main remaining weakness

Partial Sour is the clearest outlier:

- mean true-class probability delta: **-2.02 pp**;
- mean true-class margin delta: **-4.67 pp**.

The changed-decision audit shows two cases where an originally correct
Partial-Sour prediction moved to Full Sour, while one Full-Sour error was
corrected from Partial Sour to Full Sour.

Therefore the remaining sour-group loss is best interpreted as a **within-group
decision-boundary shift between Partial Sour and Full Sour**, rather than a
general absence of useful wavelet signal for sour beans.

### Wavelet-gate stability

The selected-checkpoint gate signs are not directly physically interpretable
because the learned projection branch can absorb a sign flip. The more useful
quantity is gate magnitude.

Absolute selected gate values lie in the narrow range **0.128-0.157**, with:

- mean absolute gate: **0.1465**;
- sample standard deviation: **0.0135**.

Thus all five folds learned a non-trivial residual contribution of similar
magnitude even though fold 5 used the opposite scalar sign.

### Implication for the next method-development hypothesis

The evidence now points toward **selective complementarity**, not simply a
stronger wavelet branch.

WR-HBP already improves the texture/edge-sensitive classes most clearly, while
the remaining weakness is concentrated in a small sour-class boundary. Any
next extension should preserve the successful WR-HBP path and add one
orthogonal low-frequency/color-contrast cue rather than increasing wavelet
strength or adding generic attention.
