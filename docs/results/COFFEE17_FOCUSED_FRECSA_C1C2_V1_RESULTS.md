# Coffee17 Focused FReCSA C1/C2 V1 — Results

Status: **C2 KEEP / PRIMARY EFFICIENT CANDIDATE; C1 FAIL_STOP**

Date recorded: 2026-10-11

## 1. Scope and provenance

This result closes the focused C1/C2 development screen registered in:

- `docs/experiments/FOCUSED_C1C2_V1_PROTOCOL.md`
- `docs/experiments/FOCUSED_FREQUENCY_EFFICIENCY_V1_PROTOCOL.md`

Scientific code commit used by all focused folds:

`6653a50e296f7b38591100b395f1a09dbaf68c56`

Historical B0 anchor scientific commit:

`b1ac6ca548ebb335d69066377d6f1a1fbb56df36`

Source compact analysis package SHA-256:

`70365b4661d0a9b8509d68cd3ccfe58ac524dfbf54ed93f20512a44feca72a32`

Efficiency-preflight JSON SHA-256:

`8c56b659455fb9a5f65aac550b2d90ec7e3a26c412627728456a5c55eccbe621`

The focused package reports:

- seed 42;
- five frozen Coffee17 development folds;
- B0 historical anchor verified fold-by-fold and **not retrained**;
- C1 and C2 only were trained;
- no model checkpoints were included in the compact package;
- `outer_test_accessed=false`.

This is development-screening evidence, not final outer-test evidence.

## 2. Candidate definitions

- **B0**: MobileNetV3-Large + GAP.
- **C1**: MobileNetV3-Large + late FReCSA spatial branch (FRSA) + GAP.
- **C2**: MobileNetV3-Large + late full FReCSA channel-spatial recalibration + GAP.

The FReCSA adaptation is applied only to the deepest selected MobileNetV3
feature. It is a paper-grounded MobileNetV3 adaptation, not a literal
block-by-block reproduction of the source ResNet architecture.

Frozen FReCSA settings:

- predefined high-pass response;
- `AvgPool2d(kernel=7, stride=1, padding=3)`;
- local-local interaction;
- published fixed bias `b=0.5`;
- simplified channel branch: GAP -> BN -> independent channel scaling -> sigmoid.

The published paper/source-code bias discrepancy remains documented in
`FOCUSED_FREQUENCY_EFFICIENCY_V1_PROTOCOL.md`.

## 3. Efficiency preflight

All six profiled candidates completed real CUDA forward/backward preflight.

The profiler used 30 warm-up iterations and 100 measured batch-1 iterations.
Its operation counter covers Conv2d + Linear MACs only; measured latency and
throughput are therefore the primary runtime evidence for custom frequency and
elementwise operations.

| Candidate | Params | State MB | Conv/Linear MACs (G) | Median ms | p95 ms | Throughput img/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 2,988,289 | 11.493 | 0.214097 | 5.590 | 7.770 | 178.88 | 38.13 |
| H384 | 3,414,721 | 13.128 | 0.269495 | 6.331 | 7.451 | 157.94 | 41.93 |
| C1 | 2,990,209 | 11.508 | 0.214097 | 5.613 | 5.939 | 178.16 | 37.46 |
| C2 | 2,993,089 | 11.526 | 0.214097 | 5.700 | 6.114 | 175.44 | 37.48 |
| C3 FDA-HBP384 | 3,418,601 | 13.143 | 0.269556 | 7.243 | 8.260 | 138.06 | 42.00 |
| C4 FReCSA-HBP384 | 3,419,521 | 13.161 | 0.269495 | 6.516 | 7.098 | 153.46 | 41.96 |

### Efficiency interpretation

C1 versus B0:

- +1,920 parameters = +0.064%;
- median latency +0.41%;
- throughput -0.40%.

C2 versus B0:

- +4,800 parameters = +0.161%;
- median latency +1.96%;
- throughput -1.92%.

Therefore both C1 and C2 passed the lightweight-efficiency preflight.

A separate important finding is that **FDA is parameter-efficient but not
necessarily runtime-efficient**. C3 changes the counted Conv/Linear MACs only
from 0.269495G to 0.269556G (+0.023%), but median latency rises from 6.331 ms
to 7.243 ms (+14.4%). This is why final efficiency claims must include measured
runtime rather than Params/FLOPs alone.

The small peak-memory differences between B0/C1/C2 are treated as practically
unchanged and are not used to claim memory savings. Likewise, single-sequence
p95 differences are descriptive only; final latency claims require a dedicated
repeated/interleaved benchmark.

## 4. Five-fold primary result

| Metric | Historical B0 | C1 FRSA-GAP | Delta C1-B0 | C2 FReCSA-GAP | Delta C2-B0 |
|---|---:|---:|---:|---:|---:|
| Accuracy | 90.10% | 90.10% | +0.00 pp | **91.55%** | **+1.44 pp** |
| Balanced Accuracy | 89.92% | 89.81% | -0.12 pp | **91.49%** | **+1.57 pp** |
| Macro-F1 | 89.74% | 89.68% | -0.07 pp | **91.36%** | **+1.62 pp** |
| Hard-F1 | 85.20% | 85.52% | +0.32 pp | **87.38%** | **+2.18 pp** |
| Worst-F1 | 66.15% | 60.91% | -5.24 pp | **70.72%** | **+4.57 pp** |

Parameter counts:

- B0: 2,988,289;
- C1: 2,990,209;
- C2: 2,993,089.

### Fold-consistency

C1 Macro-F1 delta was positive in 3/5 folds, but its mean Macro-F1 was slightly
negative and Worst-F1 decreased on all five folds. C1 does not justify retaining
the spatial branch alone.

C2:

- Macro-F1 positive in **5/5 folds**;
- Hard-F1 positive in **5/5 folds**;
- Balanced Accuracy positive in 4/5 folds;
- Accuracy positive in 3/5 folds;
- Worst-F1 positive in 3/5 folds.

C2 Macro-F1 deltas by fold:

| Fold | C2 Macro-F1 | Delta vs B0 |
|---:|---:|---:|
| 1 | 93.165% | +2.348 pp |
| 2 | 93.658% | +0.015 pp |
| 3 | 92.557% | +1.285 pp |
| 4 | 88.294% | +4.269 pp |
| 5 | 89.150% | +0.189 pp |

Although folds 2 and 5 have very small positive Macro-F1 deltas, the sign is
positive on every fold.

### Paired top-1 outcomes

Across the 485 fold-validation rows:

C1 versus B0:

- rescue: 7;
- damage: 7;
- both correct: 430;
- both wrong: 41;
- net: 0.

C2 versus B0:

- rescue: 13;
- damage: 6;
- both correct: 431;
- both wrong: 35;
- net: **+7 correct**.

The 485 rows must **not** be treated as 485 independent images: across the five
validation folds there are 394 unique class/filename identities. Paired counts
are descriptive, not an independent-sample significance test.

## 5. Comparison with existing master-screening leaders

| Model | Macro-F1 | Hard-F1 | Worst-F1 | Params |
|---|---:|---:|---:|---:|
| B0 GAP | 89.74% | 85.20% | 66.15% | 2.988M |
| W3 FDA-GAP | 90.47% | 85.59% | 68.00% | 2.992M |
| **C2 FReCSA-GAP** | **91.36%** | **87.38%** | **70.72%** | **2.993M** |
| B1 HBP | 91.13% | 85.63% | 60.31% | 3.562M |
| W2 WCA-HBP | 91.43% | 86.59% | 60.89% | 3.678M |
| W4 FDA-HBP | **91.52%** | 85.45% | 61.44% | 3.566M |

### C2 versus W3 lightweight incumbent

- Macro-F1: **+0.893 pp**;
- Hard-F1: **+1.795 pp**;
- Worst-F1: **+2.718 pp**;
- parameters: +920 = +0.031%.

Therefore W3 is no longer the best lightweight accuracy candidate on the
development evidence.

### C2 versus B1 HBP

- Macro-F1: **+0.235 pp**;
- Hard-F1: **+1.756 pp**;
- Worst-F1: **+10.410 pp**;
- parameters: -569,216 = **-15.98%**.

C2 exceeds the HBP baseline on all three representation-quality metrics while
using materially fewer parameters.

### C2 versus W2 WCA-HBP

- Macro-F1: -0.065 pp;
- Hard-F1: **+0.798 pp**;
- Worst-F1: **+9.829 pp**;
- parameters: -684,416 = **-18.61%**.

The Macro difference is effectively tiny, while C2 is substantially lighter
and much stronger on Worst-F1.

### C2 versus W4 FDA-HBP

- Macro-F1: -0.151 pp;
- Hard-F1: **+1.934 pp**;
- Worst-F1: **+9.281 pp**;
- parameters: -573,096 = **-16.07%**.

W4 retains the highest mean Macro-F1 by only 0.151 pp. C2 has the stronger
Hard/Worst profile with substantially lower parameter count. A same-hardware
head-to-head runtime benchmark is still required before making a final
accuracy-latency Pareto claim.

## 6. Class-level effect of C2

Mean five-fold per-class F1 delta versus B0:

| Class | Delta F1 |
|---|---:|
| Slight Insect Damage | **+5.38 pp** |
| Dry Cherry | **+5.00 pp** |
| Fade | **+4.81 pp** |
| Severe Insect Damage | **+4.17 pp** |
| Immature | **+3.27 pp** |
| Withered | **+2.56 pp** |
| Husk | **+1.82 pp** |
| Cut | **+1.71 pp** |
| Fungus Damage | **+1.17 pp** |
| Partial Sour | +0.40 pp |
| Full Black | 0.00 pp |
| Full Sour | 0.00 pp |
| Parchment | 0.00 pp |
| Shell | 0.00 pp |
| Partial Black | -0.02 pp |
| Floater | -0.40 pp |
| Broken | **-2.29 pp** |

Hard-group mean deltas:

- sour/black: +0.13 pp;
- shape/withered: **+2.51 pp**;
- insect damage: **+4.77 pp**.

The strongest improvements occur in several previously difficult fine-grained
classes, especially the insect-damage pair. The main observed class trade-off is
Broken (-2.29 pp), which should be included in later error/XAI analysis.

These are empirical associations from the development folds. They do not by
themselves prove that FReCSA is specifically learning a named physical
color/texture cue.

## 7. Training dynamics

C2 selected best epochs:

| Fold | Best epoch | Best Macro-F1 | Epoch-50 Macro-F1 | Last-minus-best |
|---:|---:|---:|---:|---:|
| 1 | 23 | 93.17% | 91.46% | -1.71 pp |
| 2 | 31 | 93.66% | 92.56% | -1.09 pp |
| 3 | 23 | 92.56% | 88.85% | -3.71 pp |
| 4 | 31 | 88.29% | 84.26% | -4.03 pp |
| 5 | 28 | 89.15% | 85.88% | -3.27 pp |

Every fold had fallen from its selected validation peak by epoch 50. Therefore
best-checkpoint selection is consequential, and simply adding capacity or
training longer is not supported by these curves.

C1 best epochs were 29, 31, 28, 34, and 43 and also showed lower epoch-50
Macro-F1 than its selected peak in every fold.

## 8. Method decision

### C1

Decision:

> **FAIL_STOP C1 / FRSA-only.**

Reason:

- mean Macro-F1 -0.07 pp versus B0;
- Worst-F1 -5.24 pp;
- rescue and damage exactly balanced;
- the spatial high-pass branch alone does not provide a reliable improvement.

No FRSA-only kernel/bias/filter/interactions search is authorized on the reused
development folds.

### C2

Decision:

> **KEEP C2 / FReCSA-GAP as the primary efficient candidate.**

Reason:

- +1.62 pp Macro-F1 versus B0;
- +2.18 pp Hard-F1;
- +4.57 pp Worst-F1;
- Macro and Hard positive in 5/5 folds;
- only +0.161% parameters and +1.96% median latency versus B0 in the focused
  preflight;
- exceeds W3 on Macro/Hard/Worst with essentially the same parameter count;
- exceeds B1 HBP on Macro/Hard/Worst with ~16% fewer parameters;
- remains within 0.151 pp Macro-F1 of W4 while showing much stronger
  Hard/Worst metrics and lower parameter count.

This does **not** authorize arbitrary stacking of FDA + FReCSA + HBP. Reused
development folds have already supported substantial architecture selection.

## 9. Next authorized step

Before any final architecture claim:

1. freeze C2 as the primary efficient model;
2. run a dedicated same-hardware, same-profiler runtime comparison for
   **B0, W3, C2, B1, and W4**;
3. use repeated/interleaved timing rather than a single sequential profile to
   reduce runtime-order noise;
4. compare Params, state size, operation estimate, median/p95 latency,
   throughput, and peak memory;
5. after the architecture is frozen, move to the preregistered untouched outer
   test rather than opening a new module-search loop.

C3/C4 remain untrained at this point. Their preflight measurements are retained
as evidence, but they should not be interpreted as accuracy results.
