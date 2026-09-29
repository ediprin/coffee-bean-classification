# Coffee17 DCL-Style Local Learning V1 — Results

Date recorded: 2026-09-29

Source package SHA-256:
`e5b799bd85a7046e310622c2521937406b70388f4cc1604ba11fb568052dd4c1`

Source summary SHA-256:
`eca2179ab56c2c281347fc93c2473b5ceb0cbd6beb2a7b29e980cc3338c728aa`

Scientific code used by all five folds:
`4dfacc01de28191c7bb11a00353a6bfc303adf5f`

Protocol:
- five locked Coffee17 development folds;
- matched raw-RGB HBP control;
- DCL training-only local RCM + original/shuffled classifier + location reconstruction;
- identical HBP inference architecture;
- strict deterministic CUDA smoke passed on all folds;
- no legacy control reuse;
- outer test untouched.

## Aggregate result

| Metric | HBP-CE | HBP-DCL | Delta DCL-CE |
|---|---:|---:|---:|
| Accuracy | 91.55% | 85.36% | -6.19 pp |
| Balanced Accuracy | 91.72% | 84.53% | -7.18 pp |
| Macro-F1 | 91.47% | 83.31% | -8.17 pp |
| Hard-F1 | 87.39% | 75.12% | -12.27 pp |
| Worst-F1 | 65.64% | 18.00% | -47.64 pp |

Macro-F1 improved in 0/5 folds.
Hard-F1 improved in 0/5 folds.

Paired predictions over 485 validation observations:
- rescue: 11
- damage: 41
- both correct: 403
- both wrong: 30
- net correct: -30

## Preregistered hard-pair confusion

| Pair | HBP-CE | HBP-DCL | Delta |
|---|---:|---:|---:|
| Withered <-> Immature | 4 | 13 | +9 |
| Severe Insect <-> Slight Insect | 8 | 10 | +2 |
| Cut <-> Slight Insect | 2 | 1 | -1 |
| Partial Sour <-> Full Sour | 6 | 16 | +10 |
| Slight Insect <-> Fade | 5 | 3 | -2 |
| Full Black <-> Partial Black | 0 | 0 | 0 |
| **Total** | **25** | **43** | **+18** |

Frozen decision gate:

`NO_CLEAR_DCL_SUPPORT`

Decision:
> **STOP DCL-style destruction/construction on these reused development folds.**
> Do not tune grid size, loss weights, or destructive-local variants after this
> result.

Interpretation:
> The tested local destruction/construction objective substantially degraded
> the Coffee17 classifier, especially the hard classes. This rejects this DCL
> implementation; it does not by itself reject every possible local-evidence
> mechanism.
