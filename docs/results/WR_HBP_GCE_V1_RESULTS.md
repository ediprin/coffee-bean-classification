# WR-HBP + GCE-LS V1 — Results

Status: **FAIL / STOP**.

Source package SHA256:

`9fbcb438deeb45559dac0b6e881aa729f7819c83a9ea6bd238d1fbcf78fd589a`

The package contains five strict-deterministic matched folds comparing the
same WR-HBP architecture and initialization under ordinary CE versus top-2
GCE-LS. The outer test was not accessed.

## Aggregate results

| Metric | WR-HBP + CE | WR-HBP + GCE-LS | Delta |
|---|---:|---:|---:|
| Accuracy | 91.7526% | 91.1340% | -0.6186 pp |
| Balanced accuracy | 91.9146% | 90.9062% | -1.0084 pp |
| Macro-F1 | 91.7669% | 90.7066% | -1.0603 pp |
| Hard-F1 | 86.4050% | 85.6004% | -0.8046 pp |
| Worst-F1 | 66.5967% | 63.0000% | -3.5967 pp |

Macro-F1 delta was negative in **5/5 folds**. Hard-F1 improved in only
**1/5 folds**. Paired predictions: 5 rescues, 8 damages, 437 both-correct,
35 both-wrong; net correct = -3.

Targeted confusion deltas:

- Full Sour <-> Partial Sour: +1 error;
- Full Black <-> Partial Black: 0;
- Severe Insect Damage <-> Slight Insect Damage: +2 errors;
- total targeted confusion: **+3**.

Largest mean per-class regression was **Fade: -11.81 pp F1**. The insect
hard-group mean fell by **-2.74 pp F1**.

## Decision

All frozen screening gates do not pass. GCE-LS is stopped. No k sweep,
label-smoothing sweep, loss mixing, or rescue tuning is authorized on the
reused development folds.

The result falsifies the specific hypothesis that restricting the WR-HBP
classification normalization to the two highest negative logits improves
Coffee17 hard-class discrimination under this recipe.
