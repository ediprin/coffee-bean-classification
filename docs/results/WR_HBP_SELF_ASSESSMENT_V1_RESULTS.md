# WR-HBP Self-Assessment V1 — Result and Implementation Audit

Status: **screening FAIL; intended localization mechanism was not instantiated correctly**.

Five strict-deterministic development folds were completed. Outer test was not accessed.

## Aggregate result

| Metric | WR-HBP base | SAR V1 | Delta |
|---|---:|---:|---:|
| Accuracy | 91.7526% | 91.9588% | +0.2062 pp |
| Balanced accuracy | 91.9146% | 92.1499% | +0.2353 pp |
| Macro-F1 | 91.7669% | 91.9668% | +0.1999 pp |
| Hard-F1 | 86.4050% | 86.8299% | +0.4248 pp |
| Worst-F1 | 66.5967% | 66.5967% | 0.0000 pp |

Only fold 5 changed a prediction: 1 rescue, 0 damage. Macro-F1 and Hard-F1
improved in only 1/5 folds, so the frozen gate failed.

The rescued sample was Partial Sour_40.jpg:
Full Sour -> Partial Sour. Its base true-class margin moved only from
-0.000671 to +0.000711.

## Critical implementation finding

V1 normalized both spatial and class vectors and then computed:

```
cosine_similarity / sqrt(128)
```

The resulting validation attention entropy was about 5.27809 nats on every
fold. A 14x14 map has 196 locations, whose uniform-distribution entropy is:

```
ln(196) = 5.278114659...
```

The mean entropy gap from perfect uniformity was only about 2e-5 nats.
Therefore the "localized" attention was effectively uniform spatial averaging.

The mean spread between the five candidate residual logits was only about
0.006--0.009 logits, and all selected best checkpoints were epoch 1.

Thus V1 is valid evidence that this exact implementation did not help, but it
is **not valid evidence that localized class-conditional reassessment itself
fails on Coffee17**.

## Correction boundary

V2 changes only the mathematically incorrect normalized-cosine scaling:

```
V1: cosine / sqrt(d)
V2: cosine * sqrt(d)
```

For unit-normalized d-dimensional vectors, random cosine similarity has scale
approximately 1/sqrt(d). Multiplying by sqrt(d) restores order-one attention
logits; dividing by sqrt(d) suppresses them again.

No top-k, feature stage, hidden size, epoch budget, optimizer, loss, base model,
or screening gate is changed.
