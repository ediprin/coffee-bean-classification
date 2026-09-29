# WR-PDR-HBP V1 — Results

Date recorded: 2026-09-29

Source package SHA-256:
`29ff8f7694ddb2e8940fb1b9bd800a712545d057091b86a63d0f14a72df88255`

Source summary SHA-256:
`66ac52fd9620eb6e2fa2b5840e7a3da1e8557b964338c1772d171f4b156236e0`

Protocol:
- strict deterministic five-fold seed-42 matched comparison;
- WR-HBP base versus WR-PDR-HBP;
- outer test untouched.

| Metric | WR-HBP | WR-PDR-HBP | Delta |
|---|---:|---:|---:|
| Accuracy | 91.75% | 90.52% | -1.24 pp |
| Balanced Accuracy | 91.91% | 90.54% | -1.37 pp |
| Macro-F1 | 91.77% | 90.33% | -1.44 pp |
| Hard-F1 | 86.41% | 83.35% | -3.05 pp |
| Worst-F1 | 66.60% | 59.93% | -6.67 pp |

Macro-F1 improved in 0/5 folds.
Paired outcome: 0 rescue, 6 damage, net -6.

Frozen screening gate: **FAIL**.

Decision:
> **STOP WR-PDR-HBP.** The tested physical-descriptor logit residual does not
> improve WR-HBP and must not be reintroduced through weight/descriptor tuning
> on the same development folds.
