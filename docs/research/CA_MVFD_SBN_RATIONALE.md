# Why confidence-aware aggregation is the next controlled MVFD-SBN test

## Evidence from the frozen Coffee17 experiments

The equal-mean MVFD-SBN teacher is already supported by a causal ablation:
removing only the feature-distillation gradient reduces mean validation
Macro-F1 from 91.77% to 89.94%.

However, the transformed views are not uniformly useful.

The original preprocessing study showed strong class-dependent effects. Examples:
- W0 strongly helps Fade but hurts Partial Black and Withered.
- F0 helps Dry Cherry, Immature and Husk but hurts Withered and Severe Insect
  Damage.
- C0 helps Dry Cherry and Partial Sour but hurts Fade and Slight Insect Damage.

The frozen MVFD-SBN class deltas do not simply reproduce all of those
view-specific advantages. Classes with at least one previously beneficial
transformed view can still be neutral or negative under equal-mean MVFD,
including Cut, Full Sour, Immature, Partial Black and Slight Insect Damage.

This comparison spans different experimental protocols and therefore is not
proof that averaging destroyed specific rescue cases. The compact MVFD package
does not contain per-sample C0/F0/W0 embeddings, so the exact counterfactual
cannot be reconstructed without checkpoints or a rerun.

It is nevertheless sufficient to motivate one controlled aggregation test,
because the broader fact -- heterogeneous view quality -- is already directly
established.

## Literature basis

Dong et al. (Neurocomputing 2026) explicitly motivate Reliability-Guided
Aggregation by noting that simple averaging may dilute discriminative cues
when different views have different reliability.

Zhang, Chen and Wang (ICASSP 2022), Confidence-Aware Multi-Teacher Knowledge
Distillation (CA-MKD), show a closely related failure mode: equal aggregation
of multiple teachers can be misled by low-quality teacher predictions. Their
sample-wise teacher weights use ground-truth cross-entropy, and they extend
confidence-aware weighting to intermediate feature distillation.

## Why not full RGA

A learned reliability MLP would add a new trainable module and additional
degrees of freedom. That is unnecessary for the first test and would make the
mechanistic interpretation less clean.

CA-MVFD-SBN therefore uses a parameter-free, source-inspired sample-wise
confidence rule and changes only the aggregation of the existing C0/F0/W0
teacher features.

## Frozen comparison

Equal-mean MVFD-SBN:

t_equal = stopgrad((e_C + e_F + e_W) / 3)

CA-MVFD-SBN:

CE_v = CE(z_v, y)

beta_v =
  1/(K-1) * [1 - exp(CE_v) / sum_u exp(CE_u)]

t_CA = stopgrad(sum_v beta_v e_v)

The remaining objective is unchanged:

L =
  CE_R
  + 0.05 (CE_C + CE_F + CE_W)
  + 0.007 ||e_R - t||^2

Primary comparison:
CA-MVFD-SBN vs the already frozen equal-mean MVFD-SBN.

No sweep is planned. If CA-MVFD-SBN does not consistently improve over the
frozen MVFD result, equal-mean MVFD remains the preferred thesis method and
reliability weighting should not be escalated further.
