# MVFD-SBN causal ablation v1: AUXCE-SBN control

## Question

MVFD-SBN is the first single-model R0-only method in this chain to show a
non-trivial mean Macro-F1 improvement over the matched R0 control.

However its objective contains two additions relative to R0:

1. low-weight C0/F0/W0 auxiliary classification losses;
2. explicit C0/F0/W0 -> R0 feature distillation.

Therefore the current result does not yet establish that feature distillation
caused the gain.

This ablation removes only the feature-distillation gradient.

## Control objective

Architecture, data, initialization, optimizer, SBN behavior, auxiliary heads,
training views and 50-epoch schedule are kept fixed.

The feature teacher is still computed for diagnostics, but:

lambda_feat = 0

Training objective:

L_control =
  CE_R
  + 0.05 [CE_C + CE_F + CE_W]

No output KD. No classifier merging. No learned routing.

## Comparison

The primary causal comparison is:

MVFD-SBN
vs
AUXCE-SBN control

where MVFD-SBN differs only by:

+ 0.007 * mean_i ||e_R_i - stopgrad(mean(e_C,e_F,e_W))||_2^2

The matched R0 control is retained as a secondary reference.

## Interpretation rule

If AUXCE-SBN reaches essentially the same R0-only performance as MVFD-SBN,
then the gain cannot be attributed specifically to explicit feature
distillation.

If MVFD-SBN consistently exceeds AUXCE-SBN while all other contracts remain
matched, that supports the feature-transfer term as the active contribution.

This remains post-primary exploratory validation evidence and is not an
independent confirmatory test.
