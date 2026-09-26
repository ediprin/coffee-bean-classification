# MVFD-SBN Final Evaluation Plan

## Purpose

Method development is frozen. The next evaluation must not be used to redesign
MVFD-SBN.

The final evaluation question is:

> Does the frozen MVFD-SBN method generalize beyond the validation evidence used
> during method development while preserving one-model raw-RGB inference?

## Existing boundary

The original preprocessing study already opened its outer test/OOF population.
Fusion, KD, MVCE, SBN, AT-SBN, MVFD-SBN, the causal ablation, and CA-MVFD-SBN
were developed after that evidence had been observed.

Therefore the previously opened Coffee17 outer population cannot be presented
as a new untouched confirmatory test for MVFD-SBN.

## Preferred confirmation hierarchy

### Option A — new independent same-task dataset

Strongest design:
- collect or obtain images not used anywhere in the current project;
- use a taxonomy that can be mapped to the same 17-class prediction task
  without post-hoc relabeling;
- freeze the mapping before model evaluation;
- evaluate R0_CONTROL and frozen MVFD-SBN once;
- report Macro-F1, Balanced Accuracy, Accuracy, Hard-F1, Worst-F1, confusion
  matrix, and per-class F1;
- do not tune the method after observing the result.

This gives the cleanest external confirmation.

### Option B — external public dataset with defensible label overlap

Use only if:
- the dataset is independently sourced;
- label semantics can be mapped before evaluation;
- the mapping does not selectively remove inconvenient classes;
- the result is presented as cross-dataset/generalization evidence rather than
  a direct replacement for the Coffee17 17-class test.

If the external taxonomy is incompatible, do not force a mapping.

### Option C — no genuinely independent compatible dataset available

If neither A nor B is feasible, keep the Coffee17 MVFD-SBN result explicitly
as post-primary exploratory validation evidence.

In that case the thesis should emphasize:
- protocol transparency;
- matched R0 control;
- causal feature-distillation ablation;
- zero inference overhead;
- failure analyses;
- the absence of a clean independent confirmation as a limitation.

Do not manufacture a new "test" by repartitioning data after seeing the current
results.

## Frozen items

The following may not change during final confirmation:
- MobileNetV3-Large backbone
- GAP + linear 17-class primary head
- C0/F0/W0 definitions
- Selective BatchNorm behavior
- equal-mean teacher
- auxiliary CE weight = 0.05 per view
- feature loss coefficient = 0.007
- 50 epochs
- AdamW lr = 0.0003
- weight decay = 0.0001
- cosine scheduler
- label smoothing = 0.1
- checkpoint selection rule = R0 validation Macro-F1 only
- raw-RGB-only inference path

## Metrics

Primary:
- Macro-F1

Required supporting:
- Balanced Accuracy
- Accuracy
- Hard-F1
- Worst-F1
- per-class F1
- confusion matrix

Efficiency:
- parameter count
- model size
- R0 batch-1 inference latency
- extra inference parameters = 0
- extra inference preprocessing branches = 0

## Decision language

If independent evaluation is positive, report that the frozen method
generalizes on that independent population.

If it is neutral or negative, report the result as-is. Do not reopen the
current validation folds to retune teacher weights, losses, or architecture.

A negative confirmation does not invalidate the mechanistic ablation; it limits
the generalization claim.
