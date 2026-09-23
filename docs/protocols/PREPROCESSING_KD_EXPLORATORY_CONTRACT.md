# Coffee17 preprocessing KD — exploratory contract v1

## Scope

This protocol is deliberately **post-primary exploratory**. The original
R0/C0/F0/W0 preprocessing study has already opened its frozen outer OOF
population before this KD design was proposed. Therefore no result produced by
this protocol may be described as a new confirmatory outer-test result.

The purpose is narrower: test whether complementary knowledge from the frozen
preprocessing teachers can be transferred into a single RGB student while
keeping deployment cost equal to one MobileNetV3-Large + GAP model.

## Frozen teachers

For each fold, the only allowed teachers are the four frozen primary
checkpoints from the same fold:

- R0 raw RGB;
- C0 Lab-L CLAHE;
- F0 luminance AFAB-2;
- W0 four-level Haar VisuShrink reconstruction.

Teacher checkpoint paths and SHA-256 values are read from
`preprocessing_primary_confirmation.json`. Any hash mismatch aborts the run.

## Student

The student is MobileNetV3-Large + GAP and receives R0/raw RGB only.

The student is constructed immediately after seed initialization and its initial
model-state fingerprint must exactly match the common primary initialization
recorded in the primary authority. Teacher construction is isolated from RNG so
loading four teachers cannot change the matched student trajectory.

## Treatments

Only two KD treatments are defined:

1. `R0`: distill from the frozen R0 teacher.
2. `ALL4`: distill from the uniform probability ensemble R0+C0+F0+W0.

No teacher selection, class-wise weight, learned gate, or validation-derived
ensemble weight is permitted in v1.

For temperature T, ALL4 teacher probabilities are computed as

`q = mean_v softmax(z_v / T)`

and **not** as `softmax(mean_v(z_v) / T)`.

The student loss is

`L = hard_weight * CE + (1-hard_weight) * T^2 * KL(q || p_student_T)`.

The frozen v1 values are:

- temperature: 2.0;
- hard weight: 0.5;
- label smoothing: 0.1.

## Matched training contract

The KD student preserves the original preprocessing-study recipe:

- image size 224;
- deterministic epoch-wise rotations;
- batch size 32;
- AdamW, lr 3e-4, weight decay 1e-4;
- cosine schedule;
- 50 epochs;
- best checkpoint selected by validation Macro-F1.

Validation remains untouched by augmentation.

## Teacher diversity audit

Before KD training, a diagnostic audit may measure teacher disagreement on the
train and validation splits: pairwise top-1 disagreement, Jensen-Shannon
divergence, entropy, oracle correctness, and class-wise disagreement.

The audit is descriptive only. It must not be used to select views, tune
weights, or redefine the two treatments.

## Deployment claim

All teacher models and C0/F0/W0 frontends are training-only. The deployed
student receives raw RGB through the R0 frontend and one MobileNetV3-Large
forward pass. Thus the KD method adds no inference-time branch relative to R0.

## Test boundary

This protocol writes validation evidence only. Any evaluation on the previously
opened Coffee17 outer OOF population is exploratory and must be labeled as such.
A clean confirmatory claim requires a genuinely independent evaluation
population not used to design this KD procedure.
