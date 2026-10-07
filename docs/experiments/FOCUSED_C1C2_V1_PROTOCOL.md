# Coffee17 Focused C1/C2 Screen V1

## Scope

This stage trains only C1 (late FRSA + GAP) and C2 (late full FReCSA + GAP)
across the same five Coffee17 development folds used by the master screening.

The B0 anchor is **not retrained**. It is reused only after an explicit
historical-anchor audit passes for every fold.

## Why historical B0 reuse is allowed here

The master B0 was trained at scientific commit:

`b1ac6ca548ebb335d69066377d6f1a1fbb56df36`

The current branch has not changed the B0 model implementation, preprocessing
loader, or core training engine since that commit. The focused runner still
does not trust this fact implicitly: it verifies the historical run contract and
the current runtime before each candidate fold.

For each fold, reuse is allowed only if all of the following pass:

1. historical protocol is `coffee17-master-representation-screening-v1`;
2. candidate/anchor are both B0, seed is 42, and outer test was not accessed;
3. validation count and validation identity-label SHA256 equal the newly
   reconstructed fold;
4. the common data/model/training/adaptation contract matches, ignoring only
   runtime paths, worker count, output path, and candidate-only new sections;
5. the current seed-42 B0 shared-core SHA256 exactly equals the historical B0
   shared-core SHA256;
6. historical validation predictions are present.

If any check fails, the runner aborts. It never silently falls back to training a
new B0 inside this stage.

## Candidates

- C1: MobileNetV3-Large + late FReCSA spatial branch (FRSA) + GAP.
- C2: MobileNetV3-Large + late full FReCSA + GAP.

Both are paired with B0. The candidate-only frequency module is created after
the shared encoder/pool/classifier and the RNG state is restored after module
construction, preserving the matched B0 initialization stream.

## Frozen training protocol

The shared protocol is identical to the master screen:

- five grouped Coffee17 development folds;
- seed 42;
- 224x224;
- batch size 32;
- deterministic per-epoch rotation selection from
  [0, 45, 90, 135, 180, 225, 270];
- MobileNetV3-Large ImageNet initialization;
- AdamW, lr 3e-4, weight decay 1e-4;
- 50 epochs, cosine scheduler;
- cross entropy, label smoothing 0.1;
- no EMA;
- no object crop;
- seeded matched mode; strict deterministic CUDA algorithms remain off;
- outer test remains locked.

## Outputs and decision

Each fold stores C1/C2 metrics, predictions, probabilities, confusion matrix,
history, config, run contract, and the verified historical-B0 audit. The compact
analysis package excludes model checkpoints.

Primary decision metric: five-fold mean Macro-F1 delta vs verified historical
B0, with positive-fold count.

Secondary evidence: Hard-F1, Worst-F1, paired rescue/damage/net-correct, and
the already-measured efficiency preflight.

C1/C2 do not replace W3 merely by beating B0. To become the lightweight winner,
they must be evaluated on the accuracy-efficiency Pareto frontier against the
existing W3 FDA-GAP result.
