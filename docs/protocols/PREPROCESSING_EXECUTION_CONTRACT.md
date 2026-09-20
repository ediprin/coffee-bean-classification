# Coffee17 preprocessing study — execution contract v1

Status: **implementation draft; freeze the Git commit and runtime before primary training**

Milestone C deliberately does not modify the legacy `engine/train.py`. The new
study uses an isolated source-only runner while reusing the existing model
builder and classification metrics. This prevents preprocessing-study changes
from altering legacy HBP/UDA/OSR experiments.

## Matched data trajectory

For train identity `i` at epoch `e`, the discrete rotation is a deterministic
function of `(seed, epoch, identity)`. The train ordering is independently a
deterministic function of `(seed, epoch)`. Therefore R0/C0/F0/W0 receive the
same identity order and same rotation schedule for a matched fold/seed, even
when they are executed in different Colab accounts or resumed after reset.

## Runtime frontend

Dataset output is RGB float `[0,1]` after rotation and square resize.

- R0: identity -> GPU -> ImageNet normalize.
- C0: exact OpenCV LAB-L CLAHE on CPU -> GPU -> normalize.
- F0: GPU -> canonical AF2 -> normalize.
- W0: GPU -> canonical WAV1 -> normalize.

The outer test is absent from every development runtime.

## Resume

`last.pt` stores optimizer, scheduler, history, best score, RNG state,
run-contract hash and the frozen initial-model fingerprint. Resume is rejected
when the run-contract differs.

## Environment

One setup runtime is explicitly frozen to
`runtime_environment.json` plus an exact package lock. Every arm verifies the
same software fingerprint before training. GPU model name is recorded but is
not part of the software fingerprint.

## Smoke vs primary

Smoke runs use 1-3 epochs, are stored under `smoke/`, and have
`scientific_evidence=false`. Primary runs use the full 50-epoch contract and
are stored under `primary/`.

No validation result is allowed to redefine R0/C0/F0/W0 after the primary
contract is frozen.
