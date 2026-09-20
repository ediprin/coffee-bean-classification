# Coffee17 preprocessing study — execution contract v2

Status: **literature-audited candidate; primary training blocked until pre-training gates PASS**

The study keeps the legacy training engine untouched. R0/C0/F0/W0 use an isolated source-only runner while reusing the same MobileNetV3-Large + GAP model builder, metrics, optimizer recipe, seed, folds and initialization.

## Matched data trajectory

For train identity `i` at epoch `e`, discrete rotation is a deterministic function of `(seed, epoch, identity)`. Train order is independently deterministic from `(seed, epoch)`. Thus matched arms receive the same identities, order and rotation schedule.

## Runtime frontend

Dataset output is RGB float after deterministic rotation and resize to 224x224.

- R0: identity -> device -> ImageNet normalize.
- C0: CPU OpenCV CIELAB L-only CLAHE, clip 2.0, 8x8 -> device -> normalize.
- F0: device -> Rec.709 luminance AFAB-2/angular-frequency cue -> one shared RGB residual gate -> normalize.
- W0: device -> four-level channel-wise Haar + VisuShrink soft threshold + inverse DWT -> normalize.

No post-hoc clipping is added to F0/W0. Their observed ranges are recorded before training by the 979-original observability audit.

## Required pre-training evidence

Every training run contract binds:

- dataset/development contract hash;
- static frontend/model preflight hash;
- 979-original preprocessing-observability hash;
- F0 frozen luminance-reference equivalence hash;
- frozen software/runtime hash;
- common model-initialization fingerprint;
- exact arm config and Git commit.

C0 and W0 are literature-derived operators and therefore do not pretend to be bitwise copies of the older detection controls.

## Resume

`last.pt` stores optimizer, scheduler, history, best score, RNG state, run-contract hash and the frozen initial-model fingerprint. Resume is rejected if the run contract changes.

## Primary evidence boundary

Smoke runs, if used, live under `smoke/` and are not scientific evidence. Primary runs are 50 epochs under `primary/`.

Outer test is absent from development runtimes. No validation result may redefine R0/C0/F0/W0 after the protocol is frozen.
