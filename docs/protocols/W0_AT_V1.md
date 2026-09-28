# Coffee17 W0 Spatial Attention Transfer V1

## Status

Exploratory post-OOF development experiment.

This experiment is a direct follow-up to three completed results:

1. W0 exposes complementary information relative to R0.
2. Full intermediate tensor matching in ML-MVFD-SBN failed badly.
3. Scalar W0 retained-energy auxiliary regression (W0-RAS) failed its screening gate.

The question is therefore narrower:

> Can W0 provide useful spatial guidance if channel identity and absolute feature
> magnitude are discarded, while the deployed model remains raw-RGB only?

## Literature anchor

The attention map and loss follow activation-based Attention Transfer
(Zagoruyko & Komodakis, ICLR 2017).

For feature tensor F in R^(C x H x W):

A(F) = mean_c F_c^2

q(F) = vec(A(F)) / ||vec(A(F))||_2

Attention-transfer loss:

L_AT = mean((q_R - stopgrad(q_W))^2)

The original public Attention Transfer implementation computes
F.normalize(x.pow(2).mean(1).view(batch,-1)) and mean squared difference, and
uses beta=1000 in its documented CIFAR/ImageNet commands. V1 adopts those
mechanics without a Coffee17 beta sweep.

## Why stage index 3

MobileNetV3-Large exposes out_indices=[3,4].

Only feature position 0 (backbone out index 3) receives attention transfer.

This is deliberately the same intermediate stage used by the completed
ML-MVFD-SBN experiment. The purpose is to change the transfer representation,
not to search for a more favorable layer:

- ML-MVFD: full C x H x W tensor MSE at stage 3;
- W0-AT: channel-collapsed, L2-normalized H x W attention at stage 3.

The final stage remains the ordinary GAP input.

## Views

Student/deployment view:

R0 = raw RGB

Teacher training-only view:

W0 = frozen four-level channel-wise Haar + VisuShrink + inverse reconstruction.

R0 and W0 are generated from the same already-augmented image, so geometry is
aligned.

The same MobileNetV3-Large weights process both views.

R0 is the only path allowed to update persistent BN running statistics.

For W0, BatchNorm uses current-view batch statistics while
track_running_stats=False, matching the established Selective-BN contract.

Teacher attention is detached. No gradient is taken through the W0 forward.

## Objective

Classification:

L_CE = CE(logits_R, y)

Attention:

L_AT = mean((q(F_R^3) - stopgrad(q(F_W^3)))^2)

Total:

L = L_CE + 1000 * L_AT

There is:

- no W0 classifier;
- no W0 CE;
- no output KD;
- no final-vector MVFD;
- no full-tensor intermediate MSE;
- no trainable attention module;
- no HBP.

## Loss-scale fail-fast

At epoch 1 compute:

rho = (1000 * mean(L_AT)) / mean(L_CE)

If rho > 1.0, abort as a loss-contract failure.

This gate does not authorize changing beta. A scale failure is a negative result
for this frozen V1 formulation.

## Experimental authority

Use the same Coffee17 development authority as the latest MVFD/HBP-MVFD line:

- 979 original images;
- clean/provenance handling from the repository;
- five deterministic development folds;
- seed 42;
- 97 validation identities per fold;
- validation identity hashes checked against
  configs/shared_multiview/sbn_reference_v1.json;
- initial primary model fingerprint must equal the matched R0 reference;
- 50 epochs;
- AdamW lr 3e-4;
- weight decay 1e-4;
- cosine schedule;
- label smoothing 0.1;
- preprocessing-study train rotations;
- checkpoint selection only by R0 validation Macro-F1;
- outer test not accessed.

## Screening gate

Relative to frozen matched R0 control, all must hold:

1. mean Macro-F1 delta > 0;
2. Macro-F1 improves in at least 3/5 folds;
3. mean Hard-F1 delta >= 0;
4. mean Worst-F1 delta >= 0;
5. loss-scale gate passes every fold.

If the gate fails, stop W0-AT V1. No beta search, alternate stage, or alternate
attention exponent is authorized from these reused development folds.

## Deployment

After training, W0 is discarded.

Inference remains:

raw RGB -> MobileNetV3-Large -> GAP -> Linear17

No additional inference parameters, view, preprocessing, or forward pass.
