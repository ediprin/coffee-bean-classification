# Coffee17 W0-HBP Matched V1

## Question

Does the frozen W0 wavelet preprocessing improve the established HBP representation on the Coffee17 preprocessing-study protocol?

## Design

Matched comparison on the exact same five development folds, seed 42:

- Control: `R0 -> MobileNetV3-Large -> HBP -> Linear17`
- Candidate: `W0 -> MobileNetV3-Large -> HBP -> Linear17`

Both arms are retrained in this experiment. This is intentional reconstruction of the matched control because the exact prior per-fold HBP control artifacts are unavailable.

## Why this is not a duplicate method experiment

Completed work already includes:

- R0/W0 preprocessing study with GAP;
- R0-HBP in other protocols;
- HBP+MVFD where W0 is a training-only auxiliary view;
- W0-RAS scalar transfer;
- W0-AT spatial transfer.

No completed experiment is the direct five-fold matched comparison of standalone W0-HBP against a simultaneously reconstructed R0-HBP control under the Coffee17 preprocessing-study folds.

Retraining R0-HBP here is a control reconstruction, not a new model proposal.

## Locked variables

Both arms share:

- Coffee17 clean identities and the same fold;
- seed 42;
- MobileNetV3-Large;
- HBP out indices [1, 3, 4];
- projection dim 512;
- linear classifier;
- image size 224;
- batch size 32;
- preprocessing-study rotation schedule [0,45,90,135,180,225,270];
- 50 epochs;
- AdamW, lr 3e-4, weight decay 1e-4;
- cross entropy with label smoothing 0.1;
- cosine scheduler;
- no EMA;
- checkpoint selection by validation Macro-F1.

The only experimental factor is input preprocessing.

## R0

Raw RGB.

## W0

Frozen preprocessing-study definition:

- channel-wise RGB Haar DWT;
- 4 levels;
- finest-detail sigma estimate median(abs(detail))/0.6745;
- universal VisuShrink threshold;
- soft thresholding of detail coefficients;
- inverse DWT reconstruction.

No W0 parameter tuning is authorized.

## Pairing integrity

For every fold the runner verifies:

- same validation identity/label hash;
- same validation row order;
- same seed;
- same initial HBP model-state fingerprint;
- same model and training hyperparameters.

## Metrics

Primary:

- Macro-F1.

Safety metrics:

- Hard-F1;
- Worst-F1.

Also report Accuracy and Balanced Accuracy.

## Gate

W0-HBP passes only if all hold:

1. mean paired Macro-F1 delta > 0;
2. Macro-F1 delta > 0 in at least 3/5 folds;
3. mean paired Hard-F1 delta >= 0;
4. mean paired Worst-F1 delta >= 0.

If the gate fails, stop. No wavelet-level, threshold, HBP-stage, resolution, or loss search on these reused development folds.

## Boundary

This is post-primary exploratory development on the already-opened Coffee17 development folds. Outer test is not accessed.
