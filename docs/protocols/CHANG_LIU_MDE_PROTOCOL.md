# Chang–Liu multiscale defect extraction: controlled Coffee17 protocol

## Source and claim boundary

The candidate is grounded in Chang and Liu, *Multiscale Defect Extraction
Neural Network for Green Coffee Bean Defects Detection*, IEEE Access 2024,
DOI `10.1109/ACCESS.2024.3356596`.  Their network uses parallel standard
`3x3` and `5x5` convolutions to capture fine defect lines and coarser bean
contours.  This repository does **not** claim an exact reproduction: their
self-built 7,300-image dataset is unavailable and their original network uses
64x64 inputs, four custom Inception stages, repeated 5x5 max pooling, and a
flattened classifier.

The implemented model is explicitly named an architectural adaptation:

```text
EfficientNetV2-B0 deep feature
  +-- standard Conv 3x3 -> BN -> ReLU --+
  +-- standard Conv 5x5 -> BN -> ReLU --+-- concat -> Conv 1x1 -> BN
                                                         |
                                       residual add <----+
                                                         |
                                                        GAP -> CE
```

Only the paper's causal multiscale operator is transferred.  The pretrained
backbone, 224x224 input, residual connection, and 1x1 fusion are adaptations.
Multiscale extraction itself is established prior art and is not claimed as a
standalone novelty.

## Frozen models

| Code | Model | Purpose |
|---|---|---|
| BE2G | EfficientNetV2-B0 + GAP + CE | strongest ordinary-pooling baseline |
| BE2H | EfficientNetV2-B0 + HBP + CE | previous second-order reference |
| MDE0 | EfficientNetV2-B0 + pointwise residual + GAP + CE | capacity control |
| MDE1 | EfficientNetV2-B0 + residual 3x3/5x5 MDE + GAP + CE | multiscale candidate |

MDE0 uses a 287-channel pointwise bottleneck.  MDE1 uses 16 channels per
spatial branch.  Their total trainable parameter difference must be below
0.1%.  MDE0 has no spatial convolution, so it controls added capacity without
providing a larger receptive field.

All training settings match the BE2G protocol: ImageNet initialization,
224x224 input, online train augmentation, cross-entropy with label smoothing
0.1, Adam at `3e-4`, cosine schedule, weight decay `1e-4`, 50 epochs, and the
same grouped Coffee17 fold.

## Fail-fast gate

The first screening is locked to seed 42 and `source/val`.  Coffee17 test is
not opened.  MDE1 proceeds to seeds 123 and 2026 only if both comparisons below
pass:

1. `BE2G_vs_MDE1`: isolates benefit over ordinary GAP.
2. `MDE0_vs_MDE1`: isolates spatial multiscale benefit from parameter capacity.

For each comparison:

- mean Macro-F1 must increase;
- mean Hard-F1 must increase;
- Worst-F1 may not decrease by more than one percentage point.

BE2H is reported as context but is not part of the causal gate because the
candidate is a GAP model.  A screening PASS is not a final result; it only
authorizes validation confirmation with the remaining two seeds.

## Command

```bash
python -u -m bilinear_lmmd.experiments.run_multiscale_defect_screening \
  --data-root /content/coffee17-mde-data/clean/folds/fold_1 \
  --baseline-root /content/backbone-results \
  --output-root /content/drive/MyDrive/coffee17-chang-liu-mde \
  --seeds 42 \
  --evaluation-split val
```

The Colab notebook is
`notebooks/coffee17_chang_liu_mde_colab.ipynb`.  It restores BE2G/BE2H from
Hugging Face, stores MDE checkpoints directly on Google Drive, prints a
one-minute heartbeat, and resumes incomplete runs after a reset.
