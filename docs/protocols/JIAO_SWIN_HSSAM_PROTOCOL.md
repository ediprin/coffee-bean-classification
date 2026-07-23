# Controlled Swin-HSSAM Reproduction for SNI Coffee Defects

## Scope

This protocol adapts Jiao et al. (2025), *Swin-HSSAM: A green coffee bean
grading method by Swin transformer*, to the repository's 21-class SNI
instance-crop classification dataset.

It is a **controlled method reproduction**, not a numerical reproduction of
the paper. The paper used a proprietary green-Arabica dataset and a different
label space. All model selection here remains validation-only; the test split
is locked.

## Paper components implemented

1. ImageNet-pretrained Swin-T backbone at 224 px.
2. Three-stage feature extraction from timm indices 1, 2, and 3
   (channels 192, 384, and 768; spatial scales 28, 14, and 7).
3. HS-FPN:
   - max/average channel screening;
   - 1x1 lateral projection to 256 channels;
   - top-down nearest-neighbor fusion;
   - 3x3 SFF refinement.
4. SAM:
   - controlled depthwise-separable convolution (CDSC);
   - three fully connected transformations;
   - channel expansion;
   - element-wise enhancement of the fused feature map before GAP.
5. Fusion Loss:
   - multiclass CE plus focal loss;
   - alpha 0.25 and gamma 2;
   - CE weight 0.7 and focal weight 0.3, which was the paper's best reported
     combination.

The optimizer, data split, augmentation, epoch budget, and evaluation metrics
remain the established SNI repository protocol. This prevents a training
recipe change from being mistaken for an architectural gain.

## Corrections to the released source

The authors' GitHub source is valuable but internally inconsistent:

- its training entrypoint instantiates `SwinTransformer`, not the provided
  `EnhancedSwinTransformer`;
- the enhanced class contains HS-FPN but no SAM;
- the ordinary Swin class applies SAM after global pooling;
- the three-input HS-FPN path replaces an intermediate fusion and can discard
  its middle-stage contribution;
- released SAM code returns a vector and omits the element-wise multiplication
  shown in Fig. 8.

This repository follows the paper's figures and prose where those conflict
with the released entrypoint:

- all selected stages remain in a conventional top-down path;
- SAM operates on a feature map and performs the Fig. 8 multiplication;
- spatial pooling is inserted before SAM's FC stack so it is well-defined for
  any feature-map size;
- focal loss is computed directly from per-sample multiclass CE rather than
  through a redundant one-hot broadcast.

These are disclosed reconstruction choices, not attributed verbatim to the
authors' implementation.

## Factorial

| Code | HS-FPN | SAM | Fusion Loss |
|---|---:|---:|---:|
| SJ0 | no | no | no |
| SJH | yes | no | no |
| SJS | no | yes | no |
| SJL | no | no | yes |
| SJHL | yes | no | yes |
| SJSL | no | yes | yes |
| SJHS | yes | yes | no |
| SJFULL | yes | yes | yes |

The full factorial mirrors the three factors in Table 7 of the paper.

## Fail-fast sequence

First run only:

```bash
python -u -m bilinear_lmmd.experiments.run_jiao_swin_hssam_screening \
  --data-root /content/sni-instance-crops \
  --output-root /content/drive/MyDrive/sni-jiao-hssam-v1 \
  --seeds 42 \
  --stage screen \
  --evaluation-split val
```

This trains `SJ0` and `SJFULL`. The full model passes only when:

- mean validation Macro-F1 rises;
- mean validation hard-group F1 rises;
- mean validation Worst-F1 falls by no more than one percentage point.

If it fails, stop. Do not train the other six variants and do not open test.

If it passes, reproduce the remaining factorial on the same validation seed:

```bash
python -u -m bilinear_lmmd.experiments.run_jiao_swin_hssam_screening \
  --data-root /content/sni-instance-crops \
  --output-root /content/drive/MyDrive/sni-jiao-hssam-v1 \
  --seeds 42 \
  --stage ablation \
  --evaluation-split val
```

Only a candidate that survives this step should be confirmed on seeds 123 and
2026. The test split must remain closed until the architecture and all
hyperparameters are frozen.

## Interpretation boundary

A positive result supports transfer of the Swin-HSSAM mechanism to SNI
instance-crop classification. It does not validate the paper's proprietary
dataset result, and it does not establish that Swin-HSSAM is universally
superior to EfficientNetV2 or HBP.

References:

- Jiao et al., 2025, DOI: 10.1371/journal.pone.0322198.
- Official paper-linked source:
  <https://github.com/tonyalice77/Papers-related-to-swin-transformer-improvements>
