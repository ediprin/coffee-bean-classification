# Coffee17 Representation Geometry Audit V1 — Results

Date recorded: 2026-09-29

Source package SHA-256:
`4a7de64d46f3af5941d569a9a1931fde585aa468719b1b647858d6bf089120c4`

Source summary SHA-256:
`d2ac61221fea304ccdadfde0a6138394bc01e9debb1f258814f00b8d2ed8c0f0`

Protocol:
- five locked Coffee17 development folds;
- raw RGB;
- frozen ImageNet MobileNetV3-Large vs frozen DINOv2-S/14;
- 5-NN and fixed-C linear probe;
- no encoder fine-tuning;
- outer test untouched.

## Linear-probe result

| Metric | MobileNetV3 | DINOv2-S/14 | Delta DINO-MobileNet |
|---|---:|---:|---:|
| Macro-F1 | 63.86% | 60.52% | -3.34 pp |
| Hard-F1 | 57.63% | 54.56% | -3.07 pp |

Macro-F1 delta was positive in only 1/5 folds.
Hard-F1 delta was positive in 0/5 folds.

Audited hard-pair confusion total:
- MobileNetV3 linear probe: 40
- DINOv2 linear probe: 43

5-NN Macro-F1 delta was -5.81 pp and negative in 5/5 folds.

Cross-representation persistence:
- 485 validation observations;
- MobileNet both decoders wrong: 124;
- of those, DINOv2 both decoders also wrong: 91;
- DINOv2 majority-correct rescues among those: 8;
- MobileNet majority-correct while DINOv2 both-wrong: 13.

Frozen decision gate:

`NO_CLEAR_H_R_SUPPORT`

Interpretation:
> Replacing the frozen global representation with DINOv2-S/14 did not improve
> Coffee17 hard-class geometry or simple-probe performance. Do not proceed to
> DINOv2 distillation by default from this result.

This does not prove that all alternative representations will fail. It closes
the specific frozen-MobileNet-vs-DINOv2 representation hypothesis tested here.
