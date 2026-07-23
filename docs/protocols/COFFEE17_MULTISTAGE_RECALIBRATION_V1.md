# Coffee17 multistage recalibration protocol v1

## Status

**Architecture freeze only. No training result is claimed in this document.**

This protocol defines a minimal test of local--global multistage representation
learning on Coffee17. It does not reproduce MFSwin, SNet, or RSCD-Net, and it
does not claim defect localization.

## Motivation grounded in the source papers

Three source papers motivate separate parts of the research question:

1. Bi et al. (2022), *Development of Deep Learning Methodology for Maize Seed
   Variety Recognition Based on Improved Swin Transformer*, reports an
   ablation from 92.91% average accuracy for Swin-T, to 94.77% with multiscale
   features, and to 96.47% with multiscale features plus feature attention.
   The principle retained here is complementary information across stages.
2. Huang et al. (2022), *Deep learning based soybean seed classification*,
   reports 94.0% mean accuracy for SNet without MFR and 96.2% with MFR. The
   principle retained here is recalibration of features associated with subtle
   damage. MFR itself is not copied.
3. Zhang et al. (2025), *Multi-class rice seed recognition based on deep space
   and channel residual network combined with double attention mechanism*,
   reports 77.78% for ResNet50, 80.55% for ResNet50-SC, 79.11% for
   ResNet50-A2, and 81.94% for RSCD-Net. This supports treating spatial and
   channel redundancy as a design concern. SCR/A2 are not inserted into v1.

These results are not numerically comparable with Coffee17 because the crops,
labels, acquisition protocols, and splits differ.

## Research question

Given three EfficientNetV2-B0 endpoint maps,

\[
F_s=f_s(x),\qquad F_m=f_m(x),\qquad F_d=f_d(x),
\]

does adaptive, per-image stage/channel recalibration improve on a fixed
multistage fusion while preserving hard and lower-tail class performance?

The tested operation is:

\[
\widetilde F_i=A_i(F_i),
\]

where \(A_i\) projects every endpoint to a shared channel dimension and aligns
its spatial resolution to the middle endpoint. Larger maps are reduced with
adaptive average pooling; smaller maps are enlarged with bilinear
interpolation.

MSF0 uses fixed weights:

\[
F_{\mathrm{MSF0}}=\frac{1}{3}
\left(\widetilde F_s+\widetilde F_m+\widetilde F_d\right).
\]

MSF1 obtains descriptors without first discarding the feature maps:

\[
d_i=\operatorname{GAP}(\widetilde F_i),
\]

then predicts channel-wise weights:

\[
(g_s,g_m,g_d)=\operatorname{softmax}_{stage}
\left(h([d_s,d_m,d_d])\right),
\]

and fuses:

\[
F_{\mathrm{MSF1}}=
g_s\odot\widetilde F_s+
g_m\odot\widetilde F_m+
g_d\odot\widetilde F_d.
\]

Only after spatial fusion is the final embedding computed with GAP.

## Frozen models

| Code | Model | Purpose |
|---|---|---|
| BE2G | EfficientNetV2-B0 + final-stage GAP | Existing first-order baseline; do not retrain when the protocol/checkpoint matches |
| BE2H | EfficientNetV2-B0 + HBP | Existing second-order multistage comparator |
| MSF0 | Three aligned stages + uniform fusion | Isolate the value of direct spatial multistage fusion |
| MSF1 | MSF0 + adaptive stage/channel gate | Isolate adaptive recalibration |

MSF0 and MSF1 have identical encoder, endpoint projections, fused embedding,
dropout, and classifier. The final gate layer in MSF1 is initialized to zero,
so MSF1 starts as exact uniform fusion rather than with an arbitrary stage
preference.

## Explicit distinction from E2/PMG

MSF0/MSF1 are not the stopped E2 progressive multi-granularity experiment:

| Property | E2 | MSF0/MSF1 |
|---|---|---|
| Input manipulation | 8x8, 4x4, 2x2 jigsaw plus full image | One intact image |
| Optimization | Four optimizer updates per batch | One update per batch |
| Classifiers | Three branch classifiers plus concat classifier | One classifier |
| Inference | Sum of four logits | Classification of one spatially fused map |
| Stage selection | Separate branch objectives | Joint feature-level weighting |

The prior E2 result remains negative evidence: generic multistage machinery
was not stable enough to replace HBP. MSF1 must therefore demonstrate an
effect beyond merely using several endpoints.

## Claim boundary

With image-level labels only, the permitted description is:

> class-supervised local--global multistage recalibration for fine-grained
> coffee-defect classification.

Do not claim that MSF1 localizes a defect. A learned stage/channel weight is
not a pixel-level defect annotation. CAM visualizations may be reported as
qualitative diagnostics only unless expert location annotations are added.

## Static acceptance before training

The implementation must satisfy:

1. exactly three endpoint maps;
2. aligned maps retain two spatial dimensions;
3. MSF0 weights equal \(1/3\);
4. MSF1 weights are finite, non-negative, and sum to one over stages for every
   sample and channel;
5. same-seed MSF0 and newly initialized MSF1 produce identical common
   parameters and identical logits before the gate learns;
6. gradients reach every endpoint projection and the adaptive gate;
7. source-only training, one classifier, and one standard CE objective;
8. parameter and latency overhead are reported before GPU training.

## Future evaluation sequence

Training is not authorized by this architecture-freeze document. When a runner
is added, the minimum sequence is:

1. one-seed validation screening of MSF0 and MSF1 while reusing BE2G/BE2H;
2. MSF1 versus MSF0 is the primary causal comparison;
3. only if MSF1 improves Macro-F1 and Hard-F1 without a material lower-tail
   collapse, add a capacity-matched control;
4. only after that control passes, confirm unchanged MSF1 on seeds 42, 123,
   and 2026;
5. keep test closed until the validation protocol is frozen.

Macro-F1 is primary. Hard-group F1 and the mean of the bottom three class F1
scores are targeted safety metrics. Worst-class F1 is reported, but it must
not be interpreted without its small class support and seed/bootstrap
uncertainty.

