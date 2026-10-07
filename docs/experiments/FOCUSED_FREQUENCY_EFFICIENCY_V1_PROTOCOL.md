# Coffee17 Focused Frequency-Efficiency Screening V1

## Purpose

This follow-up is deliberately smaller than the 17-arm master screening. It tests
only hypotheses that remain justified after the master results:

- W3 FDA-GAP is the strongest lightweight frequency arm.
- W4 FDA-HBP has the highest mean Macro-F1 and the most consistent positive
  fold count among the frequency arms.
- FReCSA (Zhuang et al., ESWA 2025) is explicitly designed for small/moderate
  datasets and adds a predefined high-pass spatial recalibration with very low
  parameter/computation overhead.
- Efficiency must be measured by real latency/throughput in addition to
  parameter count and operation counts.

The outer test set remains locked. This is development-fold screening only.

## Frozen controls and candidates

| ID | Architecture | Matched anchor | Purpose |
|---|---|---|---|
| B0 | MobileNetV3-Large + GAP | B0 | first-order lightweight control |
| H384 | MobileNetV3-Large + HBP(p=384) | H384 | compact-HBP control |
| C1 | MobileNetV3-Large + late FRSA + GAP | B0 | isolate FReCSA spatial high-pass branch |
| C2 | MobileNetV3-Large + late full FReCSA + GAP | B0 | test channel+spatial complementarity |
| C3 | MobileNetV3-Large + late FDA + HBP(p=384) | H384 | compact the successful W4 mechanism |
| C4 | MobileNetV3-Large + late full FReCSA + HBP(p=384) | H384 | frequency-spatial attention with compact HBP |

C1/C2 are late-feature MobileNetV3 adaptations, not literal reproductions of
the paper's block-wise ResNet placement. C3/C4 use p=384 by preregistration;
they must be compared to H384, not to B1 as a one-factor ablation.

## Frozen FReCSA settings

From the source-paper ablations:

- predefined high-pass filter
- low-pass implementation: AvgPool
- kernel: 7x7
- local-local interaction
- fixed bias: 0.5
- channel attention: GAP -> BN -> independent channel scaling -> sigmoid
- spatial attention: high-pass -> local-local product -> BN -> ReLU(+0.5) -> sigmoid
- placement in this adaptation: deepest selected MobileNetV3 feature only

Important source discrepancy: the published paper explicitly includes a fixed
bias term and reports b=0.5 as the selected default, while the authors'
released `models_lpf/resnet_csha.py` implementation omits that additive bias
in the forward path. V1 follows the published equations/ablation for b=0.5,
but matches the released source for `AvgPool2d(kernel=7, stride=1, padding=3)`
semantics (including PyTorch's default `count_include_pad=True`). This
difference must be reported rather than silently treated as an exact
reproduction.

No Coffee17-specific search over kernel, bias, interaction, or filter type is
allowed in this screening.

## Training contract

Same frozen Coffee17 development protocol as the master screen:

- seed 42
- five grouped folds
- image size 224
- batch size 32
- fixed 0/45/90/135/180/225/270 rotation schedule
- MobileNetV3-Large ImageNet initialization
- AdamW, lr 3e-4, weight decay 1e-4
- cosine scheduler
- 50 epochs
- cross entropy + label smoothing 0.1
- no EMA
- no object crop
- no outer-test access

Do not enable strict deterministic algorithms on CUDA because the established
HBP path uses adaptive pooling backward that is unsupported under strict mode
on the target runtime. Seeded matched initialization remains required.

## Efficiency preflight before training

Run the profiler first for B0, H384, C1-C4. Record:

- trainable/total parameters
- serialized state size
- Conv2d+Linear MAC estimate
- batch-1 median latency
- batch-1 p95 latency
- batch-1 throughput
- peak CUDA memory
- successful real forward/backward pass

The MAC estimate intentionally does not pretend to be a complete FLOP count for
custom wavelet/elementwise operators. Runtime latency/throughput is the primary
deployment evidence.

## Decision logic

Primary metric: mean Macro-F1 across five development folds.

Secondary metrics: Hard-F1, Worst-F1, positive-fold count, paired rescue/damage,
parameter count, latency, throughput, and peak memory.

Interpretation:

1. C1 vs B0 isolates the FReCSA frequency-regulated spatial branch.
2. C2 vs B0 determines whether the paper's simplified channel branch adds value
   beyond MobileNetV3's existing internal SE mechanisms.
3. C3 vs H384 determines whether FDA remains beneficial after HBP compression.
4. C4 vs H384 determines whether FReCSA remains beneficial with compact HBP.
5. FDA + FReCSA stacking is forbidden in V1. It may be tested only if individual
   components first show reproducible positive evidence.

A candidate is not promoted merely for a tiny Macro-F1 gain if it worsens
runtime efficiency materially. The final choice is made on the observed
accuracy-efficiency Pareto frontier, not parameter count alone.
