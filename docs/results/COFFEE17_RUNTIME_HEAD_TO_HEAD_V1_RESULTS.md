# Coffee17 Runtime Head-to-Head V1 — Results

Date: 2026-11-26
Status: COMPLETE / C2 RETAINED AS PRIMARY ARCHITECTURE FOR CONFIRMATION

## Provenance and checks

- Registered protocol: [COFFEE17_RUNTIME_HEAD_TO_HEAD_V1_PROTOCOL.md](../experiments/COFFEE17_RUNTIME_HEAD_TO_HEAD_V1_PROTOCOL.md).
- Scientific commit: `b659a22a72c97bfe75576424a38cbb10076ab0be`.
- User-provided artifact: `coffee17-runtime-head-to-head-v1.json`; SHA256 `75deecaf770b97db4a38aed0e29833199ea29bc4b5bb4183f0bfee977ca1016d`. Raw file is retained by user; this document records derived results, not a committed copy of the raw file.
- Config file SHA256: `73ffd6235240928a2c132f478567cfe1ec86ac8228326d51b4d58c31d06f6a54`.
- NVIDIA Tesla T4, cuda:0 (one GPU used), PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, Python 3.13.15, CUDA runtime 12.8, cuDNN version code 91900, driver 580.178.04.
- FP32 eager forward, resident synthetic 1x3x224x224 input, seed-42 random weights, TF32 off, cuDNN benchmark off. 30 initial warmups, 10 block warmups, 10 balanced rounds of 50 observations per arm. Separate throughput blocks.
- Verified from raw file: expected scientific commit, five parameter counts, 500 latency observations and 10 throughput blocks per arm; each arm occupies each ordinal position twice. Recomputed pooled medians, nearest-rank p95 and throughput from raw values, matching reported values. All three flags training_performed, dataset_accessed and outer_test_accessed are false. These are artifact checks, not independent verification of the physical execution environment.

## Main results

Development metrics are historical values from the master and focused C1/C2 screens; no accuracy measured in this profiling run. Macro/Hard/Worst are mean development-fold percentages.

| Arm | Macro-F1 % | Hard-F1 % | Worst-F1 % | Parameters | Conv/Linear MACs | State tensor MiB | Median ms | p95 ms | Sustained batch-1 img/s | Isolated peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 89.74 | 85.20 | 66.15 | 2988289 | 214097280 | 11.493 | 5.788896 | 7.762895 | 171.0459 | 19.7227 |
| W3 | 90.47 | 85.59 | 68.00 | 2992169 | 214158752 | 11.508 | 6.797204 | 7.656916 | 146.5574 | 19.7402 |
| C2 | 91.36 | 87.38 | 70.72 | 2993089 | 214097280 | 11.526 | 6.027858 | 6.675844 | 161.0209 | 19.7593 |
| B1 | 91.13 | 85.63 | 60.31 | 3562305 | 287966144 | 13.694 | 6.694491 | 7.877106 | 149.5978 | 27.1479 |
| W4 | 91.52 | 85.45 | 61.44 | 3566185 | 288027616 | 13.709 | 7.657686 | 8.632399 | 129.3927 | 27.2202 |

Peak is model+input+forward allocated memory after subtracting the common initial allocation (9,568,256 bytes), not reserved GPU/process memory. Forward increment is 7,946,240 bytes for B0/W3/C2, 13,422,592 for B1, 13,479,936 for W4. Memory is an isolated snapshot measurement; do not compare directly with earlier preflight definitions. State tensor bytes are not a serialized checkpoint file size.

## C2 relative changes

Percent changes use the comparator as denominator. Negative latency means faster; positive throughput means more images/s.

| Comparator | Delta Macro pp | Params % | Median latency % | Throughput % | Peak memory % | Rounds C2 faster /10 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | +1.62 | +0.161 | +4.13 | -5.86 | +0.186 | 1 |
| W3 | +0.893 | +0.031 | -11.32 | +9.87 | +0.096 | 10 |
| B1 | +0.235 | -15.98 | -9.96 | +7.64 | -27.22 | 10 |
| W4 | -0.151 | -16.07 | -21.28 | +24.44 | -27.41 | 10 |

C2 overhead over B0 is 0.2389625 ms and 4,800 parameters. Versus W4, C2 gives up only 0.151 pp observed development Macro-F1 while gaining 1.934 pp Hard and 9.281 pp Worst. C2 dominates W3 and B1 for the specific two objectives mean development Macro-F1 (maximize) and median latency (minimize). The two-objective Pareto frontier among these five arms is B0, C2, W4. Do not claim strict multi-objective dominance over W3 if including parameter count or peak memory, since W3 is slightly smaller.

## Timing variation and caveats

| Arm | SD of 10 round medians (ms) | Min–max round median (ms) |
|---|---:|---:|
| B0 | 0.1963 | 5.6014–6.3325 |
| W3 | 0.1235 | 6.6342–7.0000 |
| C2 | 0.0807 | 5.9439–6.1760 |
| B1 | 0.1912 | 6.5687–7.2376 |
| W4 | 0.1927 | 7.4950–8.1590 |

C2 is faster than W3, B1, W4 in every round, with round-median relative change ranges respectively -13.88 to -7.29%, -16.82 to -7.22%, -24.39 to -19.02%. C2 is slower than B0 in 9/10 rounds; median paired-round overhead +3.69% (range -2.58 to +6.94%). No outlier removal. B0 round 4 p95 10.56 ms and round 5 p95 8.65 ms suggest transient timing noise. Lower C2 pooled p95 and round dispersion are descriptive for this single session; do not claim universal tail-latency superiority or independent-run statistical significance. C2 throughput round SD 7.90 img/s vs B0 2.13, so consistency is metric-specific. Same-session order balance mitigates but does not eliminate thermal/host/runtime effects.

FDA incurs extra runtime despite very small counted Conv/Linear MAC overhead: W3 vs B0 ~+17.42% median latency and W4 vs B1 ~+14.39%. Conv/Linear MAC counts omit custom transforms, normalization, pooling and elementwise work; they do not prove identical total FLOPs. Random-weight model forward excludes transfer, normalization, softmax, preprocessing and I/O; these are not trained-checkpoint or edge/CPU/end-to-end deployment measurements. No accuracy confirmation here.

## Decision

Retain and freeze **C2 MobileNetV3-Large + late full FReCSA + GAP + Linear(17)** as the primary architecture for confirmation. Keep B0 matched baseline and W4 highest-development-Macro comparator; W3 and B1 remain documented comparators. No new module stacking or training authorized by this result. Evidence supports the accuracy–latency tradeoff under this protocol, not a new attention invention or statistical superiority on independent test data.

Next: audit actual original checkpoint availability and historical image exposure across opened OOF/development evaluations; register C2-specific confirmation protocol before any test access. If the original checkpoints are unavailable, recovery training requires explicit user authorization. If no genuinely independent test population remains, label evaluation accordingly and seek independent data for generalization claims. Update thesis novelty argument separately.
