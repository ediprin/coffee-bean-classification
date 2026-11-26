# Cross-branch experiment anti-loop audit — 2026-11-26

Status: NO_NEW_TRAINING / RESEARCH DIRECTION HOLD. User requested review of prior classification work before proposing another method.

## Scope and provenance
Read active handoff, AGENTS, ledger and master results; enumerate Git branch refs, inspect active and historical Git tree inventories and compare branch-specific paths. Directly read selected configurations, protocols and archived results, especially the old master log on agent/sni-instance-crops:
- Active snapshot: 8bb6e4f5835ff5acd91efa7c2128cb3e8dfef6c2
- Historical snapshot: 8570468435926eb66ec7a20e482905fb046a7658
- Historical master log: https://github.com/ediprin/coffee-bean-classification/blob/8570468435926eb66ec7a20e482905fb046a7658/docs/results/EXPERIMENT_MASTER_LOG.md
The earlier ledger covers mainly the later MobileNetV3 preprocessing chain and omits major historical families. Historical master log section 13 explicitly identifies many metrics as transcriptions of earlier outputs, not all backed by committed raw predictions. Treat numerical summaries as archived reported results, not independently recomputed metrics. No claim that every historical commit, notebook cell, external checkpoint or raw artifact has been audited. Never infer untested status from default-branch search or absence of result files on active branch.

## Major recovered results (percentage points unless otherwise specified)
| Family | Recorded evidence | Status |
|---|---|---|
| Backbone screen, seed123 val | EfficientNetV2-B0 HBP/GAP Macro 88.65/88.24; SHViT-S1 GAP 83.69; MobileNetV4-Conv-Medium GAP/HBP 83.28/83.28; PVTv2-B0 GAP/HBP 83.10/77.13; ConvNeXtV2-Atto GAP/HBP 82.46/71.01; SHViT-HBP 81.50 | HISTORICAL_SCREEN; do not repeat as new |
| HBP vs GAP, old clean fold1 test, 3 seeds | Macro 86.78 vs 85.46 (+1.32 ±4.51), Hard +2.48, Worst +7.64; high variability | HISTORICAL_POSITIVE, not same protocol as 965-image OOF |
| Granularity fine17 vs coarse9 | Fine test HBP gain +3.03, CI [+0.57,+5.69]; coarse +0.42, granularity difference CI [-0.88,+6.25] | HISTORICAL_POSITIVE fine gain; interaction inconclusive |
| ArcFace A2 GAP / A3 HBP | A2 Macro +0.10, Hard -0.41, Worst -9.79; A3 -4.43/-5.81/-11.86 | FAIL_STOP |
| Resolution 320 HBP | -3.49/-4.40/-6.06 vs 224 | FAIL_STOP; F2/F3 320 ArcFace planned not continued |
| M1s spatial HBP | +0.57/+1.21/-6.67 | FAIL_STOP; historical summary recovered |
| E1 global-local MoE confirmation | -1.01/-0.49/-4.16 | FAIL_STOP; historical summary recovered |
| O1 object crop | -6.96/-11.78/-15.87 | FAIL_STOP; historical summary recovered |
| EMA M1e / capacity residual C1 | Macro -0.43 / -0.30 | FAIL_STOP |
| H0/H0f hierarchy | Macro +1.22/+1.30, Hard -0.34/-0.16 | FAIL_STOP |
| Compact MPN-COV | Macro +1.08, Hard -0.09, Worst -13.54 | FAIL_STOP |
| CST classical descriptors / disagreement gate | CST Macro 71.74; gate 84.30 vs HBP 84.41 on old 979 protocol | FAIL_STOP gate |
| OMSL taxonomy-compatible contrastive | Dataset-Macro +0.08; Coffee17/CBD heterogeneous labels | ARCHIVED_STOP |
| EfficientNetV2 progressive E2/E3 | E2 three-seed val Macro 89.44 vs GAP 88.77 vs HBP 89.93; E3 consistency failed | FAIL_STOP |
| Vanilla SupCon CP1, confusion-aware SupCon CP2 | All six comparisons FAIL; archived JSON metrics_available=false | FAIL_STOP, exact deltas missing |
| Hong DSConv/SPPF | HCD1 three-seed Macro 88.28 vs GAP 88.77/HBP 89.93; combination failed | FAIL_STOP |
| Chang-Liu MDE | MDE1 Macro +0.65 vs GAP but Hard -0.66; vs HBP Macro -1.43 | FAIL_STOP |
| EfficientNetV2 multistage MSF0/MSF1/MSFC | MSFC Macro 92.37; adaptive MSF1 92.38, Hard -1.12 vs capacity-matched MSFC | FAIL_STOP adaptive selection; control only screening |
| EfficientNetV2 DCL0/1/2 | DCL0 confirmation vs GAP Macro +1.17, Hard -0.11, Worst -6.46; DCL1 not confirmed, DCL2 identical to DCL1 screening | FAIL_STOP; distinct earlier DCL from later HBP DCL |
| SNI instance-crop dataset | Two Roboflow sources (Adrian/Faruq), 31,074 crops, 21 nominal then 15 visual classes; v1 multiresolution screen +1.73 Macro, ontology fails; v2 multiresolution -0.68 Macro, Worst -21.89; selective HBP fails; v3 crop gate fails; v3.1 group-primary Jiao protocol not run | ARCHIVED/BLOCKED; does not satisfy public-with-paper requirement without new evidence |

## Anti-loop decisions
Do not propose backbone sweep, ArcFace, vanilla SupCon, confusion-aware contrastive, progressive multigranularity, local max/MoE, spatial HBP, object crop, hierarchy supervision, covariance pooling, adaptive multistage recalibration, DCL, MDE, DSConv/SPPF or SNI crop reconstruction as fresh ideas. ConvNeXt V2 Atto was actually screened; Nano/Tiny are distinct variants but no reason to claim a new method or launch a sweep. Distinguish exact tested configuration from broader family; this is a decision boundary, not proof all variants fail. Active branch's 17-arm master, focused FReCSA, recovery, OOF and frozen prototype results remain valid within their respective scopes. C2 final NO_INTERNAL_SUPPORT overrides earlier KEEP (development only). HBP older positive results must not disappear; no claim of stable superiority across protocols.

## Required workflow
Before new training proposal: search across active and archived branches, read old master log, verify config AND result/decision, inspect nearby methods, compare against Kaewkamol 2025 and other coffee papers, state material mechanism difference, data/protocol and whether any independently held data is available. If results absent, BLOCKED_RESULT_MISSING; if protocol explicitly says not run, REGISTERED_NOT_RUN. Do not recover weights by repeating completed runs without user authorization. No new experiment authorized by this audit. All future Coffee17 tests are retrospective internal given previous exposure. Actual thesis objective remains improved performance with secondary public dataset plus associated paper.
