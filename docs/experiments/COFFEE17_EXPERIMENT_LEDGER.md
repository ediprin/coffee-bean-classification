# Coffee17 Experiment Ledger — Check Before Proposing Anything New

Last updated: 2026-09-29

This file is the **first stop before proposing, implementing, or training any
new Coffee17 method**.

The purpose is to prevent repeated experiments and repeated hypotheses under
new names.

## Mandatory workflow

Before proposing a new experiment:

1. Check this ledger.
2. Search the repository for the mechanism and close synonyms.
3. If an implementation/config already exists but its outcome is missing from
   the repository, mark it `BLOCKED_RESULT_MISSING`; do **not** casually rerun
   or propose a near-duplicate.
4. If a family is `FAIL_STOP` or `CLOSED`, do not reopen it without a new,
   explicitly stated hypothesis that is materially different from the failed
   mechanism.
5. After every completed run, commit a result record under `docs/results/`
   and update both this ledger and
   `docs/experiments/COFFEE17_EXPERIMENT_MASTER_RESULTS.md` **before** proposing
   the next method.

Status meanings:

- `KEEP`: retained baseline / useful evidence.
- `PASS_EXPLORATORY`: passed its development gate but not a universal claim.
- `FAIL_STOP`: tested and failed its frozen gate; do not tune/repeat on reused folds.
- `CLOSED`: the tested family has enough negative evidence to stop same-family variants.
- `BLOCKED_RESULT_MISSING`: implementation/config is present, but the result artifact is
  not currently recorded in the repo. Treat as already occupied territory until
  the old result is recovered.
- `ANALYSIS_ONLY`: diagnostic evidence, not a candidate model.

## Canonical ledger

| Family / experiment | Mechanism actually tested | Status | Canonical evidence |
|---|---|---|---|
| M0 GAP | MobileNetV3-Large + GAP + CE | KEEP | `docs/results/FINAL_HBP_RESULTS.md` |
| M1 HBP | MobileNetV3-Large + hierarchical bilinear pooling + CE | KEEP | `docs/results/FINAL_HBP_RESULTS.md` |
| S1 SPPF-Attention-HBP | SPPF + channel/spatial attention before HBP | FAIL_STOP | `docs/results/FINAL_HBP_RESULTS.md` |
| E1 HBP local MoE | HBP global expert + intermediate 14x14 local expert using 1x1 projection + global max pooling + learned gate | BLOCKED_RESULT_MISSING | config: `configs/coffee17/E1_mobilenetv3_hbp_local_moe_source.yaml`; protocol: `docs/protocols/HBP_MOE_PROTOCOL.md`; implementation commit `f10a09f9fc37467528d4e4a62e2a7266452c9d9e` |
| M1s spatial HBP | spatially preserved HBP instead of immediate spatial averaging | BLOCKED_RESULT_MISSING | config: `configs/coffee17/M1s_mobilenetv3_sp_hbp_source.yaml`; implementation commit `110f4afa710cc9b4442ffba714bd5384a8a8a149` |
| O1 HBP object crop | object-centric crop + HBP | BLOCKED_RESULT_MISSING | config: `configs/coffee17/O1_mobilenetv3_hbp_object_crop_source.yaml`; implementation commit `e0c18e67c087371e14de37525c1fe06476f2740b` |
| Primary preprocessing | R0 / C0 / F0 / W0 standalone preprocessing study | ANALYSIS_ONLY | `docs/experiments/COFFEE17_EXPERIMENT_MASTER_RESULTS.md` |
| Shared multi-view / preprocessing transfer | MVCE, SBN, AT-SBN, MVFD, W0-RAS, W0-AT, ML-MVFD, HBP+MVFD | CLOSED | master results, sections 8-15 |
| W0-HBP matched | direct W0 input into HBP | FAIL_STOP | `docs/results/W0_HBP_MATCHED_RESULTS.md` |
| WR-HBP V1 | tiny L1 luminance wavelet residual before raw-RGB HBP | PASS_EXPLORATORY | `docs/results/WAVELET_RESIDUAL_HBP_RESULTS.md` |
| WR + CLAHE | CLAHE extension of WR-HBP | FAIL_STOP | `docs/results/WAVELET_CLAHE_RESIDUAL_HBP_RESULTS.md` |
| WR-PDR-HBP | physical-descriptor logit residual on top of WR-HBP | FAIL_STOP | `docs/results/WR_PDR_HBP_V1_RESULTS.md` |
| WR-HBP GCE | top-2 generalized cross entropy / label smoothing variant | FAIL_STOP | `docs/results/WR_HBP_GCE_V1_RESULTS.md` |
| WR-HBP SAR V1 | self-assessment / reassessment branch | FAIL_STOP | `docs/results/WR_HBP_SELF_ASSESSMENT_V1_RESULTS.md` |
| Frozen representation audit | frozen ImageNet MobileNetV3 vs frozen DINOv2-S/14, kNN + linear probe | FAIL_STOP | `docs/results/COFFEE17_REPRESENTATION_GEOMETRY_V1_RESULTS.md` |
| DCL local learning V1 | local region destruction/shuffle + original/shuffled auxiliary task + location reconstruction | FAIL_STOP | `docs/results/COFFEE17_DCL_LOCAL_LEARNING_V1_RESULTS.md` |

## Explicit near-duplicate blocks

The following proposals are **not new** enough to justify another run without
recovering/using the prior evidence first:

- "use global max/local strongest feature" -> overlaps E1 local MoE;
- "preserve spatial HBP map" -> overlaps M1s spatial HBP;
- "crop tightly around the bean/object" -> overlaps O1 object crop;
- "add SPPF + attention before HBP" -> S1 already failed;
- "replace global representation with DINOv2 and distill it" -> frozen
  representation gate failed; distillation is not justified without new
  teacher evidence;
- "shuffle/mask/destruct patches to force local learning" -> DCL V1 failed
  strongly; destructive-local variants are blocked by default;
- "change WR-HBP loss to GCE" -> failed;
- "add physical descriptor residual to WR-HBP logits" -> failed;
- "another preprocessing-to-raw feature transfer" -> transfer family closed.

## Known repository gap

Three older local-evidence experiments have implementation/config evidence but
their numerical outcome is not currently stored in the canonical result
documents:

- E1 HBP local MoE;
- M1s spatial HBP;
- O1 HBP object crop.

Because of that gap, **do not infer that they were never tested**. Recover their
old outputs/results first if a future decision depends on them. Until then they
remain `BLOCKED_RESULT_MISSING`, not `NOT_RUN`.

This rule exists specifically to stop the project from cycling through old
ideas under slightly different names.
