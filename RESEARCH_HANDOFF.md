# Coffee17 Research Handoff

Updated: 2026-11-26. Last recorded completed experiment: runtime head-to-head (2026-11-26).
Repository: https://github.com/ediprin/coffee-bean-classification
Active research branch: `codex/representation-screening-v1`.

## 0. Start here — current decision

**C2 = MobileNetV3-Large + late full FReCSA + GAP + Linear(17) is the primary efficient architecture candidate.** It is selected from development evidence, not yet confirmed on an independent final test. Do not restart an architecture search merely because a new chat begins.

**Next task:** recovery analysis passed. Run the inference-only internal OOF notebook with saved recovery checkpoints and original dataset; actual weight and source hashes must pass before inference. No training. C2 remains frozen, internal OOF only. Thesis novelty is still open; do not claim a new attention mechanism merely from integrating two published methods.

The canonical evidence is [C1/C2 results](docs/results/COFFEE17_FOCUSED_FRECSA_C1C2_V1_RESULTS.md), [experiment ledger](docs/experiments/COFFEE17_EXPERIMENT_LEDGER.md), and [master results](docs/experiments/COFFEE17_EXPERIMENT_MASTER_RESULTS.md). Read the latest versions, not just this snapshot. If a later experiment exists, use its audited evidence and update this file.

## 1. Research scope and constraints

- Master's thesis in Informatics / machine learning, fine-grained **single-bean image classification**, not object detection, instance segmentation, counting, or SNI batch grading. The earlier detection direction was abandoned for feasibility.
- Objective: achieve a strong accuracy/efficiency balance, particularly on visually similar defect classes, with a compact single-model inference path.
- Dataset: Coffee17, from Arwatchananukul et al. (2024), *Implementing a deep learning model for defect classification in Thai Arabica green coffee beans*, Smart Agricultural Technology, DOI 10.1016/j.atech.2024.100680. Original public dataset: 979 images, 17 defect categories. Kaggle source: https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original . Public dataset with associated paper is a thesis constraint; a Roboflow-only dataset is not acceptable.
- 979 is the original population, not necessarily the cleaned experimental population. The primary preprocessing study reported 965 clean OOF images. Reconstruct/check the provenance and fold manifests; do not silently assume counts or change exclusions. Rotations of an original image must remain within its assigned partition, not be treated as independent observations.
- Do not conflate Coffee17 with USK-Coffee (four classes), Indonesian SNI dataset, or the earlier 21-target detection taxonomy. Paper-reported unseen accuracy 88.63% and CV accuracy 98.78–99.84% are literature figures, not directly comparable to our development Macro-F1.

## 2. Exact retained architecture

C2 inference: raw RGB 224x224 -> ImageNet normalization -> MobileNetV3-Large (`mobilenetv3_large_100`) -> deepest selected feature -> simplified channel attention -> frequency-regulated spatial attention -> GAP -> existing classifier (dropout configuration 0.2, linear 17 logits).

Implementation is in `src/bilinear_lmmd/modeling/frequency_screening.py` (`FREQUENCY_CANDIDATES["C2"]`); inspect the exact pinned source when deriving formulas or drawing the architecture.

For feature tensor X (channel attention first):
- d = GAP(X); A_c = sigmoid(BN(d) * s), where s is independent learnable per-channel scale; Y = X * A_c.
- L = AvgPool7x7(Y), stride 1, padding 3, default count_include_pad=True.
- H = Y - L; S = H * Y; A_s = sigmoid(ReLU(BN(S) + 0.5)); Z = Y * A_s.
- GAP(Z) -> existing linear classifier. No explicit DWT, FFT, HBP, image-space frequency preprocessing, extra inference branch, or ensemble in C2.

This is a late-feature MobileNetV3 adaptation of Zhuang et al., *Frequency Regulated Channel-Spatial Attention module for improved image classification* (Expert Systems with Applications, 2025). Official code: https://github.com/CoVIS-UNT/FReCSA . Paper and released implementation disagree on the fixed bias: paper selects 0.5, released forward omits it; our V1 follows paper bias and released pooling semantics. See [focused efficiency protocol](docs/experiments/FOCUSED_FREQUENCY_EFFICIENCY_V1_PROTOCOL.md). Do not describe as exact paper reproduction or a novel FReCSA invention.

## 3. Current quantitative evidence

Five development folds, seed 42. Percentages below are mean fold metrics, rounded. Do not combine historical protocols as if they were one matched experiment.

| Arm | Macro-F1 | Hard-F1 | Worst-F1 | Parameters |
|---|---:|---:|---:|---:|
| B0 GAP | 89.74 | 85.20 | 66.15 | 2,988,289 |
| C1 FRSA-GAP | 89.68 | 85.52 | 60.91 | 2,990,209 |
| C2 FReCSA-GAP | 91.36 | 87.38 | 70.72 | 2,993,089 |
| W3 FDA-GAP | 90.47 | 85.59 | 68.00 | 2,992,169 |
| B1 HBP | 91.13 | 85.63 | 60.31 | 3,562,305 |
| W2 WCA-HBP | 91.43 | 86.59 | 60.89 | 3,677,505 |
| W4 FDA-HBP | 91.52 | 85.45 | 61.44 | 3,566,185 |

C2: accuracy 91.55%, balanced accuracy 91.49%; versus historical audited B0: +1.62 pp Macro, +2.18 pp Hard, +4.57 pp Worst, +4,800 parameters (+0.161%). Macro and Hard improved in 5/5 folds; Macro deltas +2.348, +0.015, +1.285, +4.269, +0.189 pp. Rescue 13, damage 6, net +7 correct fold observations. C1: -0.07 pp Macro, -5.24 pp Worst; rescue/damage 7/7; FAIL_STOP. C1 vs C2 supports added channel recalibration within this tested sequence, not proof of a causal synergy or isolated channel-only effect (there is no channel-only control in this screen).

C2 vs W3: +0.893 pp Macro, +1.795 Hard, +2.718 Worst, +920 parameters. C2 vs W4: -0.151 pp Macro, +1.934 Hard, +9.281 Worst, -16.07% parameters. W4 remains the highest Macro comparator in the master/focused screening family; not a claim of highest Macro across every historical protocol. C2 is the retained efficient candidate, not necessarily a statistically superior model.

C2 strongest class F1 gains versus B0: Slight Insect Damage +5.38 pp, Dry Cherry +5.00, Fade +4.81, Severe Insect Damage +4.17, Immature +3.27, Withered +2.56. Broken -2.29 pp. Insect hard group +4.77 pp, shape/withered +2.51, sour/black +0.13. These are empirical associations, not proof of particular physical color/texture cues. Selected epochs 23/31/23/31/28; all five epoch-50 scores fell below peak. Use best validation Macro-F1 checkpoint, not last epoch.

Efficiency preflight (CUDA, random-weight architecture profiling; 30 warmups, 100 iterations, batch 1):

| Arm | Median ms | p95 ms | Reciprocal median img/s | State MiB |
|---|---:|---:|---:|---:|
| B0 | 5.590 | 7.770 | 178.88 | 11.493 |
| C1 | 5.613 | 5.939 | 178.16 | 11.508 |
| C2 | 5.700 | 6.114 | 175.44 | 11.526 |
| H384 | 6.331 | 7.451 | 157.94 | 13.128 |
| C3 FDA-HBP384 | 7.243 | 8.260 | 138.06 | 13.143 |
| C4 FReCSA-HBP384 | 6.516 | 7.098 | 153.46 | 13.161 |

C2 median overhead +1.96% vs B0. C3 median overhead +14.4% vs H384 despite counted Conv/Linear MAC overhead +0.023%. C2 counted Conv/Linear MACs 214,097,280, same as B0; this excludes pooling/BN/elementwise/frequency operations. Do not call it full FLOPs or claim memory savings from small allocator differences. Reciprocal median is not independently measured sustained throughput. W3/W4/B1 were not in that profiling run. C3/C4/H384 have preflight evidence only, not classification scores.

## 4. Experimental history — avoid loops

Use the ledger for status and exact definitions. Important earlier stages:
- M0 GAP and M1 HBP retained as controls. SPPF-Attention-HBP failed. Earlier DSConv + GAP failed in a different protocol; do not equate DSConv quantization operator with dynamic convolution.
- Fixed input preprocessing: raw RGB R0, Lab-L CLAHE C0, angular/frequency F0, Haar+VisuShrink W0. Primary OOF Macro 86.79/86.10/86.11/87.14%. None gave convincing standalone gain; complementary errors existed. Uniform late fusion helped but expensive; learned routing unstable with small validation sets.
- KD-ALL4 did not convincingly beat raw baseline. Shared multi-view CE collapsed auxiliary F0 due to inappropriate BN stats; selective BN repaired optimization but not meaningful classification gains. MVCE/SBN/AT-SBN/MVFD, W0-RAS, W0-AT, ML-MVFD, HBP+MVFD transfer family CLOSED. Do not reopen as another near-duplicate preprocessing-transfer mechanism.
- Direct W0-HBP failed matched test. WR-HBP (tiny luminance Haar residual) initially passed exploratory gate 91.02 -> 91.77 Macro, 5/5 positive; later master B2 vs B1 -0.09 pp, 2/5 positive. Do not imply stable universal improvement across protocols. WR+CLAHE, WR-PDR, GCE, SAR failed their gates. SAR had +0.20 pp but only 1/5 positive. An older WR-specific outer protocol remains registered, not C2 authorization.
- Frozen DINOv2 representation vs MobileNetV3 failed (linear probe Macro -3.34 pp). No teacher evidence for default DINO distillation. DCL local destruction failed (91.47 -> 83.31 Macro, Worst 65.64 -> 18.00). HF-deep MRMR20 fusion failed (-0.06 Macro, -0.44 Hard, -2.36 Worst). Do not rescue those families by arbitrary tuning.
- E1 local MoE, M1s spatial HBP, O1 object crop: implementation exists but numerical results missing, status BLOCKED_RESULT_MISSING. Recover old output, not assume never tested.
- Master screen: 17 arms x 5 folds = 85 runs, scientific pin below. B0 GAP, B1 HBP512, B2 WR-HBP; R1/R2 MFR GAP/HBP; R3 LRBP-I adapted; R4 multistage adaptive fusion; R5 HBP CE/focal; W1/W2 WCA GAP/HBP; W3/W4 FDA GAP/HBP; W5/W6 FcaNet LF2 GAP/HBP; W7 dual-domain DWT fusion; F1 late FFC; F2 late GFNet-style. W3 and W4 led frequency screening. W5 marginally higher Macro than W3 but worse Worst and more params; R2 raises Worst at substantial parameter cost; R3/R4 degraded Macro. R3 etc. are adaptations, not exact reproductions. Full 17-arm raw aggregate resides in the master analysis package; do not invent missing rows. W3/W4 originally retained; C2 subsequently supersedes W3 as efficient accuracy candidate.
- Focused FRSA/FReCSA: all six preflight arms pass CUDA forward/backward, C1/C2 alone trained for 10 runs (B0 reused after audit). C1 FAIL_STOP; C2 KEEP. No automatic FDA+FReCSA+HBP stacking, kernel/bias tuning, compact-HBP training, or random new backbone search on repeatedly reused folds.

## 5. Frozen training and evaluation contract

Reference config: `configs/representation_screening/MASTER_SCREENING_V1.yaml` and `configs/frequency_screening/FOCUSED_EFFICIENCY_V1.yaml`. Historical master/focused: ImageNet pretrained MobileNetV3-Large, raw RGB, 224x224, batch 32, seed 42, AdamW 3e-4, weight decay 1e-4, cosine annealing, 50 epochs, CE label smoothing 0.1, dropout 0.2, no EMA, no object crop, rotations 0/45/90/135/180/225/270, end-to-end fine-tuning. HBP uses stages [1,3,4] and p=512; not in C2. Earlier experiments can differ, including strict deterministic mode. Master/focused use seeded matched initialization, not strict CUDA determinism (adaptive pooling backward incompatibility in earlier run). Do not claim bit-identical reproducibility across runtime versions.

Primary metric mean Macro-F1. Secondary Accuracy, balanced accuracy, Hard-F1, Worst-F1 (minimum class F1 per fold, averaged across folds unless explicitly pooled), per-class F1, confusion, paired rescue/damage, positive-fold count, efficiency. Hard set as documented: sour/black = Partial Black, Partial Sour, Full Sour; shape/withered = Withered, Immature, Cut; insect = Slight/Severe Insect Damage. Check metric implementation before aggregating.

Historical B0 reuse requires matching validation identity-label SHA, fold/count/seed, common data/model/training contract and shared-core initialization SHA; abort if mismatch. This does not eliminate runtime-related training variability.

**Critical validity boundary:** 485 validation rows across five development folds represent 394 unique class/filename identities, so the rows and folds cannot be assumed independent. No naive pooled row bootstrap or independence-based p-value as confirmatory evidence. These are repeatedly reused development folds with selection-overfitting risk. The master record explicitly says original preprocessing OOF on 965 images was already opened. The later protocol reports its outer test locked, but its disjoint outer folds cover the clean Coffee17 population. Audit identity exposure across all earlier studies (including opened OOF and cross-fold development) before calling any new C2 evaluation independent/untouched. A locked fold-specific evaluation is not automatically a globally unseen external test. If independence cannot be justified, report it honestly and consider a genuinely independent dataset/collection; do not claim external generalization from the same images.

## 6. Literature map (read primary papers, not just names)

Project Kopi has coffee papers and representation/frequency papers. Verify any new factual claims using primary literature. Key method sources:
- Arwatchananukul et al. 2024 (Coffee17, MobileNetV3; DOI above).
- Yu et al. 2018, *Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition*, ECCV; Kong & Fowlkes 2017, *Low-rank Bilinear Pooling for Fine-Grained Classification*, CVPR (second-order heads).
- Huang et al. 2022, *Deep learning based soybean seed classification* (SNet/MFR, DOI 10.1016/j.compag.2022.107393); Bi et al. 2022, *Development of Deep Learning Methodology for Maize Seed Variety Recognition Based on Improved Swin Transformer* (MFSwin, DOI 10.3390/agronomy12081843).
- Jiao et al. 2025, *Swin-HSSAM: A green coffee bean grading method by Swin transformer*, DOI 10.1371/journal.pone.0322198; Sandhya et al. 2026, *Deep-CofeNet*, DOI 10.1007/s44196-026-01377-x. Different class sets/protocols; do not directly compare headline accuracies.
- Zhang et al. 2024, *Hybrid attention network for citrus disease identification* (FdaNet/HaNet, CEA 220 108907); Zhang et al. 2025, *Fine-grained recognition of citrus varieties via wavelet channel attention network* (WCANet, KBS 311 113128); Qin et al. 2021, FcaNet (ICCV); Xiang et al. 2025, DWTFormer (DOI 10.1186/s13007-025-01349-w); Li et al. 2020, WaveCNet; Chi et al. 2020, FFC; Rao et al. 2021, GFNet. WaveCNet not faithfully implemented in master screen.
- Zhuang et al. 2025, FReCSA (ESWA), primary C2 inspiration; Chen et al. 2023, *Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks*, CVPR (FasterNet/PConv, real latency vs FLOPs). FasterNet is not a trained Coffee17 candidate.
- HaNet-lite, FiGKD distillation, new backbones were suggestions, not completed experiments. Do not present them as tested. Check ledger before revisiting.

## 7. Code / artifact map

| Purpose | Path or identifier |
|---|---|
| Master notebook | `notebooks/Coffee17_Master_Representation_Screening_V1_Kaggle.ipynb` |
| Master analysis-only | `notebooks/Coffee17_Master_Representation_Screening_Analysis_Only_Kaggle.ipynb` |
| Master runner/summary | `src/bilinear_lmmd/experiments/run_master_screening_v1.py`, `run_master_screening_summary_v1.py` in same directory |
| Model builders | `src/bilinear_lmmd/modeling/frequency_screening.py`, `representation_screening.py`, `models.py` |
| Focused preflight | `notebooks/Coffee17_Focused_Efficiency_Preflight_Kaggle.ipynb`; `src/bilinear_lmmd/experiments/profile_frequency_efficiency_v1.py` |
| Focused C1/C2 | `notebooks/Coffee17_Focused_C1C2_V1_Kaggle.ipynb`; `src/bilinear_lmmd/experiments/run_focused_c1c2_v1.py`, `run_focused_c1c2_summary_v1.py` |
| Focused protocols | `docs/experiments/FOCUSED_C1C2_V1_PROTOCOL.md`, `FOCUSED_FREQUENCY_EFFICIENCY_V1_PROTOCOL.md` |
| Canonical results | `docs/results/COFFEE17_FOCUSED_FRECSA_C1C2_V1_RESULTS.md`; master results and ledger linked above |
| Prior WR-only outer protocol | `docs/protocols/WR_HBP_FINAL_OUTER_CONFIRMATION_V1.md` (not authority for C2) |

Scientific pins:
- Master and historical B0: `b1ac6ca548ebb335d69066377d6f1a1fbb56df36`.
- Focused preflight: `2b6d3c7741495ea7287aab99e8c61da7f4050f24`.
- Focused C1/C2: `6653a50e296f7b38591100b395f1a09dbaf68c56`.

Artifact names (user/Kaggle storage; do not assume binary files are committed):
- `coffee17-master-screening-analysis-package.zip`
- `focused_efficiency_preflight.json` (SHA256 `8c56b659455fb9a5f65aac550b2d90ec7e3a26c412627728456a5c55eccbe621`)
- `coffee17-focused-c1c2-analysis-package.zip` (SHA256 `70365b4661d0a9b8509d68cd3ccfe58ac524dfbf54ed93f20512a44feca72a32`)

Compact packages include metrics, predictions/probabilities, config, contracts, histories and summaries, but no checkpoints. **Known operational flaw:** existing focused training notebook deletes project .pt files after compact ZIP creation. Earlier master packaging also deleted checkpoints and led to accidental retraining. Exact original best checkpoints may be missing; actual availability has not been verified. Do not run those cleanup cells again. Do not invent checkpoint availability, silently rebuild, or use random weights for accuracy evaluation. Recover original best.pt with hashes first; if unavailable, require explicit preregistered recovery and user authorization before training, and label recovered checkpoints accurately. Save a compact analysis ZIP separately from a preserved checkpoint/contract bundle.

## 8. Next work, in order

1. Implement or locate dedicated runtime benchmark (B0/W3/C2/B1/W4) only, no data/training needed for architecture-only timing. Pin scientific code, same GPU/CPU device, dtype, batch 1, input size, backend and evaluation/inference mode. Record versions, hardware, warmups, repeated/interleaved measurements with synchronization, median/p95, actual sustained throughput if measured, peak memory with proper resets, parameters/state size and operation counting scope. Report variation; do not use one noisy sequential run as definitive. Distinguish random-weight architecture timing from deployment with actual trained/exported checkpoints.
2. Recover and verify original artifacts/checkpoints. Audit prior data exposure and the outer protocol. Write a C2-specific registered confirmation plan; WR-HBP authorization is not transferable. User decides any checkpoint reconstruction or additional training; don't turn packaging into training.
3. Freeze architecture and evaluation plan before final test. Compare against matched B0 and relevant retained comparators as justified; no tuning after test results. If test independence is compromised, report limitations and plan truly external confirmation. Do not promise a new final score.
4. Thesis positioning: empirical evidence for lightweight feature recalibration and accuracy/efficiency tradeoff, not a fabricated novel module. A possible working title is "Optimasi MobileNetV3 Berbasis Frequency-Regulated Channel-Spatial Attention untuk Klasifikasi Fine-Grained Cacat Biji Kopi." Title/novelty not formally approved. Strengthen methodological argument and error analysis; don't use novelty concerns as license for endless module stacking.
5. After any completed experiment, commit canonical result with provenance, update ledger, master results and this handoff. Preserve negative results and deviations.

## 9. AI handoff rules

Respond in Indonesian, academic but direct. Distinguish observed results, interpretations, hypotheses and proposals. Do not invent papers, scores, commits, test passes, or citations. Check original evidence and current branch. Cite primary papers and private experiment artifacts appropriately. Never claim Kaggle runtime tested unless actually executed. Prefer small, auditable next steps, not endless plans. Do not add multi-seed runs, retrain historical baselines merely to obtain already available scores, or rerun completed experiments without a justified new protocol and user agreement. Before training, audit dataset source and fold identity, exact pinned code, config, GPU backward and resume semantics. On Kaggle, original dataset is under /kaggle/input/datasets, not notebook outputs; validate 979/17 and avoid accidental double counting. Package existing outputs without rerunning training, and never delete original checkpoints/logs to shrink output. If a user uploads a result package, analyze it directly rather than request a new run. Use [AGENTS.md](AGENTS.md) for permanent workflow rules.

A new conversation can start with: "Lanjutkan penelitian Coffee17 dari RESEARCH_HANDOFF.md pada branch codex/representation-screening-v1 di ediprin/coffee-bean-classification. Baca AGENTS.md, ledger, master results dan canonical C2 results. Lanjutkan pekerjaan terakhir; jangan mengulang eksperimen atau mengakses outer test tanpa protokol."


## 10. Runtime benchmark implementation checkpoint (2026-11-26)

Runtime-only comparison was executed on Tesla T4; see completed result below. See [registered runtime protocol](docs/experiments/COFFEE17_RUNTIME_HEAD_TO_HEAD_V1_PROTOCOL.md) and [Kaggle notebook](notebooks/Coffee17_Runtime_Head_To_Head_V1_Kaggle.ipynb). Scientific pin `b659a22a72c97bfe75576424a38cbb10076ab0be`. No data or checkpoints needed, no training. The user returned `coffee17-runtime-head-to-head-v1.json`; the results have been audited. Do not repeat profiling by default. Original checkpoint availability and final evaluation provenance remain unresolved.


## Runtime Head-to-Head V1 — completed 2026-11-26

Canonical result: `docs/results/COFFEE17_RUNTIME_HEAD_TO_HEAD_V1_RESULTS.md`. Artifact SHA256 `75deecaf770b97db4a38aed0e29833199ea29bc4b5bb4183f0bfee977ca1016d`. Tesla T4, FP32 eager, synthetic inputs, random weights; no training/dataset/test access. 10 balanced rounds, 500 latency observations per arm. Median ms / throughput img/s / isolated peak MiB: B0 5.789/171.05/19.72; W3 6.797/146.56/19.74; C2 6.028/161.02/19.76; B1 6.694/149.60/27.15; W4 7.658/129.39/27.22. C2 vs B0 +4.13% latency, +0.161% params, +1.62 pp development Macro-F1. C2 vs W3 -11.32% latency, +0.893 pp Macro; vs B1 -9.96% latency, +0.235 pp Macro; vs W4 -21.28% latency, +24.44% throughput, -27.41% peak memory, -16.07% params, -0.151 pp Macro. C2 faster than W3/B1/W4 in all ten rounds. Two-objective Macro-F1/latency frontier: B0, C2, W4. C2 retained and architecture frozen for confirmation, not independently validated. No new training or architecture search authorized. Next audit checkpoint availability and historical image exposure, then register C2-specific evaluation protocol. Runtime numbers are model-only and hardware/session-specific; no end-to-end or independent significance claim.


## C2 evaluation readiness audit — 2026-11-26

See [readiness audit](docs/protocols/COFFEE17_C2_EVALUATION_READINESS_AUDIT_V1.md). Both compact ZIPs omit checkpoint binaries; 25 selected checkpoint hashes exist in fold results. Validation identities B0/C2 match in all five folds (97 each; 394 distinct identities across 485 records). Fold assignment exposes each clean identity in development of four other folds, and prior primary OOF was already opened. Therefore any future same-Coffee17 OOF is **internal retrospective held-out evaluation**, not independent confirmation of architecture selection. Exact historical train/test manifests not included in compact ZIPs and still need verification. C2 remains frozen. Next execute [read-only checkpoint inventory](notebooks/Coffee17_Checkpoint_Inventory_Only_Kaggle.ipynb) against saved master and focused outputs; CPU, no dataset, no training, no checkpoint loading. Return `coffee17-checkpoint-inventory-v1.json`. Original checkpoint availability unresolved; missing files within scanned scope do not prove permanent loss. Primary B0/C2 need 5/5 hash matches each; W4 optional if complete. No outer-test evaluation authorized yet. External generalization requires new independently collected/held data with compatible class labels, not a fresh split of the already exposed population. No new training without explicit authorization.


## Checkpoint inventory outcome — 2026-11-26

Canonical result: `docs/results/COFFEE17_CHECKPOINT_INVENTORY_V1_RESULTS.md`. User JSON SHA256 `f99d68912e4e0aad65a7031c033a4ac0376d85916d4950e1b7654fad34b26bfb`. Zero weight files/members hashed, one ZIP inspected, no scan errors, 0/25 exact matches. Original primary checkpoints not found within mounted scope; not proof of permanent loss. No training or test access. Do not repeat the same scan; check older saved versions/backups if available. If no recovery source remains, propose one matched B0/C2 recovery per fold (10 runs total), only with explicit authorization and frozen recovery-specific protocol. Preserve all weights/logs; recovered checkpoints are not originals. Same Coffee17 OOF remains internal only. C2 architecture unchanged.


## Authorized B0/C2 recovery V1 — 2026-11-26

User confirmed previous mounted outputs were correct and authorized matched recovery. No further inventory repeat. [Protocol](docs/protocols/COFFEE17_B0_C2_RECOVERY_INTERNAL_OOF_V1.md), [notebook](notebooks/Coffee17_B0_C2_Recovery_V1_Kaggle.ipynb), scientific pin `12cc8e614c63968cd2df69db42490f6f3da19450`. B0 and C2 only, five original folds, seed 42, 50 epochs, no outer test. All five historical validation hashes and original pretrained shared-core SHA required before training. Preserve all 20 best/last checkpoints plus contracts and manifests; separate compact analysis ZIP. Recovered scores/weights separate from historical. Return `coffee17-b0-c2-recovery-analysis-v1.zip` and retain saved output. Internal OOF plan registered but evaluator execution locked pending successful artifact audit. No independent confirmation claim, no new architecture search. Actual Kaggle recovery not yet run.


## Recovery audit completed — 2026-11-26

[Canonical recovery results](docs/results/COFFEE17_B0_C2_RECOVERY_V1_RESULTS.md). Source ZIP SHA256 `07e9e796cc6357bfa178f3e46ec85c86a372dd2a2f893dfe1840207209b4ee6f`. All ten 50-epoch histories, non-weight completion hashes, five historical validation hashes, manifest coverage and recomputed prediction metrics pass. B0/C2 mean Macro 89.4684/91.5181%, Hard 84.5785/87.4576%, Worst 58.0513/64.0160%; Macro +2.0497 pp, 4/5 positive, one tie; rescue/damage 11/2, net +9. Recovery C2 Worst lower than historical C2 (70.72%); preserve both. Physical weights absent from compact ZIP; mounted weights and actual image bytes must pass audit before any inference. Next execute [internal OOF notebook](notebooks/Coffee17_B0_C2_Internal_OOF_V1_Kaggle.ipynb), inference-only runner scientific pin `cdef70e2d56eabbe1c83dbdaf2e000b65375176d`, attach original dataset and saved recovery project, not ZIP alone. All 10 exact best and last files checked against audited marker SHA, 979 raw source byte hashes, original selected epochs/classes/config and strict load plus synthetic forward before test inference. Report `coffee17-b0-c2-internal-oof-v1.zip`; no new training. Same-population evaluation is retrospective internal OOF only. No OOF measurements yet; Kaggle runtime not tested. No architecture tuning authorized.
