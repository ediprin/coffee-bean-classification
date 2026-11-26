# Coffee17 matched B0/C2 checkpoint recovery V1 — audited development results

Date: 2026-11-26. Status: RECOVERY_ANALYSIS_PASS / OOF_INFERENCE_PENDING_MOUNTED_CHECKPOINT_VERIFICATION.
Source: coffee17-b0-c2-recovery-analysis-v1.zip, SHA256 07e9e796cc6357bfa178f3e46ec85c86a372dd2a2f893dfe1840207209b4ee6f.
Protocol: docs/protocols/COFFEE17_B0_C2_RECOVERY_INTERNAL_OOF_V1.md.
Scientific pin: 12cc8e614c63968cd2df69db42490f6f3da19450.

## Artifact audit
83 ZIP entries, CRC passed. All 10 arm histories contain exactly epochs 1–50. Recomputed accuracy, balanced accuracy, Macro-F1, eight-class Hard-F1 and Worst-F1 from all ten validation prediction CSVs: match reported values. Selected first maximum validation Macro-F1 agrees with reported validation metric in each arm. Completed marker hashes match all 60 non-weight files listed in the markers. Ten best and ten last SHA256 entries are distinct; best hashes agree with fold results. Checkpoint bytes are not included in the analysis ZIP and cannot be verified here. They must be checked against the audited completion markers in the saved Kaggle output before inference.
Canonical clean/fold manifest SHA256 agree with authority (7b1af9eecef18171a12d57f68dbfb7329a77c69f16354a0555366b537cacdc3c / 639b4d248afbf1e8261d9e7b811ca44ffc3ffb40a51256a1d424b6c749b95fe8). 979 raw, 965 clean, 965 distinct exact content hashes, 17 classes. Per-fold train/val/test identities mutually exclusive; test union equals 965 clean identities, no overlap between test bins. Counts: 669/97/199, 672/97/196, 675/97/193, 677/97/191, 682/97/186. Validation identity fingerprints match historical values for all five folds; validation prediction identities and labels match manifest. Across folds 485 validation rows represent 394 distinct images. All ten canonical contract and config hashes verified; authority and contracts report historical pretrained core matched, outer_test_accessed=false. Historical full content hash unavailable, so not proof of byte-for-byte equivalence to original historical dataset.
Runtime reported Tesla T4 cuda:0 (2 visible GPUs), torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, Python 3.13.15, CUDA 12.8; cuDNN deterministic false, TF32 matmul false, TF32 cuDNN true. This is a single seeded matched recovery, not strict determinism or a new independent multi-seed study.

## Mean development-fold results (percent)
| Metric | B0 recovery | C2 recovery | Delta pp | Positive folds |
|---|---:|---:|---:|---:|
| Accuracy | 89.8969 | 91.7526 | +1.8557 | 4/5 |
| Balanced accuracy | 89.6092 | 91.7661 | +2.1569 | 4/5 |
| Macro-F1 | 89.4684 | 91.5181 | +2.0497 | 4/5 |
| Hard-F1 | 84.5785 | 87.4576 | +2.8791 | 3/5 |
| Worst-F1 | 58.0513 | 64.0160 | +5.9647 | 3/5 |

| Fold | B0 Macro % | C2 Macro % | Delta pp | B0 selected epoch | C2 selected epoch |
|---|---:|---:|---:|---:|---:|
| 1 | 91.2796 | 92.3745 | +1.0949 | 44 | 30 |
| 2 | 93.6585 | 93.6585 | 0 | 37 | 43 |
| 3 | 90.0386 | 92.2709 | +2.2323 | 34 | 22 |
| 4 | 83.4047 | 89.2548 | +5.8501 | 16 | 46 |
| 5 | 88.9605 | 90.0316 | +1.0711 | 32 | 25 |

Macro SD across folds B0 3.8141 pp, C2 1.8175 pp. Rescue 11, damage 2, both correct 434, both wrong 38, net +9. Fold 4 contributes over half total Macro delta; without fold 4, mean delta remains +1.0996 pp. Hard-F1 declines in fold 1 (-0.199 pp), ties fold 2, improves three others. Worst-F1 declines fold 5 (-9.524 pp), ties fold 2, improves three others. No fold Macro decline. These are descriptive; overlapping validation identities and reused development population preclude independent-sample significance claims.

Mean per-class F1 delta C2-B0 (pp): Fade +6.095, Slight Insect Damage +5.766, Withered +4.829, Full Sour +3.399, Partial Sour +2.626, Severe Insect Damage +2.564, Dry Cherry +2.222, Immature +2.128, Husk +1.818, Cut +1.744, Fungus Damage +1.176, Broken +0.500, Floater/Full Black/Parchment/Shell 0, Partial Black -0.023. Gains in insect/fade/withered recur descriptively; not causal attribution.

Historical focused C2: Macro 91.36, Hard 87.38, Worst 70.72, accuracy 91.55. Recovery C2 Macro +0.15 pp, Hard +0.08 pp, Worst -6.70 pp approximately; report separately, no substitution for original weights. Historical B0 Macro 89.74, Hard 85.20, Worst 66.15. Recovery paired gain +2.05 Macro versus historical +1.62. No independent confirmation or architecture tuning.

## Decision
Recovery analysis passes. C2 frozen. Next execute the registered internal OOF evaluation for B0 and C2 only, conditional on mounted saved checkpoint and source-image byte verification. No new training. Use all 965 identities exactly once per arm, no TTA, model selection, or ensemble. This is retrospective internal held-out evaluation, not external generalization. Any actual checkpoint mismatch or missing file blocks inference; do not reconstruct again automatically.
