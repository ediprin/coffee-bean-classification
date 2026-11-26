# C2 final evaluation readiness audit V1 — 2026-11-26

Status: BLOCKED_CHECKPOINT_INVENTORY / INTERNAL_ONLY_COFFEE17.
No training, checkpoint deserialization, image materialization or outer-test predictions performed.

## Evidence checked

- Master compact ZIP SHA256 8d5c304c72ec8e1000f1a32e3ca0faa5dc6093466443f4c179068d40097272a3 (699 entries).
- Focused compact ZIP SHA256 70365b4661d0a9b8509d68cd3ccfe58ac524dfbf54ed93f20512a44feca72a32 (144 entries).
- Neither archive has checkpoint files (.pt/.pth/.safetensors). Each relevant fold result records best_checkpoint_sha256. No claim that original files are permanently lost.
- Parsed B0 and C2 validation predictions: canonical class/filename and actual label sets agree in all five folds; each contains 97 distinct identities. Recorded validation identity-label SHA agrees between master and focused fold results. Across folds 485 records represent 394 identities: 311 occur once, 75 twice, 8 three times. Pairwise intersection counts:

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 97 | 10 | 9 | 12 | 10 |
| 2 | 10 | 97 | 10 | 3 | 8 |
| 3 | 9 | 10 | 97 | 12 | 14 |
| 4 | 12 | 3 | 12 | 97 | 11 |
| 5 | 10 | 8 | 14 | 11 | 97 |

- Audited `src/bilinear_lmmd/data/preparation/prepare_preprocessing_folds.py`, `materialize_preprocessing_development.py`, existing WR-HBP outer protocol, and master results. Exact SHA deduplication is implemented; do not imply biological specimen groups or perceptual near-duplicate grouping were verified. The source archives do not include clean/fold manifests; exact train/test identities/content and historical manifest correspondence must be checked when available.

## Data independence decision

The fold generator stratifies identities into five disjoint outer test bins. For each outer fold, the remaining identities are split into train and validation (~70/10/20 overall); test union is the complete clean population. Thus every identity is in development of four other folds. Fold-specific train/val/test separation is compatible with the source, but the complete population was exposed to the development process across folds. Additionally master results explicitly record an earlier opened preprocessing OOF of 965 images. Exact identity correspondence with that historical OOF has not been independently reconstructed here. Neither a new random split of the same clean Coffee17 nor a new seed repairs prior population-level exposure. Do not label this independent confirmation or nested CV of the full selection procedure. Model-selection bias is a known issue: Cawley & Talbot (2010), https://www.jmlr.org/papers/v11/cawley10a.html ; Varma & Simon (2006), https://doi.org/10.1186/1471-2105-7-91 .

An internal retrospective locked-architecture five-fold OOF evaluation may still quantify fold-specific held-out prediction performance, subject to provenance checks; it does not constitute external confirmation. An independent new population with a prospectively defined compatible 17-class label mapping and no selection/training exposure is required for the stronger claim. No external dataset is yet established. Do not silently reduce classes or merge datasets.

## Recovery gate

Notebook: `notebooks/Coffee17_Checkpoint_Inventory_Only_Kaggle.ipynb`.
Read-only CPU scan of attached Kaggle saved output directories and direct ZIP members; 25 exact recorded best-checkpoint hashes (B0, C2, W3, B1, W4 x five folds), no deserialization or training. Primary required matches B0 and C2 5/5 each. W4 is a useful optional comparator if all five exact checkpoints exist, W3/B1 optional only. A missing match within mounted scope does not prove permanent loss; check other versions/local backups. Inventory report is not evaluation authorization. Source checkpoint hashes are embedded in the notebook, derived from the two ZIP fold_result files. No original artifact is deleted or rewritten. Notebook authored and repository file verified separately; Kaggle execution is pending.

## Conditional C2 evaluation plan (NOT AUTHORIZED)

1. Recover original B0/C2 best.pt per fold; compare bytes against expected SHA. Preserve original contracts, logs and checkpoints. If not recoverable, report blocker. A single registered recovery run, if later explicitly authorized by the user, must preserve the original split, exact architecture, training contract and selected checkpoint rule; recovered weights and development metrics must be labeled separately. No strict determinism change or new search by default.
2. Verify exact clean and fold manifest fingerprints, all classes, identity/hash separation within each fold, train/val contract correspondence, exact selected checkpoints, inference normalization and class order. Verify actual prediction coverage once per clean identity per arm. Test labels cannot determine checkpoint or architecture. Test predictions are not authorized until a dedicated evaluator and immutable C2 protocol are reviewed.
3. Internal OOF scope if executed: B0 and C2, optionally W4 only if all exact checkpoints available and declared before test. Existing frozen seed 42, original per-fold checkpoint, no ensemble or TTA, no tuning. Primary pooled Macro-F1 difference C2 minus B0, secondary Accuracy, Balanced Accuracy, Hard-F1 (same eight classes), Worst-F1, per-fold metrics, confusion and paired rescue/damage. Report fold mean separately from pooled metric. Prespecify descriptive decision: pooled Macro delta >0, positive Macro delta in >=3/5 folds, pooled Hard and Worst not worse than B0. Call outcome internal support / no internal support, not independent confirmation. No naive independent-sample p-value or claim of selection-adjusted CI. Keep any future external test separate with frozen model and explicit data/label/provenance protocol.
4. If independence is required for thesis, seek genuinely new data; no claim it is available until verified. Do not access test images or train a new model solely to bypass this audit.

C2 remains frozen as primary architecture from development + runtime evidence. No novel FReCSA invention claimed. The WR-HBP confirmation protocol does not transfer to C2.


## Checkpoint inventory outcome — 2026-11-26

Canonical result: `docs/results/COFFEE17_CHECKPOINT_INVENTORY_V1_RESULTS.md`. User JSON SHA256 `f99d68912e4e0aad65a7031c033a4ac0376d85916d4950e1b7654fad34b26bfb`. Zero weight files/members hashed, one ZIP inspected, no scan errors, 0/25 exact matches. Original primary checkpoints not found within mounted scope; not proof of permanent loss. No training or test access. Do not repeat the same scan; check older saved versions/backups if available. If no recovery source remains, propose one matched B0/C2 recovery per fold (10 runs total), only with explicit authorization and frozen recovery-specific protocol. Preserve all weights/logs; recovered checkpoints are not originals. Same Coffee17 OOF remains internal only. C2 architecture unchanged.
