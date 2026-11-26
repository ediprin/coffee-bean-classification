# Coffee17 B0/C2 internal OOF V1 — audited results

Status: COMPLETE / NO_INTERNAL_SUPPORT. No tuning or rescue training authorized. B0 remains the accuracy/efficiency reference; C2 is the frozen evaluated candidate, not a confirmed superior model.

## Provenance
User ZIP coffee17-b0-c2-internal-oof-v1.zip, SHA256 ec6d17f6426f075a6b5b84e0dd898ef8510569c72ac57384b0b2d827082c3fc3. 45 entries, CRC passed. Scientific commit cdef70e2d56eabbe1c83dbdaf2e000b65375176d. Registered plan docs/protocols/COFFEE17_B0_C2_RECOVERY_INTERNAL_OOF_V1.md. Compared with recovery ZIP: authority SHA and canonical fold manifest fingerprint agree; all ten selected checkpoint hashes match recovery completion markers and OOF fold records. Each fold prediction identity set equals its recovery test assignment; every identity exactly once per arm; 965 paired unique identities, 17 classes. Pooled rows exactly equal concatenated fold rows (sorted by identity); identity-fold maps match. Independently recomputed five metrics from ten fold CSVs and two pooled CSVs, verified correctness flags, finite normalized probabilities and argmax labels, all twelve confusion matrices. Paired counts recomputed. Runtime authority reports 979 original source byte checks, Tesla T4, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, CUDA 12.8, no training. Physical weights and original image bytes not included in uploaded ZIP; their verification is reported by the runtime, not independently performed here. Evaluation is retrospective internal fold-held-out OOF on a previously exposed population, not independent confirmation.

## Pooled results (965 per arm; percent, delta percentage points)
| Metric | B0 | C2 | Delta |
|---|---:|---:|---:|
| Accuracy | 87.0466 | 86.5285 | -0.5181 |
| Balanced accuracy | 86.9110 | 86.5783 | -0.3328 |
| Macro-F1 | 86.8490 | 86.5665 | -0.2826 |
| Hard-F1 | 81.2547 | 80.1174 | -1.1373 |
| Worst-F1 | 65.4545 | 68.9655 | +3.5110 |

Mean fold Macro-F1 B0 86.6453 ±1.0753 pp (SD); C2 86.4796 ±1.2393 pp. Mean fold Hard 80.9941 vs 80.0293; mean fold Worst 57.2807 vs 61.7158. Mean fold and pooled are different estimands; use pooled as registered primary.

| Fold | n | B0 Macro % | C2 Macro % | Delta pp | Rescue | Damage |
|---|---:|---:|---:|---:|---:|---:|
| 1 |199|88.0918|88.5580|+0.4662|5|5|
| 2 |196|87.3071|86.6523|-0.6548|6|8|
| 3 |193|85.3861|85.4867|+0.1006|7|7|
| 4 |191|85.9481|85.7607|-0.1874|4|5|
| 5 |186|86.4937|85.9404|-0.5533|6|8|

Paired outcomes: 807 both correct, 97 both wrong, 28 rescue, 33 damage, net -5 correct. B0 840/965, C2 835/965. No fold accuracy increase (ties folds 1 and 3).

Registered gate: pooled Macro >0 FAIL; >=3/5 positive Macro folds FAIL (2/5); Hard delta >=0 FAIL; Worst delta >=0 PASS. Overall NO_INTERNAL_SUPPORT. Do not reinterpret Worst gain as passing the primary gate.

## Error structure
| Class | B0 F1 % | C2 F1 % | Delta pp |
|---|---:|---:|---:|
| Broken |94.21|92.56|-1.65|
| Cut |83.08|82.81|-0.26|
| Dry Cherry |90.00|94.23|+4.23|
| Fade |73.53|73.53|0|
| Floater |94.74|96.84|+2.11|
| Full Black |100|100|0|
| Full Sour |85.52|85.52|0|
| Fungus Damage |91.78|90.41|-1.37|
| Husk |94.64|96.36|+1.72|
| Immature |83.45|77.70|-5.76|
| Parchment |93.58|94.44|+0.87|
| Partial Black |95.45|92.91|-2.54|
| Partial Sour |76.00|75.51|-0.49|
| Severe Insect Damage |81.08|79.65|-1.44|
| Shell |93.91|92.31|-1.61|
| Slight Insect Damage |65.45|68.97|+3.51|
| Withered |80.00|77.88|-2.12|

5 classes improve, 9 decline, 3 tie. Slight Insect Damage remains worst in both arms, recall 65.45% to 72.73%, precision 65.45% to 65.57%. Insect group +1.04 pp, shape/withered -2.71 pp, sour/black -1.01 pp. Symmetric confusion counts B0/C2: Withered–Immature 12/14, Severe–Slight insect 16/14, Cut–Slight insect 7/6, Partial–Full sour 14/15, Slight insect–Fade 8/7, Full–Partial black 0/0. Not causal proof of feature suppression or improved texture extraction.

## Development versus OOF
Historical screening C2-B0 +1.62 pp Macro; matched recovery +2.05 pp Macro; OOF -0.28 pp Macro. Recovery mean validation Macro B0 89.47, C2 91.52; pooled OOF B0 86.85, C2 86.57. These are different subsets/aggregation, not a directly comparable generalization gap estimate. Development evidence did not transfer to registered internal evaluation. Runtime benchmark showed C2 +0.161% params, +4.13% median latency vs B0; with OOF Macro as objective B0 is faster and higher scoring than C2. Do not use development Pareto frontier as final OOF frontier. W3/B1/W4 lack matched OOF and cannot be ranked by that endpoint.

## Decision
Stop C2 superiority claim and any post-test rescue on the same evaluation. No independent test, no equivalence or significant inferiority claim. Preserve all original and recovered artifacts. A descriptive error analysis/Grad-CAM is permissible on existing predictions and saved checkpoints only, without using observations to tune a model. Thesis framing must be reconsidered: empirical analysis of representation/efficiency tradeoffs and negative confirmation, or a genuinely new independently evaluated research protocol if a superior novel model is required. Do not silently substitute another screened candidate based on opened OOF. No new architecture or training run was performed in this audit.
