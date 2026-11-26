# Coffee17 Prototype Geometry V1 — audited results

Decision: NO_CLEAR_FROZEN_MULTICENTER_SIGNAL. Stop frozen K1/K2 decoder path; no automatic K sweep or end-to-end multi-prototype training.
Source: coffee17-prototype-geometry-v1.txt, SHA256 a33b6ce8feea236a74c0f91e815f4650f68266145ad4cf7c453af768b6cf3154.
Protocol: docs/protocols/COFFEE17_PROTOTYPE_GEOMETRY_V1.md. Scientific source pin cdef70e2d56eabbe1c83dbdaf2e000b65375176d.

## Audit
Parsed full JSON, independently recomputed all five metrics for all 15 fold-arm combinations and their means from 485 validation rows (394 distinct identities). All match. All five B0 selected weight hashes, validation identities, labels and linear predictions match recovery analysis ZIP. All 85 class-fold occupancies sum to recorded class train counts; sums match each fold train size. Physical weights, raw images and embeddings not in report, so prototype fitting and source/weight verification cannot be independently repeated here. No neural training or new outer-test inference reported. Reused development observations, not independent evidence.

## Mean fold scores (%)
|Metric|B0 linear|K1|K2|K2-K1 pp|K2-B0 pp|
|---|---:|---:|---:|---:|---:|
|Accuracy|89.8969|88.0412|88.4536|+0.4124|-1.4433|
|Balanced accuracy|89.6092|87.7367|88.2857|+0.5490|-1.3235|
|Macro-F1|89.4684|87.5163|88.1177|+0.6015|-1.3506|
|Hard-F1|84.5785|81.0568|80.9534|-0.1034|-3.6252|
|Worst-F1|58.0513|54.6667|55.2727|+0.6061|-2.7786|

|Fold|B0 Macro|K1 Macro|K2 Macro|
|---|---:|---:|---:|
|1|91.2796|88.2223|89.5132|
|2|93.6585|91.7817|91.6460|
|3|90.0386|87.8179|88.9828|
|4|83.4047|81.8823|81.6689|
|5|88.9605|87.8770|88.7779|

K2 vs K1 rescue/damage 7/5, both correct 422, both wrong 51, net +2. K2 vs B0 3/10, both correct 426, both wrong 46, net -7. K1 vs B0 0/9. K2 Macro below linear in every fold. Gate K2-K1 positive Macro PASS, >=3/5 positive PASS, Hard nonnegative FAIL.

## Geometry
85 class-fold pairs: K2 train mean cosine increases in all 85, unweighted mean +0.0372103. This is training-set fitting evidence, not evidence of discrimination on validation. 12/85 smaller clusters contain one sample; 32/85 contain at most three samples; 32/85 smaller than 10% of class training count. Immature, Partial Black, Partial Sour, Slight Insect Damage have minimum cluster counts >=14 in every fold. Full Sour has singleton clusters in folds 2–4, Cut folds 1–2, Severe Insect Damage folds 1 and 3, Shell folds 3 and 5, etc. No cluster membership images or embedding values supplied; cannot infer physical subtypes, outliers or label errors from occupancies alone.
Conclusion: heterogeneous frozen embedding geometry, extra train fit does not translate into improved hard-class discrimination or superiority to linear classifier. Negative result does not falsify learned end-to-end metric learning; no automatic rescue. Next recommendation (not authorized training): one accuracy-first alternative backbone baseline under matched development split, after active branch audit and independent evaluation plan. No novelty claim for backbone swap. No new training, OOF access or implementation performed here.
