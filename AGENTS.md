# AGENTS.md — Coffee17 research workflow

Scope: this repository, especially branch `codex/representation-screening-v1`.

## On entry to a new session

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md), [experiment ledger](docs/experiments/COFFEE17_EXPERIMENT_LEDGER.md), [master results](docs/experiments/COFFEE17_EXPERIMENT_MASTER_RESULTS.md), and the latest canonical results under `docs/results/`. Inspect relevant configs, code and scientific commit pins before proposing a change. If this guidance conflicts with a newer verified result or an explicit user instruction, explain the conflict; do not silently rewrite history.

Current handoff: user authorized B0 SAM reference only, five development folds. See protocol and cross-branch audit; C2 and frozen prototype remain stopped. No other training, tuning or test inference authorized.

## Research integrity

- Answer in Indonesian, precise and direct. Separate measured facts, interpretations, hypotheses and proposals. Verify literature claims against primary papers. Never invent experimental results, citations, test passes or commit identifiers.
- Consult the ledger and search close synonyms before proposing a method. Respect FAIL_STOP/CLOSED/BLOCKED_RESULT_MISSING. Missing results are not evidence of untested methods. No near-duplicate rescue or module stacking by default.
- Keep the public Coffee17 17-class classification scope and matched protocol unless the user explicitly authorizes a scientifically justified change. Do not mix results from different protocols or call development metrics final test metrics.
- Preserve identity/group separation, exact split manifests, seed, normalization and checkpoint selection. Audit historical-anchor provenance and runtime differences before comparisons. Avoid independent-sample inference on overlapping validation folds.
- Before final evaluation, audit all earlier data exposure (including already-opened primary preprocessing OOF); require a frozen model-specific evaluation protocol. Never tune, select checkpoints or retrain from test results. Do not call a reused population independent external data.
- Report computational counts accurately (partial MAC estimate is not full FLOPs), measured latency and its environment, and uncertainty. Never infer real deployment efficiency solely from parameter counts.

## Execution and artifact safety

- Do not retrain completed arms merely to reproduce existing reported scores. Use saved outputs and inspect compact packages directly. No new multi-seed experiments by default.
- Separate analysis/profiling from training. A packaging, analysis, or latency-only notebook must not invoke any training path.
- Preserve original `best.pt`, `last.pt`, run contracts, logs, histories and prediction artifacts. Never delete original checkpoints or logs to make a compact ZIP. Package only the required files into a separate staging directory; preserve a separate checkpoint bundle.
- If exact checkpoints are missing, report the gap; seek original saved outputs first. Any recovery training needs a registered, unchanged protocol and explicit authorization before execution. Do not represent recovered or random weights as original trained checkpoints.
- On Kaggle, use the original Coffee17 dataset input, not copies in notebook outputs; verify 979 original images/17 classes, provenance, and fold SHA. Handle changed mount layouts explicitly, do not guess or silently fall back to another dataset. Pin scientific commit, validate config, perform forward/backward preflight on target device, and fail safely on mismatch. Do not claim real Kaggle execution merely from static checks.
- Resume only matching run contracts. Do not overwrite completed results or invalidate resume markers during packaging. Prefer small, auditable experiments and verified exact links. Disclose blockers and untested portions.

## After work

For every completed experiment, record artifacts, scientific commit, protocol, metrics, negative evidence, limitations and decision under `docs/results/`. Update the ledger, master results and RESEARCH_HANDOFF.md before proposing another experiment. Keep the handoff concise enough for a fresh session but link to full evidence. Do not fabricate missing raw artifacts or promote a hypothesis to an empirical result.


## Recovery audit completed — 2026-11-26

[Canonical recovery results](docs/results/COFFEE17_B0_C2_RECOVERY_V1_RESULTS.md). Source ZIP SHA256 `07e9e796cc6357bfa178f3e46ec85c86a372dd2a2f893dfe1840207209b4ee6f`. All ten 50-epoch histories, non-weight completion hashes, five historical validation hashes, manifest coverage and recomputed prediction metrics pass. B0/C2 mean Macro 89.4684/91.5181%, Hard 84.5785/87.4576%, Worst 58.0513/64.0160%; Macro +2.0497 pp, 4/5 positive, one tie; rescue/damage 11/2, net +9. Recovery C2 Worst lower than historical C2 (70.72%); preserve both. Physical weights absent from compact ZIP; mounted weights and actual image bytes must pass audit before any inference. Next execute [internal OOF notebook](notebooks/Coffee17_B0_C2_Internal_OOF_V1_Kaggle.ipynb), inference-only runner scientific pin `cdef70e2d56eabbe1c83dbdaf2e000b65375176d`, attach original dataset and saved recovery project, not ZIP alone. All 10 exact best and last files checked against audited marker SHA, 979 raw source byte hashes, original selected epochs/classes/config and strict load plus synthetic forward before test inference. Report `coffee17-b0-c2-internal-oof-v1.zip`; no new training. Same-population evaluation is retrospective internal OOF only. No OOF measurements yet; Kaggle runtime not tested. No architecture tuning authorized.


## Final internal OOF outcome — 2026-11-26

[Audited results](docs/results/COFFEE17_B0_C2_INTERNAL_OOF_V1_RESULTS.md). ZIP SHA256 `ec6d17f6426f075a6b5b84e0dd898ef8510569c72ac57384b0b2d827082c3fc3`. All 965 test identities per arm matched recovery manifest, 10 selected weight hashes matched recorded recovery markers, recomputed metrics and confusion matrices matched. Runtime reports 979 source-byte checks; physical weights/images not in analysis archive. Pooled B0/C2 Macro 86.8490/86.5665%, Hard 81.2547/80.1174%, Worst 65.4545/68.9655%, accuracy 87.0466/86.5285%. Macro delta -0.2826 pp, 2/5 folds positive, Hard -1.1373 pp, Worst +3.5110 pp, rescue/damage 28/33 net -5. **NO_INTERNAL_SUPPORT** (three of four criteria fail). C2 is the frozen evaluated candidate, not a confirmed superior model. B0 remains accuracy/latency reference. No post-test tuning, rescue training, candidate substitution or independent confirmation claim. Next: descriptive error analysis and optional XAI with preserved checkpoints, reconsider thesis framing and novelty; genuinely new independent protocol/data required if pursuing a superior new model. No new training authorized.


## Cross-branch anti-loop audit — 2026-11-26

Read the [cross-branch audit](docs/audits/COFFEE17_CROSS_BRANCH_ANTI_LOOP_AUDIT_20261126.md) and the [historical master log](https://github.com/ediprin/coffee-bean-classification/blob/8570468435926eb66ec7a20e482905fb046a7658/docs/results/EXPERIMENT_MASTER_LOG.md) before any new proposal. Earlier active ledger omitted older branches. ConvNeXtV2-Atto, EfficientNetV2, MobileNetV4, PVTv2 and SHViT already screened; ArcFace, vanilla SupCon and confusion-aware SupCon, progressive multigranularity, hierarchy, MPN-COV, EMA, DSConv, Chang-Liu MDE, adaptive multistage recalibration and earlier EfficientNet DCL all tested and stopped per original gates. E1 local MoE, M1s spatial HBP and O1 crop now have recovered historical result summaries (all FAIL_STOP); raw artifacts not independently verified. C2 final OOF failure supersedes development KEEP. SNI instance-crop v1/v2/v3 history exists; v3.1 Jiao protocol recorded not run, public-with-paper requirement unresolved. No new training, backbone sweep or method selected. Research direction HOLD pending genuinely distinct, source-checked hypothesis and evaluation plan. Historical scores belong to their original protocols; archived transcriptions are not raw-prediction reaudits.


## B0 SAM reference V1 — authorized implementation, 2026-11-26

[Protocol](docs/protocols/COFFEE17_B0_SAM_REFERENCE_V1.md), [notebook](notebooks/Coffee17_B0_SAM_Reference_V1_Kaggle.ipynb), scientific pin `a539a776b7960c9622fd59280d17252a0c27401a`. One new arm B0_SAM (5 development folds, seed42, 50 epochs, rho .05, AdamW), identical B0 architecture and original loader, no B0 retraining and no outer-test inference. Attach original Coffee17 and audited saved recovery output; historical weights not needed for baseline comparison. Mandatory source/manifest and B0 contract hashes, numerical-stack check, CPU/GPU SAM tests and B0 GPU preflight. Extracted core CPU fixture passed; full Kaggle execution not verified. Preserve all 10 best/last checkpoints, history and logs. Return `coffee17-b0-sam-reference-analysis-v1.zip`. Diagnostic gate Macro positive, >=3/5 folds, Hard and Worst nonnegative. No rho tuning, stacking or test inference automatically. SAM reference is not novelty; independent data unresolved. No new empirical classification results yet.
