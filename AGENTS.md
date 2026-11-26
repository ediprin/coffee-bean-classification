# AGENTS.md — Coffee17 research workflow

Scope: this repository, especially branch `codex/representation-screening-v1`.

## On entry to a new session

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md), [experiment ledger](docs/experiments/COFFEE17_EXPERIMENT_LEDGER.md), [master results](docs/experiments/COFFEE17_EXPERIMENT_MASTER_RESULTS.md), and the latest canonical results under `docs/results/`. Inspect relevant configs, code and scientific commit pins before proposing a change. If this guidance conflicts with a newer verified result or an explicit user instruction, explain the conflict; do not silently rewrite history.

Current handoff: C2 MobileNetV3-Large + late FReCSA + GAP is the retained efficient development candidate. Runtime-only head-to-head B0/W3/C2/B1/W4 is complete; C2 is frozen for confirmation. Data-exposure audit completed: same-Coffee17 evaluation is internal only, not independent confirmation. Next run the checkpoint inventory and verify original manifests; final evaluation and methodological novelty remain unresolved. See docs/protocols/COFFEE17_C2_EVALUATION_READINESS_AUDIT_V1.md. The existing WR-HBP outer protocol is not authority to test C2.

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
