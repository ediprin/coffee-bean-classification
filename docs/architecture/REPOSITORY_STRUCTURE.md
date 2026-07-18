# Repository structure

The repository is organized by responsibility. Generic, reusable code stays
separate from dataset preparation and experiment orchestration.

```text
bilinear-LMMD/
├── configs/
│   ├── backbones/       # backbone comparison
│   ├── cbd/             # CBD multiclass experiments
│   ├── coffee17/        # Coffee17 and adaptation ablations
│   ├── granularity/     # controlled fine/coarse experiment
│   ├── paper/           # paper-protocol reproduction
│   ├── roast/           # roast-level dataset
│   └── usk/             # USK-Coffee benchmark
├── docs/
│   ├── architecture/    # repository and system design
│   ├── protocols/       # preregistered experiment protocols
│   └── results/         # locked summaries and report artifacts
├── notebooks/           # Colab/Kaggle entry notebooks
├── src/bilinear_lmmd/
│   ├── analysis/        # XAI and post-hoc diagnostics
│   ├── core/            # configuration and artifact infrastructure
│   ├── data/
│   │   ├── loaders.py   # runtime datasets and transforms
│   │   └── preparation/ # dataset-specific preparation commands
│   ├── engine/          # generic train, evaluate, and benchmark commands
│   ├── experiments/     # screening and confirmation orchestration
│   ├── modeling/        # models, losses, and label structures
│   └── reporting/       # aggregation, comparison, and OOF merging
└── tests/               # mirrors the source concerns
```

## Dependency direction

Code should follow this direction to avoid circular experiment dependencies:

```text
core ───────────────┐
data ───────────────┼─> engine ─> experiments
modeling ───────────┘       └────> reporting
analysis <──────────────────────── experiments
```

- `modeling` must not import experiment runners.
- `engine` contains only reusable execution logic.
- dataset-specific assumptions belong in `data/preparation` or an experiment.
- experiment runners may compose lower-level modules but should not become
  dependencies of the reusable model or data layers.
- locked results belong in `docs/results`; generated checkpoints remain under
  ignored `outputs/` directories or a remote artifact store.

## Command migration

The flat module namespace was intentionally removed. Use the concern-qualified
commands below:

| Purpose | Command module |
|---|---|
| Train | `bilinear_lmmd.engine.train` |
| Evaluate checkpoint | `bilinear_lmmd.engine.evaluate_checkpoint` |
| Benchmark efficiency | `bilinear_lmmd.engine.benchmark` |
| Prepare a dataset | `bilinear_lmmd.data.preparation.prepare_*` |
| Run an experiment | `bilinear_lmmd.experiments.run_*` |
| Aggregate reports | `bilinear_lmmd.reporting.aggregate_ablation` |

Examples:

```bash
python -m bilinear_lmmd.data.preparation.prepare_coffee17 --output data/coffee
python -m bilinear_lmmd.engine.train \
  --config configs/coffee17/M1_mobilenetv3_hbp_source.yaml
python -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root data/coffee17/folds/fold_1 \
  --output-root outputs/backbones \
  --seeds 42 123 2026
```

## Adding OMSL

The planned ontology-marginalized work should respect the same boundaries:

- ontology schemas and observation mappings: `configs/omsl/`;
- manifest preparation: `data/preparation/`;
- reusable marginal likelihood: `modeling/`;
- generic multi-source batching: `data/`;
- screening and confirmation: `experiments/`;
- final locked evidence: `docs/results/`.
