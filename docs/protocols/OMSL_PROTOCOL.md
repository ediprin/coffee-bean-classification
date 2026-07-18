# OMSL multi-source protocol

## Scope

Ontology-Marginalized Supervision Learning (OMSL) trains one canonical leaf
classifier from datasets whose observed labels have different granularities.
It does **not** treat Coffee17 as the universal ontology; the canonical leaves
and every dataset observation map are explicit, versioned research inputs.

The probability of an observed label is the sum of its compatible canonical
leaf probabilities. The implemented negative log-likelihood is

```text
-logsumexp(compatible logits) + logsumexp(all logits)
```

This fixed probability aggregation follows the structural precedent in MSeg
(Lambert et al., CVPR 2020). It is a baseline contribution, not a claim that
label marginalization itself is new.

## Taxonomy-compatible contrastive ablation

`OMSL1` adds a supervised contrastive regularizer to the model embedding:

- equal compatible sets are positives;
- disjoint sets are definite negatives;
- overlapping parent-child sets are ignored.

The conservative default prevents the training code from silently assigning a
fine pseudo-label to a coarse observation. `nested_positives: true` exists only
as a pre-registered ablation and must not be mixed into the primary comparison.
The representation-learning motivation is grounded in Grafit (Touvron et al.,
ICCV 2021) and fine-grained angular contrastive learning (Bukchin et al., CVPR
2021), while this exact compatibility rule is the component being tested here.

## Ontology audit gate

Mappings support `exact`, `reviewed`, and `provisional` status. Training rejects
`provisional` mappings unless `allow_provisional: true` is explicitly set. The
checked-in CBD mapping is a draft and therefore emits a warning. It must be
audited against dataset annotation documentation before confirmatory runs.

Within each dataset, mappings must be non-empty and non-overlapping. Partial
coverage is allowed: a dataset need not observe all canonical leaves.

## Locked comparison

Keep the backbone, split, augmentation, seed, and optimizer identical:

1. separate per-dataset CE baseline;
2. shared backbone with dataset-specific heads;
3. `OMSL0`: marginal likelihood only;
4. `OMSL1`: OMSL plus taxonomy-compatible contrastive loss.

Select hyperparameters on validation only. Confirm with seeds 42, 123, and
2026, then open each dataset test split once. Report the unweighted mean across
dataset Macro-F1 as the selection metric, plus every dataset result separately.

## Run

```bash
python -u -m bilinear_lmmd.engine.train_omsl \
  --config configs/omsl/OMSL0_efficientnetv2_gap.yaml \
  --seed 42 --resume

python -u -m bilinear_lmmd.engine.train_omsl \
  --config configs/omsl/OMSL1_efficientnetv2_gap_tc.yaml \
  --seed 42 --resume
```

Outputs are `last.pt`, `best.pt`, `history.json`, and `metrics.json`.

## Primary references

- Lambert et al., *MSeg: A Composite Dataset for Multi-Domain Semantic
  Segmentation*, CVPR 2020.
- Touvron et al., *Grafit: Learning Fine-Grained Image Representations With
  Coarse Labels*, ICCV 2021.
- Bukchin et al., *Fine-Grained Angular Contrastive Learning With Coarse
  Labels*, CVPR 2021.
- Lv et al., *PRODEN: Progressive Identification of True Labels for
  Partial-Label Learning*, ICML 2020 (baseline with different assumptions).
