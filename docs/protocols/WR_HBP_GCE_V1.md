# WR-HBP + Gradient-Boosting Cross Entropy V1

Status: **frozen development protocol; training pending**.

## Research question

Can a loss that concentrates classification gradients on the most confusing
negative classes improve the difficult Coffee17 classes without sacrificing
overall WR-HBP performance?

The matched comparison is:

- control: **WR-HBP + Cross Entropy**;
- candidate: **WR-HBP + Gradient-Boosting Cross Entropy (GCE-LS)**.

Architecture, initialization, data, augmentation, optimizer, scheduler, epoch
budget, checkpoint selection, and inference are identical. Only the training
classification loss differs.

## Literature basis

Sun et al., *Fine-Grained Recognition: Accounting for Subtle Differences
between Similar Classes*, AAAI 2020, DOI 10.1609/aaai.v34i07.6882.

Their GCE loss keeps the ground-truth class and only the top-k highest-scoring
negative classes in the softmax normalization. The stated motivation is that
fine-grained errors are concentrated among a small subset of visually similar
classes, so easy negatives should not dilute the gradient used to resolve those
ambiguities.

This experiment adapts the loss only; it does **not** reproduce the paper's
Diversification Block.

## Frozen Coffee17 adaptation

Coffee17 has 17 classes and therefore only 16 negative classes per sample.
Using the paper's absolute k=15 would be almost standard cross entropy here.
V1 therefore pre-registers **top_k=2** to preserve the intended small confusing
subset. There is no k sweep on validation.

The existing WR-HBP recipe uses label smoothing 0.1. To avoid changing two
factors at once, the candidate applies the same smoothing on the restricted
support consisting of the target plus two selected negatives. With
top_k=num_classes-1 this implementation reduces to ordinary PyTorch cross
entropy with the same label smoothing. With smoothing=0 it reduces to the GCE
equation from Sun et al.

We call this Coffee17 adaptation **GCE-LS**.

## Frozen model and training recipe

Both arms:

- Coffee17 clean preprocessing-development folds;
- folds 1..5;
- seed 42;
- MobileNetV3-Large;
- HBP stages [1,3,4], projection 512;
- WR branch: luminance L1 Haar LH/HL/HH + VisuShrink soft threshold;
- shallow residual injection, hidden width 16, zero-initialized tanh gate;
- input 224;
- batch 32;
- rotations 0/45/90/135/180/225/270;
- AdamW, lr 3e-4, weight decay 1e-4;
- cosine schedule;
- 50 epochs;
- checkpoint selected by validation Macro-F1;
- no EMA;
- outer test untouched.

Control loss:

```
CrossEntropy(label_smoothing=0.1)
```

Candidate loss:

```
for each sample:
    keep ground-truth logit
    select top-2 highest negative logits
    softmax only over target + selected negatives
    apply label smoothing 0.1 on this restricted support
```

No class pair is hard-coded into the loss. Hard negatives are selected
dynamically from the current per-sample logits.

## Strict determinism

This branch inherits the deterministic corrections from
`codex/physical-logit-residual-wr-hbp-v1` at
`13d643c52ec0cf7e9167a188eb6a9c79aedd81d1`.

Required runtime contract:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `torch.use_deterministic_algorithms(True)`;
- cuDNN benchmark off;
- cuDNN deterministic on;
- TF32 off;
- CPU median fallback for CUDA VisuShrink under strict determinism;
- deterministic reshape/block-mean HBP alignment for exact integer grids.

The CE and GCE arms must start from an exactly identical complete WR-HBP state,
not merely the same backbone.

## Targeted diagnostic pairs

The following pairs are diagnostics only and do not alter sampling or loss:

- Full Sour <-> Partial Sour;
- Full Black <-> Partial Black;
- Severe Insect Damage <-> Slight Insect Damage.

For each fold we count symmetric pairwise confusions in both arms.

## Frozen screening gate

GCE-LS passes development screening only if all conditions hold:

1. mean paired Hard-F1 delta > 0;
2. Hard-F1 improves in at least 3/5 folds;
3. mean paired Worst-F1 delta >= 0;
4. mean paired Macro-F1 delta >= 0;
5. total targeted-pair confusion count does not increase.

Failure means **STOP**. Do not tune k, smoothing, LR, loss mixing, pair lists, or
checkpoint metric on these reused development folds.

A pass is only a promotion to a separately frozen confirmation step; it is not
a final superiority claim.

## Claim boundary

This is post-primary exploratory method development on reused Coffee17
development folds. The outer test must remain untouched.

The scientifically allowed interpretation is limited to the tested mechanism:
whether dynamically restricting CE normalization to the two currently hardest
negative classes improves hard-class discrimination for WR-HBP under the frozen
Coffee17 recipe.
