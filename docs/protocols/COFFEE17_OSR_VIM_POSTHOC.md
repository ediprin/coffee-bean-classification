# Coffee17 OSR ViM post-hoc protocol

## Decision context

The EfficientNetV2-B0 + HBP negative control and ARPLoss-no-CS fail-fast both
failed on Coffee17 semantic OSR. ViM is the final low-cost candidate because it
reuses the frozen EfficientNetV2-B0 + GAP + CE checkpoints. It does not train a
new backbone or alter known-class predictions.

Primary references:

- Wang et al., *ViM: Out-of-Distribution with Virtual-logit Matching*, CVPR
  2022: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html
- Official implementation: https://github.com/haoqiwang/vim

## Frozen implementation

For classifier weight `W`, bias `b`, embedding `x`, and known-train logits:

1. classifier origin: `u = -pinv(W) b`;
2. fit the uncentred empirical covariance of `x_train - u`;
3. use the official principal-dimension rule: 1000 when `F >= 2048`, 512
   when `F >= 768`, otherwise `F // 2`;
4. residual norm: `v(x) = ||(x-u) R||_2`, where `R` is the complement of the
   principal subspace;
5. virtual-logit scale:
   `alpha = mean(max(logits_train)) / mean(v(x_train))`;
6. knownness score (the sign-reversed OOD score from the paper):
   `logsumexp(logits) - alpha * v(x)`.

Only known-training embeddings fit the origin, residual space, and alpha. The
operational rejection threshold is fitted only from known-validation at 95%
known acceptance. Unknown-test samples are used only to calculate metrics.

## Comparison and gate

- Same checkpoints and known-class predictions for MSP and ViM.
- Tiers: near, medium, and far; near/medium are the primary decision tiers.
- Seeds: 42, 123, and 2026.
- Primary metrics: balanced-test AUROC and OSCR.
- A primary tier passes only when:
  - mean AUROC gain over MSP is at least 2 percentage points;
  - mean OSCR gain is positive; and
  - AUROC improves in at least two of three seeds.

If neither near nor medium passes, ViM stops and the frozen result remains
EfficientNetV2-B0 + GAP + CE + MSP. A failure must not trigger another
unregistered training method.

## Run

```bash
python -u -m bilinear_lmmd.experiments.run_osr_vim_screening \
  --data-root /content/coffee17-osr-data/clean/folds/fold_1 \
  --prepared-root /content/coffee17-osr-prepared \
  --output-root /content/drive/MyDrive/coffee17-osr-v1 \
  --seeds 42 123 2026
```

The runner requires existing `OSR0_<tier>_seed<seed>/best.pt` checkpoints and
performs inference/PCA only—no optimizer or training epoch is run.
