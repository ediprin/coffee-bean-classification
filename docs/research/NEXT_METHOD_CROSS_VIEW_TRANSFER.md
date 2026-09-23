# Next Method Research: Cross-View Transfer After MVCE-SBN

Last updated: 2026-09-23

## Empirical trigger

The current experimental chain establishes:

- fixed R0/C0/F0/W0 preprocessing does not yield a consistent aggregate gain;
- late probability fusion proves real class-dependent complementarity;
- vanilla frozen-teacher KD does not transfer that complementarity into a raw-RGB student;
- naive fully shared multi-view CE fails because auxiliary views are normalized with R0 BatchNorm statistics;
- selective BatchNorm (SBN) removes the F0 optimization collapse:
  mean F0 CE at the best epoch changes from 2.022 to 0.629;
- after fixing normalization, MVCE-SBN recovers Macro-F1 from 88.93% to 90.68%, but remains effectively tied with the matched R0 control at 90.65%, while Hard-F1 is still lower.

Therefore the unresolved problem is no longer whether auxiliary transformed views can be optimized. They can. The remaining problem is transferring useful C0/F0/W0 information into the R0 deployment representation without multi-branch inference.

## Most directly relevant literature

### 1. Zhang et al., CVPR 2020 — Auxiliary Training: Towards Accurate and Robust Models

This is currently the closest method match.

Core design:
- one shared convolutional feature extractor;
- a primary classifier trained on clean images;
- separate auxiliary classifiers trained on transformed/corrupted images;
- selective batch normalization so transformed views use their own batch statistics while deployment statistics are driven by clean images;
- input-aware self-distillation between the primary and auxiliary outputs;
- a late L2 classifier-weight merging stage so auxiliary classifiers approach the primary classifier;
- auxiliary classifiers are discarded at inference.

Paper objective:

L1 = sum_j alpha_j CE(g_j(f(T_j(x))), y)

Omega(theta_g^0, theta_g^j)
  = KL(g_0(f(T_0(x))), g_j(f(T_j(x))))
    + gamma ||theta_g^0 - theta_g^j||_2^2

L = L1 + lambda sum_{j>0} Omega(theta_g^0, theta_g^j)

The paper's CIFAR100 ablation reports:
- complete auxiliary training: 79.47% accuracy;
- without selective BN: 76.37%;
- without self-distillation: 78.44%;
- without attention: 77.50%;
- without weight merging: 78.32%.

The published experiments state alpha_0 = 1, alpha_aux = 0.05, lambda = 0.05 and gamma in {0,1}.

Important implementation detail from the authors' released code:
- clean logits are detached and used as the target for auxiliary-output distillation;
- each auxiliary CE is weighted 0.05;
- each auxiliary distillation term is weighted 0.05;
- classifier-weight L2 merging is enabled only late in training;
- the released CIFAR code uses 210 epochs and starts merging at epoch 180, whereas the paper text reports a 300-epoch CIFAR schedule. Therefore the exact schedule should not be copied blindly into Coffee17.

This method is attractive because our own SBN result independently validates the normalization part on Coffee17.

### 2. Black & Souvenir, WACV 2024 — Multi-view Classification Using Hybrid Fusion and Mutual Distillation

Relevant finding:
- multi-view training can improve later single-view inference;
- their Hotels-8k single-view top-1 rises from 0.463 to 0.498 and Google LandmarksV2 from 0.818 to 0.851;
- their ablation shows that a one-way direction using the multi-view prediction to teach the single-view prediction reaches 0.499 single-view top-1, essentially the same as full mutual distillation at 0.498.

This supports a later fallback in which a training-only fused multi-view target explicitly supervises R0.

However, their model uses a hybrid CNN-Transformer multi-view fusion module and natural image views. It is therefore less direct than Zhang et al. for the next Coffee17 experiment.

### 3. Dong et al., Neurocomputing 2026 — Multi-View Consistency Distillation

Relevant design:
- a training-only multi-view teacher aggregates multiple views;
- a standard single-view student is supervised by feature-level consistency;
- all teacher modules are discarded at deployment;
- inference complexity remains that of the baseline student.

Their teacher uses saliency purification, cross-view patch alignment and reliability-guided aggregation. This directly supports the principle of training-time privileged multi-view transfer, but the full design is too heavy and task-specific for the next Coffee17 experiment.

### 4. Liu et al., Neural Networks 2024 — Spectral Decomposition and Transformation

Relevant observation:
- frequency-domain augmented images can have a substantially different distribution from original images;
- the method uses separated batch normalization for original and frequency-augmented streams;
- the original-image branch is retained for downstream use.

This independently supports retaining SBN as a fixed foundation rather than returning to shared R0 running statistics.

## Next experiment selected for development

Working name:

**AT-SBN: Auxiliary-Training Selective-BN**

This experiment should test the Zhang-style mechanism before introducing a heavier fused teacher.

### Architecture

Shared:
- MobileNetV3-Large feature extractor;
- the already validated selective-BN behavior.

Heads:
- h_R: primary 17-class linear classifier for R0;
- h_C: auxiliary 17-class linear classifier for C0;
- h_F: auxiliary 17-class linear classifier for F0;
- h_W: auxiliary 17-class linear classifier for W0.

No attention module, feature-fusion block, class-dependent routing, or additional inference module in the first Coffee17 adaptation.

Reason for omitting the paper's auxiliary attention/bottleneck initially:
- the immediate Coffee17 hypothesis is classifier/domain decoupling plus explicit transfer;
- adding attention simultaneously would confound the diagnosis;
- the paper's own ablation still leaves a measurable benefit without attention, although smaller than the full method.

### Stage-1 objective

Let

z_R = h_R(f(R0(x)))
z_v = h_v(f(v(x))), v in {C,F,W}

with SBN applied exactly as validated in MVCE-SBN.

Use:

L_primary = CE(z_R, y)

L_aux = sum_v CE(z_v, y)

L_KD = sum_v KL(sg(softmax(z_R / tau)) || softmax(z_v / tau))

L_stage1 = L_primary + alpha L_aux + beta tau^2 L_KD

where sg denotes stop-gradient on the raw-primary target.

This keeps R0 as the primary classifier while auxiliary tasks reshape the shared feature extractor.

### Stage-2 classifier merging

Late in training add:

L_merge = sum_v (
  ||W_v - W_R||_2^2 + ||b_v - b_R||_2^2
)

L_stage2 = L_stage1 + gamma L_merge

At inference:
- discard h_C, h_F, h_W;
- use R0 only;
- retain only R0 running BN statistics;
- no extra inference parameters or forward passes.

### What must be logged

In addition to the usual validation metrics:
- per-view CE;
- per-view top-1 disagreement;
- pairwise JS divergence between heads;
- primary-vs-aux classifier weight distance;
- R0 rescue/damage relative to the frozen matched R0 control;
- per-class F1 delta;
- best epoch;
- whether F0 remains optimized normally under SBN.

This allows us to distinguish:
1. auxiliary-head specialization,
2. cross-view diversity,
3. successful transfer into R0,
4. class-specific interference.

## Decision logic

AT-SBN should not be called successful merely because auxiliary losses are low.

The relevant endpoint is the R0-only primary classifier.

Evidence in favor requires:
- R0-only validation Macro-F1 to improve over the frozen matched R0 control in a reasonably consistent fold pattern;
- no severe Hard-F1 degradation;
- F0 optimization remains healthy;
- auxiliary-head diversity does not collapse immediately;
- inference remains identical to the one-model R0 path.

If AT-SBN only improves auxiliary heads but not the R0 primary head, then classifier decoupling is not sufficient. The next literature-supported escalation is explicit multi-view-to-R0 supervision, using the Black & Souvenir / Dong-style principle: construct a training-only fused multi-view target and distill it directly into the R0 feature/logit representation.

## Research direction status

Current preferred order:

1. **AT-SBN** — first, because it is the closest published analogue to the empirical Coffee17 failure and preserves zero inference overhead.
2. **Training-only fused multi-view teacher -> R0** — only if AT-SBN fails to transfer gains.
3. Avoid arbitrary view dropping, post-hoc subset selection, class-wise routing on tiny validation folds, or blind tuning of 0.5/0.5 weights.
