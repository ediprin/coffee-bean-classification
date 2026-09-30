# WR-HBP Final Outer-Test Confirmation V1

Status: **registered before outer-test access; checkpoint-loss recovery amendment registered before outer-test access**

Purpose: perform the one-shot final confirmation of the only Coffee17 method
that passed the frozen development gate:

- control: R0-HBP
- candidate: WR-HBP

No new method search, tuning, checkpoint selection, or retraining is authorized
after the outer test is materialized.

## 1. Development evidence required before authorization

The final test may be materialized only if all five completed development folds
from `coffee17-wavelet-residual-hbp-v1` are available together with their exact
selected checkpoints.

Operational packaging is not part of the scientific estimand. The exact frozen
development artifacts may be mounted either as the saved
`coffee17-wavelet-residual-hbp-project/` directory or as the dedicated
`wavelet-residual-hbp-final-confirmation-bundle.zip`. In both cases the same
checkpoint hashes and run contracts below must pass before any outer-test
identity is materialized. The compact analysis-package ZIP is insufficient
because it intentionally omits `best.pt` and `last.pt`.

For every fold, the authority builder must verify:

- protocol = `coffee17-wavelet-residual-hbp-v1`;
- seed = 42;
- matched shared-core initialization;
- matched validation rows;
- R0-HBP was retrained as the matched control;
- WR-HBP was trained as the candidate;
- `outer_test_accessed = false`;
- both `best.pt` files exist;
- each actual checkpoint SHA-256 equals the SHA-256 recorded in
  `pair_result.json`.

The development gate is recomputed from those five pair results and must still
be PASS:

1. mean development Macro-F1 delta > 0;
2. Macro-F1 positive in at least 3/5 folds;
3. mean development Hard-F1 delta >= 0;
4. mean development Worst-F1 delta >= 0.

Only then is an authority file emitted with:

`decision = AUTHORIZE_OOF_TEST_EVALUATION`.

This authority explicitly sets
`further_primary_tuning_authorized = false`.

### 1.1 Checkpoint-loss recovery amendment

The original selected development checkpoints are preferred and remain the
primary handoff. If, and only if, those exact files are no longer recoverable
before any outer-test image has been materialized, one recovery run is
authorized under the following frozen constraints:

- the method remains exactly R0-HBP versus WR-HBP V1;
- the same Coffee17 clean population, five development folds, seed 42, 224x224
  input, optimizer, 50 epochs, checkpoint selection rule, HBP stages [1,3,4],
  projection dimension 512, and WR-HBP wavelet branch are retained;
- no architecture, preprocessing, loss, hyperparameter, hard-group definition,
  screening gate, or outer-test rule may be changed;
- recovery training must use strict deterministic CUDA execution with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms enabled,
  cuDNN benchmark disabled, cuDNN deterministic enabled, and TF32 disabled;
- the recovery is a single preregistered reconstruction attempt, not a new
  method-search round;
- the same four-part development gate must PASS before any outer-test identity
  is materialized;
- if the recovery gate fails, the outer test remains unopened and WR-HBP is not
  advanced to final confirmation under this protocol.

Recovery checkpoints are **not described as the lost original checkpoints**.
The final authority must record
`development_mode = checkpoint_loss_recovery_v1` and the exact recovery code
commit. The audited recovery runtime is pinned to
`9004462d4c114e594aced9954a56435231c84ac3`. It preserves the frozen WR-HBP
V1 architecture/configuration, uses strict deterministic recovery, and adds a
fail-fast check that the shared RGB-HBP initialization SHA-256 equals the
original WR-HBP V1 fingerprint
`6d425c6c149005afde1224ed74ccfe4eb84c95e574d21e25a0e728578d79f9cc`.
The authority independently verifies the same fingerprint in every fold and
arm contract. This exact recovery-runtime commit passed repository CI before
being registered here. Notebook-source inspection tests are validated by branch
CI and are excluded from the pinned Kaggle runtime preflight; Kaggle preflight
tests only the scientific/runtime code. This amendment exists solely because
checkpoint files were lost while the outer test remained untouched.

## 2. Outer-test structure

The existing Coffee17 fold manifest is preserved exactly.

Each fold has a locked `test` subset that has never been materialized during
method development. The five test subsets are disjoint and their union equals
the clean Coffee17 population.

For fold k:

- load the already-selected R0-HBP checkpoint from development fold k;
- load the already-selected WR-HBP checkpoint from development fold k;
- materialize only fold k's locked outer-test identities;
- evaluate both checkpoints on exactly the same identities.

No training occurs in final confirmation.

No checkpoint is selected by outer-test performance.

## 3. Frozen inference models

R0-HBP:

`RGB -> MobileNetV3-Large -> HBP -> Linear17`

WR-HBP:

`RGB -> MobileNetV3-Large + L1 luminance Haar/VisuShrink residual -> HBP -> Linear17`

The WR-HBP architecture is exactly the frozen V1 development architecture:

- MobileNetV3-Large;
- HBP stages [1,3,4];
- projection dimension 512;
- linear 17-class classifier;
- dropout 0.2;
- luminance-only L1 Haar detail;
- bands LH/HL/HH;
- VisuShrink soft threshold;
- 16-channel residual projection;
- shallow residual injection;
- learned tanh gate from the selected development checkpoint.

## 4. Metrics

Per outer fold and pooled out-of-fold test predictions:

- Accuracy;
- Balanced Accuracy;
- Macro-F1;
- Hard-F1;
- Worst-F1;
- six preregistered hard-pair confusion counts;
- paired rescue/damage.

Hard-F1 uses the same development-era hard set recorded for WR-HBP:

- sour/black: Partial Black, Partial Sour, Full Sour;
- shape/withered: Withered, Immature, Cut;
- insect damage: Slight Insect Damage, Severe Insect Damage.

Separately, explanatory confusion counts are frozen for six difficult pairs:

1. Withered ↔ Immature
2. Severe Insect Damage ↔ Slight Insect Damage
3. Cut ↔ Slight Insect Damage
4. Partial Sour ↔ Full Sour
5. Slight Insect Damage ↔ Fade
6. Full Black ↔ Partial Black

Those pair counts do not redefine Hard-F1.

The pooled prediction set must contain each clean Coffee17 identity exactly
once per arm.

## 5. Frozen final confirmation gate

WR-HBP is `CONFIRMED_WR_HBP` only if **all** hold:

1. pooled Macro-F1 delta (WR-HBP - R0-HBP) > 0;
2. per-fold Macro-F1 delta > 0 in at least 3/5 outer folds;
3. pooled Hard-F1 delta >= 0;
4. pooled Worst-F1 delta >= 0.

Otherwise:

`WR_HBP_NOT_CONFIRMED`.

A paired class-stratified bootstrap with 10,000 replicates reports a 95% confidence interval for pooled
Macro-F1 delta as uncertainty information. It is **not** an
additional pass/fail criterion because the decision gate was deliberately kept
parallel to the development gate.

## 6. One-shot boundary

After any outer-test prediction is produced:

- no method changes;
- no wavelet-branch tuning;
- no HBP changes;
- no loss changes;
- no feature fusion;
- no hyperparameter rescue;
- no checkpoint reselection;
- no retraining based on test results.

If WR-HBP is confirmed, it is the final proposed model.

If WR-HBP is not confirmed, the final thesis result reports the negative
confirmation honestly; HBP remains the reference model and WR-HBP is described
as a development-stage improvement that did not independently confirm.

## 7. Efficiency reporting

Parameter counts are inherited from the frozen development record:

- R0-HBP: 3,562,305 trainable parameters;
- WR-HBP: 3,563,202 trainable parameters;
- overhead: 897 parameters (0.0252%).

End-to-end latency should be measured separately if an efficiency claim beyond
parameter count is made.
