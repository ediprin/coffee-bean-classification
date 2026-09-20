# Coffee17 Preprocessing Study — Dataset Contract v1

Status: **draft implementation contract; freeze before primary training**

The lineage is:

`Kaggle ZIP -> raw provenance audit -> canonical class-only originals -> exact duplicate/conflict audit -> frozen clean population -> deterministic 5-fold assignments -> train+validation development materialization -> primary training -> explicitly authorized outer-test materialization`.

No random 70/20/10 split is created before cleaning.

The production provenance gate verifies the Coffee17 class counts already encoded in `prepare_coffee17.py` (979 images, 17 canonical classes), records both archive SHA-256 and a repack-stable content SHA-256, and asserts `split_created=false`, `model_accessed=false`, and `training_executed=false`.

Cleaning follows the existing repository policy: keep one lexicographic canonical identity for exact same-class duplicates; quarantine every identity in an exact cross-class hash group.

Five outer folds are frozen with seed 42. Train/validation/test are disjoint within each fold, every clean class must appear in all splits, and each clean identity must appear in outer test exactly once across the five folds.

`prepare_preprocessing_folds.py` writes manifests only. `materialize_preprocessing_development.py` may create only `source/train/` and `source/val/`. `source/test/` is forbidden during primary training.

Outer-test materialization requires both an explicit `--authorize-test` flag and an authority JSON whose decision is `AUTHORIZE_OOF_TEST_EVALUATION` and which closes further primary tuning.
