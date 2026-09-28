# Coffee17 preprocessing study — Kaggle-only workflow

Frozen training-code commit: `7de2abb46efd5e71dbab508e2ae4b4e61102a7aa`.

This workflow uses only Kaggle runtime/storage:
- Coffee17 is mounted under `/kaggle/input`.
- Working evidence/checkpoints live under `/kaggle/working/coffee17-preprocessing-project`.
- No Google Drive and no Hugging Face are used.
- After each stage, use **Save Version** so the project directory becomes Kaggle Notebook Output.
- In the next stage, add the previous notebook output through **Add Input → Your Work / Notebook Output**.

## Execution order

1. `Coffee17_R0_Seed42_Kaggle.ipynb`
   - Inputs: Coffee17 dataset only.
   - First run: execute setup/audit + audit-summary cells, inspect observability, then run training.
   - Save Version.

2. `Coffee17_C0_Seed42_Kaggle.ipynb`
   - Inputs: Coffee17 dataset + R0 notebook output.
   - The notebook merges prior evidence/results exactly and reuses the frozen runtime/audit.
   - Save Version.

3. `Coffee17_F0_Seed42_Kaggle.ipynb`
   - Inputs: Coffee17 dataset + C0 notebook output.
   - Adds the frozen F0 detection-reference equivalence check.
   - Save Version.

4. `Coffee17_W0_Seed42_Kaggle.ipynb`
   - Inputs: Coffee17 dataset + F0 notebook output.
   - Its output now contains R0+C0+F0+W0 (20 primary runs).
   - Save Version.

5. `Coffee17_Preprocessing_Validation_Decision_Kaggle.ipynb`
   - Input: W0 notebook output.
   - No training, no outer-test access.
   - Save Version only if decision is `AUTHORIZE_OOF_TEST_EVALUATION`.

6. `Coffee17_Preprocessing_OOF_Kaggle.ipynb`
   - Inputs: Coffee17 dataset + Validation Decision output.
   - One-time outer-test inference only.
   - Save Version.

7. `Coffee17_Preprocessing_Analysis_Kaggle.ipynb`
   - Inputs: Coffee17 dataset + OOF output.
   - Bootstrap, per-class/confusion analysis, efficiency, final report.
   - No training.

## Kaggle settings

For arm training and OOF/efficiency, enable a GPU accelerator and Internet (GitHub clone + package installation). The Coffee17 images themselves are read from the mounted Kaggle Input; the notebook does not download the dataset from another service.

The first R0 run freezes exact package versions into the project. Later notebooks install/reuse that frozen lock and verify the software fingerprint before training/evaluation. If Kaggle's environment is incompatible, the notebook fails before producing a mixed-environment primary result.

## All-in-one option

If you want one notebook and one Run All instead of the staged notebooks, use:

`notebooks/Coffee17_Preprocessing_All_Kaggle.ipynb`

It automatically runs the complete frozen pipeline:

`provenance -> fold gate -> observability -> F0 equivalence -> 20 primary runs -> primary confirmation -> OOF -> bootstrap/per-class analysis -> efficiency -> final report`.

It aborts before training if the static/observability gates do not pass. It does not tune or alter the treatment based on validation results. After the exact 20 primary runs are complete, the confirmation step must emit `AUTHORIZE_OOF_TEST_EVALUATION` before the notebook opens the outer test for one-time OOF inference.

The notebook can also reuse an earlier saved Kaggle Notebook Output of itself for exact-contract resume.
