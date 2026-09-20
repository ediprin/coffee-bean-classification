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
