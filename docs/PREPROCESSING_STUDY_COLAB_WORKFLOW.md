# Coffee17 preprocessing study — GitHub-first Colab workflow

Branch: `codex/preprocessing-study-v1`.

This follows the active detection-repository pattern:

- GitHub contains code, configs, and notebooks.
- Each Colab runtime clones the branch directly.
- Google Drive contains persistent evidence, logs, checkpoints, OOF outputs,
  and final analysis.
- Outer test is absent throughout primary training.

Run the four primary notebooks on separate runtimes if desired:
`Coffee17_R0_Seed42_Colab.ipynb`,
`Coffee17_C0_Seed42_Colab.ipynb`,
`Coffee17_F0_Seed42_Colab.ipynb`, and
`Coffee17_W0_Seed42_Colab.ipynb`.

Each arm runs/resumes folds 1–5. After all 20 runs, execute
`Coffee17_Preprocessing_Validation_Decision_Colab.ipynb`; only if it authorizes
OOF, run `Coffee17_Preprocessing_OOF_Colab.ipynb`, then
`Coffee17_Preprocessing_Analysis_Colab.ipynb`.

No source ZIP/bootstrap/code-authority workflow is part of the experiment.
