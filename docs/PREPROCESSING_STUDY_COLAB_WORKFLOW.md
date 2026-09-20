# Coffee17 preprocessing study — GitHub-first Colab workflow

Branch: `codex/preprocessing-study-v1`.

GitHub contains code/configs/notebooks. Google Drive contains persistent provenance, runtime, observability, checkpoints, OOF outputs and analysis. The outer test is absent throughout primary training.

Before any training, the first arm runtime performs/reuses three common gates under one Drive lock:

1. Coffee17 provenance + deterministic 5-fold manifests.
2. Static frontend/model preflight.
3. A **979-original preprocessing observability audit** for R0/C0/F0/W0. This audit performs no training and no model inference.

Primary arms are:

- R0: raw RGB.
- C0: LAB-L CLAHE, clip 2.0, 8x8.
- F0: Rec.709 luminance AFAB-2/angular-frequency shared gate.
- W0: four-level Haar + VisuShrink soft-threshold + inverse reconstruction.

Run the four primary notebooks:
`Coffee17_R0_Seed42_Colab.ipynb`,
`Coffee17_C0_Seed42_Colab.ipynb`,
`Coffee17_F0_Seed42_Colab.ipynb`, and
`Coffee17_W0_Seed42_Colab.ipynb`.

Each arm runs/resumes folds 1–5. After all 20 primary runs, execute
`Coffee17_Preprocessing_Validation_Decision_Colab.ipynb`. Only if it authorizes OOF may `Coffee17_Preprocessing_OOF_Colab.ipynb` be run, followed by `Coffee17_Preprocessing_Analysis_Colab.ipynb`.

No source ZIP/bootstrap/code-authority workflow is part of the experiment.
