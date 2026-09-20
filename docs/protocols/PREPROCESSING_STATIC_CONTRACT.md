# Coffee17 Preprocessing Study — Static Contract v2

Status: **literature-audited candidate contract; no primary training before observability PASS**

Primary arms:

- `R0`: raw RGB identity control.
- `C0`: RGB -> CIELAB, CLAHE on L only, `clipLimit=2.0`, `tileGridSize=8x8`, LAB -> RGB.
- `F0`: patch-wise angular-frequency processing derived from AFAB-2, computed once on Rec.709 luminance `Y=0.2126R+0.7152G+0.0722B`; the resulting min-max gate is shared across the untouched RGB channels using `x + x * G_Y`. No post-hoc clipping.
- `W0`: channel-wise RGB Haar DWT, four decomposition levels, VisuShrink noise estimate `median(|d|)/0.6745`, universal threshold `T=sigma*sqrt(2 log n)`, soft threshold on high-frequency coefficients, inverse DWT reconstruction. No residual detail-energy gate.

Scientific lineage:

- C0 follows the LAB-L CLAHE preprocessing described by Mohanty et al. (2026), including clip limit 2.0 and 8x8 tiles.
- F0 uses Xu et al. (2025) as the frequency/angular basis and the color-texture separation argument of Maenpaa & Pietikainen (2004). Its exact shared-luminance transfer is additionally checked against the frozen coffee-detection implementation at commit `6ef389c23932e44fe4135c32d471b3008b1cbf39`.
- W0 follows the denoising equations described by Yang et al. (2025): RGB channels processed separately, Haar, VisuShrink, soft thresholding, inverse reconstruction. The paper supports the operator family; its WaveLiteNet effect size is not treated as the standalone W0 effect.

All frontends are deterministic and parameter-free. ImageNet normalization is applied only after the preprocessing frontend.

## Pre-training gates

Primary training is blocked unless:

- the dataset provenance gate covers the 979 original Coffee17 images;
- R0/C0/F0/W0 preserve BCHW shape/dtype and produce finite outputs;
- R0 is exact identity;
- C0 is active and remains in [0,1];
- F0 is active, remains within its residual theoretical [0,2] bound for [0,1] inputs, and preserves RGB chromaticity under the shared gate;
- W0 is active and reconstructs a finite RGB image;
- F0 passes bitwise equivalence against the frozen luminance reference implementation;
- the full 979-original observability audit is complete and records pixel range, distribution shift, fraction outside [0,1], SSIM, chromaticity shift, and per-class effects;
- repeated same-seed MobileNetV3 builds have the same initial state fingerprint and parameter count.

No preprocessing parameter may be changed after viewing primary validation/OOF results.
