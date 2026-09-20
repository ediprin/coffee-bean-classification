# Coffee17 Preprocessing Study — Static Contract v1

Status: **draft implementation contract; freeze before primary training**

Primary arms:

- `R0`: raw RGB identity control.
- `C0`: RGB -> LAB, CLAHE on L only, `clipLimit=3.0`, `tileGridSize=8x8`, LAB -> RGB.
- `F0`: canonical AF2, patch 32, overlap 0.50 (stride 16), gamma 0.10, 360 angular bins, hard entropy threshold, inverse FFT, overlap averaging, residual min-max gate.
- `W0`: standalone WAV1, RGB luminance `0.2126R+0.7152G+0.0722B`, two-level orthonormal Haar detail energy, bilinear upsampling, per-level min-max mean cue, then the canonical residual min-max gate.

`F0` and `W0` intentionally retain the canonical residual dynamic range. For input in `[0,1]`, their theoretical output bound is `[0,2]`. Clipped variants are not primary arms.

All frontends are deterministic, parameter-free, preserve BCHW shape/dtype and are external to MobileNetV3. ImageNet normalization is applied after the frontend in Milestone C.

## Software lineage

- C0: `ediprin/coffee-bean-detection`, `agent/af2-clahe-control`, `src/coffee_detector/classical_enhancement/operator.py`
- F0: `ediprin/coffee-bean-detection`, `agent/af2-spectral-factorization`, `src/coffee_detector/afab/operator.py`
- W0: `ediprin/coffee-bean-detection`, `agent/af2-rad-wavelet-refinement`, `src/coffee_detector/af2_spectral/operator.py`, arm `WAV1`

The cross-repo equivalence command is intentionally separate because the three references live on different frozen detection refs. Each reference checkout is compared bitwise on a frozen random probe before primary training.

## Static gate

Training is blocked unless:

- all four arms preserve shape/dtype and produce finite outputs;
- all four arms are deterministic and have zero trainable/persistent state;
- R0 is exact identity;
- C0 stays in `[0,1]` and is active;
- F0/W0 are active, non-decreasing under their residual gate and stay within the canonical `[0,2]` bound for `[0,1]` probes;
- repeated same-seed MobileNetV3 builds have the same state fingerprint and parameter count;
- C0/F0/W0 pass their frozen cross-repo bitwise-equivalence checks.
