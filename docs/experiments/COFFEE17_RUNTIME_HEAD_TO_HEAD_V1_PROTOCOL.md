# Coffee17 Runtime Head-to-Head V1 — registered protocol

Status: IMPLEMENTED / AWAITING CUDA EXECUTION. No measured result yet.

Purpose: compare the retained efficient candidate C2 with historical B0, W3, B1 and W4 using the exact existing model builders and one common runtime. No architecture or accuracy retraining; no Coffee17 dataset, checkpoint or outer-test access.

## Frozen configuration

- Runner: `src/bilinear_lmmd/experiments/profile_runtime_head_to_head_v1.py`
- Scientific code pin: `b659a22a72c97bfe75576424a38cbb10076ab0be`
- Notebook: `notebooks/Coffee17_Runtime_Head_To_Head_V1_Kaggle.ipynb`
- Config: `configs/frequency_screening/FOCUSED_EFFICIENCY_V1.yaml`, overriding pretrained=false only for profiling.
- Fixed seed 42 random weights, one identical synthetic FP32 tensor (1,3,224,224), eager eval/inference mode, CUDA device 0; TF32 and cuDNN benchmark off. No autocast/compilation/export. All models must return finite 17 logits and match historical parameter counts.
- B0 2,988,289; W3 2,992,169; C2 2,993,089; B1 3,562,305; W4 3,566,185.
- 30 initial warmups per model, 10 block warmups, 10 rounds with 50 observations per model each (500 per model). Seeded cyclic order and reverse cyclic order put each arm in each ordinal position exactly twice. All five models resident during latency and throughput measurement. Latency is synchronized host wall-clock model forward, excluding input transfer, preprocessing, softmax and I/O.
- Throughput measured in separate 50-forward blocks without synchronization between individual forwards, with synchronization at both block boundaries. Report total images / total elapsed time and dispersion across rounds.
- Memory measured separately with one model resident, same synthetic input, cache cleared between arms, reset CUDA allocated peak after warmup. Report empty, resident, peak, incremental forward and model+input+forward bytes. Not process-wide reserved memory.
- Record pooled median/p95 (nearest rank), round median dispersion, paired round percentage differences against appropriate anchors, raw timing observations, environment/version/device and config hash. Conv2d+Linear MAC counter is partial; it excludes wavelet/pooling/BN/elementwise operations. State tensor bytes are not a serialized file size.

## Analysis boundary

Compare C2 with B0 (incremental cost), W3 (previous lightweight incumbent), B1 and W4 (higher parameter comparators). Retain historical development accuracy from canonical reports; do not claim it was measured in this notebook. Interpret observed latency differences with round-to-round variation; do not equate small differences with statistical or practical superiority. This is architecture-only random-weight timing on one CUDA environment, not proof of trained-checkpoint, CPU, edge-device, quantized or end-to-end deployment latency. No independent accuracy confirmation. No new training or architecture search authorized by this protocol.

Output: `/kaggle/working/coffee17-runtime-head-to-head-v1.json`. Preserve previous output; no checkpoint deletion, no dataset mounts required. Notebook uses isolated temporary checkout. Target CUDA execution still needs to be run by user; authoring environment lacks CUDA/timm and direct git network access, so actual GPU measurements and runtime test pass are not yet verified. Notebook checks helper ordering and finite forward outputs before timing.
