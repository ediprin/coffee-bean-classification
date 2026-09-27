# CURRENT RESEARCH CONTEXT — READ THIS FIRST

> **Purpose:** handoff file so a new ChatGPT/Codex session can recover the exact research state without restarting or repeating experiments.
>
> **User directive:** before proposing a new experiment, read this file and inspect the relevant repo branches/artifacts. Do not repeat completed experiments. Do not force a narrative from one paper. Preserve the core constraints below.

## 1. Current thesis direction

Current main task: **fine-grained multi-class classification of green coffee-bean defects on Coffee17 (17 classes)**.

Current working research framing:

**Analisis dan optimasi representasi warna–tekstur pada model ringan untuk klasifikasi fine-grained cacat biji kopi.**

Hard constraints:

- Coffee17 is the main dataset.
- Preprocessing must have a **substantive scientific role**.
- Final deployment should stay **lightweight**.
- Prefer a **single raw-RGB model / single backbone at inference** when possible.
- Evaluate Macro-F1, Hard-F1, Worst-F1, efficiency, and stability.
- XAI can be secondary.
- Do not make the method a pile of unrelated modules.
- Every proposed mechanism must have literature support and must be checked against experiments already completed.

Research principle:

```
problem fit > evidence > novel mechanism > complexity
```

## 2. Interaction / reasoning rules for future sessions

Important user preference:

- Do not repeatedly write rhetorical contrast chains like “bukan X, bukan Y...”.
- Do not reset the research direction when reading one new paper.
- Do not call a method “final” before the evidence supports it.
- Do not propose an experiment until checking whether it already exists in the repo.
- Do not ask the user to rerun analyses already completed.
- Clearly separate: **established evidence**, **interpretation**, and **new hypothesis**.
- Keep answers direct and technically grounded.
- User is especially sensitive to “labil” reasoning: changing the proposed method after every paper.

## 3. Coffee17 / protocol context

Coffee17 is the current public dataset used for fine-grained classification.

Protocol protections already established in repo:

- Original Coffee17 has 979 images.
- Split/identity handling was designed to avoid augmentation leakage.
- Rotation augmentation is applied after split / train-only in the controlled protocol.
- Preprocessing primary study used frozen arms R0/C0/F0/W0.
- OOF analysis contract used identical paired identities and paired stratified bootstrap.
- Once the primary OOF was opened, later variants are exploratory.

Relevant protocol branch:

`codex/preprocessing-study-v1`

Files:

- `docs/protocols/PREPROCESSING_STATIC_CONTRACT.md`
- `docs/protocols/PREPROCESSING_OOF_ANALYSIS_CONTRACT.md`

## 4. Preprocessing arms already studied

Frozen preprocessing definitions:

### R0
Raw RGB identity.

### C0
RGB -> CIELAB -> CLAHE on L channel -> LAB -> RGB.

Parameters:

- clipLimit = 2.0
- tileGridSize = 8x8

### F0
Coffee17 adaptation derived from Xu et al. AFAB-2 / frequency-angular processing:

- Rec.709 luminance
- patch-wise angular-frequency processing
- min-max gate shared across original RGB channels
- residual form similar to `x + x * G_Y`
- deterministic / parameter-free preprocessing frontend

Important: **F0 is an adaptation of one part/principle of Xu’s AFAB, not a reproduction of the complete LFDet AFAB+CGFI system.**

### W0
Channel-wise RGB Haar DWT:

- four decomposition levels
- VisuShrink noise estimate `median(|d|)/0.6745`
- universal threshold
- soft threshold high-frequency coefficients
- inverse DWT reconstruction

## 5. Primary preprocessing results already obtained

Most recently stated standalone Macro-F1 means:

- R0: **86.79**
- C0: **86.10**
- F0: **86.11**
- W0: **87.14**

Interpretation:

- No transformed view decisively replaces raw RGB.
- W0 is the strongest standalone mean among these four in the latest reported result.
- F0 is not privileged merely because it came from Xu.
- The important empirical finding is **complementarity**, not standalone dominance.

Reported prior late-fusion / oracle results from completed analysis:

- R0 + C0 late fusion: ~87.42
- R0 + F0: ~87.17
- R0 + W0: ~88.19
- R0+C0+F0+W0: ~88.17
- Oracle 4-view: ~94.09
- 69 of 126 R0 errors were reportedly rescuable by at least one preprocessing view.

**Verification status:** these late-fusion/oracle numbers were carried from a prior completed chat/artifact and should be re-opened from the exact repo artifact before formal thesis citation. Do not ask the user to rerun them.

Established qualitative result:

**Preprocessing contains complementary information to raw RGB.**

## 6. HBP evidence already established

HBP implementation:

`src/bilinear_lmmd/modeling/models.py` — `HierarchicalBilinearPooling`

Core implementation:

- exactly 3 feature maps
- each stage projected with Conv1x1 + BN + ReLU to 512 channels
- earlier stages adaptive-average-pooled to deepest spatial size
- pairwise products: (0,1), (0,2), (1,2)
- spatial mean
- signed sqrt + L2 normalization per pair
- concatenate 3 x 512 = 1536-D embedding

MobileNetV3-Large HBP commonly uses feature indices `[1,3,4]`.

### Main content-clean HBP vs GAP results (3 seeds)

GAP M0:

- Accuracy: 85.26 ± 1.26
- Balanced Accuracy: 85.39 ± 1.40
- Macro-F1: 85.46 ± 1.31
- Hard-F1: 81.52 ± 2.13
- Worst-F1: 55.64 ± 2.60

HBP M1:

- Accuracy: 86.93 ± 3.14
- Balanced Accuracy: 86.63 ± 3.20
- Macro-F1: 86.78 ± 3.20
- Hard-F1: 84.01 ± 4.58
- Worst-F1: 63.27 ± 3.33

Paired deltas:

- Accuracy +1.68 ± 4.33
- Balanced Accuracy +1.24 ± 4.60
- Macro-F1 +1.32 ± 4.51
- Hard-F1 +2.48 ± 6.55
- Worst-F1 +7.64 ± 5.76

Efficiency:

- GAP: ~2.988M params, ~11.40 MB, batch-1 CUDA ~5.922 ms
- HBP: ~3.562M params, ~13.59 MB, batch-1 CUDA ~7.425 ms

Interpretation:

- HBP helps Coffee17 **on average**.
- Seed variability is high.
- Do not claim universal superiority.

## 7. Fine17 vs Coarse9 granularity experiment

Controlled same-image comparison, labels merged from Fine17 to Coarse9.

Locked test, 3 seeds:

Fine17:

- GAP: ~84.95
- HBP: ~87.97
- gain: +3.03 ± 1.65

Coarse9:

- GAP: ~89.56
- HBP: ~89.98
- gain: +0.42 ± 1.74

Difference-in-difference:

- +2.60 ± 3.33

Paired bootstrap:

- Fine gain: +3.03, CI95 ~[+0.57,+5.69], prop > 0 ~0.992
- Coarse gain: +0.42, CI ~[-2.35,+3.22], prop ~0.607
- Granularity effect: +2.60, CI ~[-0.88,+6.25], prop ~0.931

Correct claim:

**HBP gives a larger point gain on Fine17 than Coarse9 in this experiment, but the interaction is directional rather than conclusively established as a universal causal effect of class granularity.**

Do not simplify to “more classes => HBP better”.

## 8. Major negative / failed experiments — DO NOT REPEAT BLINDLY

Several module-stacking attempts failed or were unstable:

- HBP 320 vs 224: degraded.
- ArcFace GAP: tiny Macro gain, Hard/Worst degraded.
- ArcFace HBP: degraded.
- Spatial HBP 14x14: Worst degraded.
- Global/local MoE: degraded.
- Object crop: strongly degraded.
- EMA HBP: no reliable improvement.
- SPPF-HBP: degraded.
- Capacity residual HBP: degraded.
- Full-family hierarchy: gate failed.
- Compact MPN-COV: unstable Worst-F1 / failed.
- DSConv-only promising at one seed, failed 3-seed confirmation.
- SupCon/confusion-aware direction failed.

General empirical warning:

**Extra modules frequently redistribute errors and destabilize the hard/worst classes.**

## 9. Multi-view / preprocessing transfer experiments already completed

### Shared multi-view training
Branch: `codex/shared-multiview-v1`

Single MobileNetV3 shared across R0/C0/F0/W0 with multi-view CE.

Result:

- initial shared multi-view underperformed matched R0 control.
- F0 training CE nearly collapsed on several folds when auxiliary views shared deployment BN behavior.

### Selective BN
Branch: `codex/shared-multiview-sbn-v1`

Selective BN:

- R0 updates persistent BN running stats.
- transformed views use current batch statistics.
- affine parameters shared.

Result:

- removed severe F0 optimization collapse.
- R0-only inference remained approximately tied with matched R0 control.

### AT-SBN
Branch: `codex/preprocessing-nuisance-analysis-v1`
Protocol: `docs/protocols/AT_SBN_V1.md`

Separate auxiliary classifiers + SBN + auxiliary CE + self-distillation / classifier merging.

Result:

- no meaningful R0-only Macro-F1 gain versus matched control.

### Preprocessing KD
Branch: `codex/preprocessing-kd-v1`

Frozen R0/C0/F0/W0 teachers; raw-only student.

Treatments:

- R0 teacher
- ALL4 probability ensemble

Do not propose generic preprocessing KD again as an untried idea.

## 10. MVFD-SBN — successful aggregate transfer on GAP

Branch:

`codex/mvfd-sbn-v1`

Scientific commit:

`fdae1fff10348f78ca67bb00948ae5768e9c98d9`

Config:

`configs/mvfd_sbn/MVFD_SBN_ALL4.yaml`

Deployment:

```
raw RGB -> MobileNetV3-Large -> GAP -> Linear17
```

Training views:

- R0
- C0
- F0
- W0

Selective BN:

- only R0 updates persistent running stats
- auxiliary views use batch stats
- affine shared

Teacher:

```
t = stopgrad((eC + eF + eW) / 3)
```

Loss:

```
L = CE_R
  + 0.05 * (CE_C + CE_F + CE_W)
  + 0.007 * mean_i ||e_R - t||^2
```

Frozen 5-fold means:

R0 control:

- Accuracy 90.93
- BAcc 90.56
- Macro-F1 90.65
- Hard-F1 87.01
- Worst-F1 64.55

MVFD:

- Accuracy 91.55
- BAcc 92.09
- Macro-F1 91.77
- Hard-F1 85.26
- Worst-F1 67.88

Delta:

- Accuracy +0.619
- BAcc +1.527
- Macro-F1 +1.115
- Hard-F1 -1.750
- Worst-F1 +3.333

Macro improved 4/5 folds; Hard declined.

### AUXCE matched ablation

AUXCE Macro-F1 ~89.94 vs MVFD ~91.77, gain ~+1.826.

Representation:

- AUXCE cosine ~0.95295, L2 ~6.429
- MVFD cosine ~0.98997, L2 ~2.430

Supported claim:

**Explicit final GAP feature transfer contributes to aggregate improvement vs matched AUXCE, while class-level tradeoffs remain.**

### Confidence-aware MVFD

Confidence-weighted teacher aggregation was tested.

Result:

- Macro ~91.38 vs equal-mean MVFD ~91.77.

Equal mean remained preferable.

Do not propose confidence weighting as an untried fix.

## 11. Intermediate multi-level MVFD failed

Direct intermediate spatial MSE + final GAP distillation was tested.

Result:

- Macro-F1 ~86.78
- Hard-F1 ~79.55
- Worst-F1 ~57.85

Relative to MVFD:

- Macro about -4.99
- Hard about -5.71
- Worst about -10.03
- 0/5 folds positive

Scale problem:

- epoch-1 weighted intermediate loss was larger than CE
- direct spatial MSE magnitude was badly mismatched

Correct conclusion:

**Direct spatial MSE distillation at intermediate feature maps with weights copied from final-feature MVFD does not work under this protocol.**

Do not generalize to “all intermediate guidance is invalid”; the failure is confounded by loss scaling/formulation.

## 12. HBP + MVFD-SBN V1 failed

Branch:

`codex/hbp-mvfd-sbn-v1`

Scientific commit:

`e2e57da7bff5efd7f972c3d87ee4c111c99b51a7`

Later branch commit:

`b701fc...`

Protocol:

`docs/protocols/HBP_MVFD_SBN_V1.md`

Config:

`configs/hbp_mvfd_sbn/HBP_MVFD_SBN_ALL4.yaml`

Matched H0 = HBP R0-only vs HM = HBP + MVFD-SBN.

Results:

- GAP frozen reference Macro ~90.65, Hard ~87.01, Worst ~64.55
- HBP R0 Macro ~91.58, Hard ~86.23, Worst ~65.26
- HBP+MVFD Macro ~90.61, Hard ~84.38, Worst ~67.00

HM - H0:

- Macro ~-0.97 pp
- Hard ~-1.86 pp
- Worst ~+1.74 pp

Fold gates:

- Macro positive 1/5
- Hard positive 1/5
- screening failed

Mechanistic diagnostics:

- HBP embedding is concat of 3 individually L2-normalized bilinear blocks, norm ~sqrt(3)
- teacher/raw cosine already ~0.989
- feature KD with lambda 0.007 became tiny
- auxiliary CE dominated optimization
- weighted feature KD was roughly negligible relative to aux CE / primary CE

Class-level harm included:

- Partial Sour
- Cut
- Withered
- Full Sour
- Partial Black

Across 485 unique validation images:

- both correct 434
- MVFD harmed 10 previously-correct samples
- rescued 6 previously-wrong samples

Correct claim:

**The GAP MVFD formulation/weighting does not transfer cleanly to the normalized HBP embedding.**

Do not claim HBP is incompatible with all preprocessing transfer or all distillation.

## 13. XAI result already obtained

HBP vs GAP XAI showed:

- aggregate foreground focus did not improve
- rescued samples showed a positive relative confidence-drop pattern
- harmed samples showed the opposite

Supported claim:

**HBP representation can be more discriminative on rescued cases.**

Do not claim HBP simply “looks more at the bean”.

## 14. Key coffee-domain papers and what they actually imply

### Chang & Huang 2021
“Deep Learning Model for the Inspection of Coffee Bean Defects”

Important observation:

- common pooling / dimensionality reduction can lose coffee-bean features
- model modifications were motivated by retaining useful feature information
- padding blur and dead neurons were also analyzed

Use this paper as evidence that **feature preservation matters in coffee defect recognition**.

### Chang & Liu 2024
“Multiscale Defect Extraction Neural Network for Green Coffee Bean Defects Detection”

Important findings:

- coffee defects have subtle visual differences
- different receptive fields capture different information
- 3x3 retained detailed texture
- larger filters captured broader contour behavior
- their 3x3 + 5x5 combination was chosen for defect + edge information
- immature vs withered confusion was attributed to fine lines
- withered vs cut confusion involved wrinkle / incomplete contour similarity

Use this paper as evidence for **multi-scale texture/edge/contour representation**.

Do not copy their entire network.

### Jiao et al. 2025 — Swin-HSSAM
Important design:

- Swin backbone
- features from stages 2, 3, 4
- hierarchical feature fusion
- selective attention before classification

Use as coffee-domain evidence that **multi-stage information can be useful**.

Do not infer that the same fusion must work on Coffee17.

### Hu et al. 2025 — Siamese few-shot
Important point:

- explicitly discusses subtle visual differences among coffee defect categories
- uses grayscale partly because defects are textural
- supports the importance of discriminative texture representations

## 15. Xu et al. 2025 — critical correction

Paper:

**More signals matter to detection: Integrating language knowledge and frequency representations for better fine-grained aircraft recognition**

Do not read AFAB in isolation.

LFDet has three coordinated components:

1. **AFAB** — data-space frequency augmentation
2. **CGFI** — feature-space content-aware global frequency enhancement
3. **FTIF** — text-image semantic supplementation

### AFAB

AFAB includes:

- patch-wise DFT
- patch-specific adaptive high-pass filtering
- patch-specific chaotic amplitude suppression
- iDFT recovery
- gating / residual fusion with the raw spatial image

Patch-wise DFT is important because Xu argues global DFT ignores local details needed for fine-grained recognition.

AFAB-1:

- adaptive high-pass radius based on patch energy

AFAB-2:

- angular density distribution over Fourier amplitude
- entropy-derived adaptive threshold
- suppress low-density directions
- preserve original phase
- reconstruct spatial image

### Raw + recovered gating

Xu does not simply replace the raw image with a frequency-transformed image.

Recovered frequency space is normalized and used to modulate raw spatial information, then residual fusion produces the enhanced image.

### CGFI — the direct “partner” after AFAB

Xu explicitly states that even with a high-quality AFAB data space, **feature encoding can still lose critical information**, especially during feature fusion / channel reduction.

CGFI then performs content-aware frequency filtering on high-dimensional feature maps:

```
X_f^{l*} =
F^{-1}[ T(F(X_f^l)) ⊙ F(X_f^l) ]
```

Thus Xu’s conceptual pipeline is:

```
data-space signal enhancement
        ->
feature-space signal preservation/enhancement
        ->
fine-grained classifier
```

### Xu ablation warning

AFAB components in isolation did not always produce the full benefit.

Xu reports that:

- AFAB-1 alone and AFAB-2 alone can help
- combining them without the other LFDet components can produce an unexpected drop relative to individual subcomponents
- AFAB becomes more useful when paired with feature-space / classifier-side components
- the paper attributes this to the network needing mechanisms that can exploit the high-quality data space

This is highly relevant to interpreting F0.

## 16. Important interpretation of F0 vs Xu

Our F0 should not be treated as “Xu’s complete method”.

Our F0 is a **deterministic Coffee17 preprocessing adaptation inspired mainly by AFAB-2 / luminance frequency-angular gating**.

Therefore:

- mediocre F0 standalone performance does not prove Xu’s overall data-space + feature-space concept is invalid
- it also does not justify copying AFAB+CGFI wholesale

Correct question:

**What useful preprocessing-derived information exists in Coffee17, and how can a lightweight raw-only model absorb it without doubling inference cost?**

## 17. Current efficiency constraint

A proposed dual-view inference such as:

```
R0 -> model branch A
W0/F0 -> model branch B
fusion
```

would require multiple forward passes and therefore sacrifices the efficiency objective.

This was proposed briefly and then rejected.

Current deployment preference:

```
training:
raw + preprocessing-derived auxiliary signal

inference:
raw RGB -> one lightweight model -> prediction
```

Preprocessing can be heavier during training if the final deployed model remains light, but preprocessing/inference cost must be explicitly measured if preprocessing remains active at test time.

## 18. Current unresolved research question

The strongest coherent question at this moment is:

> **How can complementary information exposed by deterministic preprocessing be transferred into a lightweight fine-grained Coffee17 representation while retaining single-model raw-RGB inference?**

This is the core research problem.

Do not prematurely equate this with one particular module.

## 19. Current candidate interpretation: F0 and W0 may serve different roles

This is a **hypothesis / working interpretation**, not yet a conclusion.

### W0
Empirically strongest standalone preprocessing mean among R0/C0/F0/W0.

Potential role:

- strong alternate image representation / benchmark of preprocessing usefulness

### F0
Its mathematical structure is closer to a **saliency / structure guidance signal**:

- local frequency
- angular edge/texture orientation
- residual gate on raw signal

Potential role:

- training-time guidance rather than replacement image

This role separation is plausible but **not yet proven**.

## 20. Latest proposed idea — NOT YET APPROVED

A possible lightweight route discussed immediately before this file was written:

- raw RGB goes through the ordinary lightweight backbone
- F0-derived information produces a training-time guidance map
- one intermediate raw feature stage receives an auxiliary guidance loss
- inference discards F0/guidance and uses the raw model only

Abstractly:

```
x -> backbone -> classifier

training only:
x -> F0 -> guidance map G

L = L_CE + lambda * L_guidance(F_k, G)
```

Why it came up:

- Xu: data-space enhancement can be lost during encoding
- Chang: coffee defects depend on subtle local lines/texture/contour
- HBP: Coffee17 benefits on average from multi-stage interactions
- MVFD final-vector transfer can improve GAP but HBP transfer failed
- intermediate full-tensor MSE failed badly

**Important:** this specific “F0 guidance at middle stage” idea was proposed too quickly. The user challenged whether it was another unstable jump. It is **not frozen**, and must not be implemented until literature + repo duplication checks are completed.

## 21. What to do next — exact procedure

Before any new training:

1. **Audit repo for materially similar experiments**
   - terms: guidance, attention transfer, spatial guidance, feature modulation, preprocessing guidance, F0 feature, W0 feature, frequency guidance, wavelet guidance, intermediate transfer.
   - Ensure the idea has not already been tested under another name.

2. **Audit literature for the mechanism, not just the topic**
   - preprocessing-derived / transform-derived guidance
   - attention transfer
   - frequency-guided feature learning
   - wavelet-guided feature learning
   - training-only auxiliary guidance with raw-only inference
   - lightweight fine-grained representation preservation

3. **Compare candidate guidance source(s) using existing Coffee17 evidence**
   - F0: frequency-angular local structure
   - W0: wavelet denoised multi-resolution representation
   - possibly C0 only if class-level evidence supports luminance contrast

4. **Use existing predictions to inspect class-level rescue/harm**
   - especially known hard/confused Coffee17 defect groups
   - determine whether F0/W0 rescue the same classes where HBP helps or harms
   - do not rerun generic complementarity/oracle analysis if the artifacts already contain it

5. Only then freeze **one minimal falsifiable experiment**.

## 22. Candidate experiment criteria if guidance survives audit

The first experiment should be minimal:

- same Coffee17 folds
- same backbone
- same training recipe
- one guidance source
- one feature stage
- one small auxiliary loss
- raw-only inference
- no second backbone
- no ensemble
- no extra classifier unless required by the cited method

Primary screening metrics:

- Macro-F1
- Hard-F1
- Worst-F1

Efficiency:

- params
- FLOPs
- latency
- peak memory if available

A candidate should not be promoted merely for +Macro if it materially damages Hard/Worst without a defensible tradeoff.

## 23. Claims that are currently supported

Supported:

- Coffee17 is a fine-grained 17-class classification problem with subtle visual confusion.
- deterministic preprocessing views contain complementary information to raw RGB.
- W0 has the strongest latest-reported standalone preprocessing Macro-F1 among the four frozen arms.
- HBP improves average Coffee17 performance with high seed variability.
- HBP’s larger Fine17 point gain than Coarse9 is directional evidence, not proof that class count causes HBP benefit.
- MVFD-SBN can transfer multi-view information into raw-only GAP inference and improve aggregate Macro-F1.
- MVFD-SBN harms Hard-F1 under the frozen GAP setup.
- direct intermediate spatial MSE transfer failed badly.
- GAP MVFD weighting transferred to normalized HBP embedding failed screening.
- Xu’s AFAB is intended to work in a larger data-space -> feature-space information-preservation framework.
- Chang provides coffee-specific evidence that fine lines, texture, wrinkle and contour matter and can be confused.

## 24. Claims that remain hypotheses

Do not state as established:

- F0 is the best preprocessing for the thesis.
- W0 is the final preprocessing.
- F0 is the best guidance signal.
- middle-stage guidance is the correct location.
- preprocessing guidance will improve HBP.
- frequency guidance is novel by itself.
- wavelet guidance is novel by itself.
- AFAB+CGFI should simply be copied to Coffee17.
- more classes cause HBP to work better.
- HBP is incompatible with KD.

## 25. Useful repo branches / commits

Known branches:

- `codex/preprocessing-study-v1`
- `codex/preprocessing-kd-v1`
- `codex/shared-multiview-v1`
- `codex/shared-multiview-sbn-v1`
- `codex/preprocessing-nuisance-analysis-v1`
- `codex/mvfd-sbn-v1`
- `codex/hbp-mvfd-sbn-v1`
- `agent/hbp-backbone-compatibility`
- `agent/hong-classification-ablation`

Known scientific commits:

- MVFD-SBN: `fdae1fff10348f78ca67bb00948ae5768e9c98d9`
- HBP-MVFD-SBN V1: `e2e57da7bff5efd7f972c3d87ee4c111c99b51a7`
- F0 frozen luminance reference mentioned in protocol: `6ef389c23932e44fe4135c32d471b3008b1cbf39`

## 26. Key source papers already uploaded / available in project context

Coffee-domain PDFs already available include:

- Arwatchananukul et al. 2024 — Coffee17
- Chang & Huang 2021 — Deep Learning Model for the Inspection of Coffee Bean Defects
- Chang & Liu 2024 — Multiscale Defect Extraction Neural Network for Green Coffee Bean Defects Detection
- Jiao et al. 2025 — Swin-HSSAM
- Hu et al. 2025 — Siamese networks for few-shot coffee bean defect detection
- Hsia et al. 2022 — Explainable and lightweight CNN for green coffee quality detection
- Liang et al. 2023
- Kesiman et al. 2023
- Febriana et al. 2022 — USK-Coffee
- Wang et al. 2021
- Gope et al.
- Vachmanus et al. 2026
- Sandhya et al. 2026
- Muchtar et al. 2025
- Ke et al. 2025
- Hashmi et al. 2025
- Motta et al. 2024
- Hassan 2024
- Manansala & Paglinawan 2024
- others in the project file set

Xu et al. 2025 (“More signals matter to detection...”) is available in the user Library / prior uploaded materials and should be read in full when making claims about AFAB/CGFI.

## 27. Immediate handoff state

At the moment this context file was created, the conversation had reached this exact point:

- user objected to unstable method switching
- assistant acknowledged that merely copying AFAB+CGFI would be overcorrection
- latest working principle is **preprocessing-derived information transfer into a lightweight raw-only model**
- F0-vs-W0 role distinction is under investigation
- “F0 guidance at middle stage” is only a candidate hypothesis
- **next action must be evidence audit, not immediate training**

If a new chat starts, continue from here instead of restarting the preprocessing/MVFD/HBP discussion.
