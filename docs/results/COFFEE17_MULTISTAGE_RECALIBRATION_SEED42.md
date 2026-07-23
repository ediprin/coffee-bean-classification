# Coffee17 multistage recalibration: seed-42 validation result

## Status

**SCREENING COMPLETE — ADAPTIVE STAGE SELECTION FAILED THE CAPACITY CONTROL.**

- Dataset: Coffee17 clean grouped fold 1.
- Selection split: validation, 97 images.
- Training seed: 42.
- Test opened: no.
- Seeds 123/2026 opened: no.
- Canonical raw artifact:
  `ediprin/coffee-backbone-checkpoints`,
  namespace `coffee17-multistage-recalibration-v1`,
  file `val_reports/multistage_recalibration_capacity_control.json`.

This document records the exact console result supplied after the persistent
Colab run. It is screening evidence, not a multi-seed or test claim.

## Models

| Code | Description | Parameters |
|---|---|---:|
| BE2G | EfficientNetV2-B0 + final-stage GAP | existing baseline |
| BE2H | EfficientNetV2-B0 + HBP | existing comparator |
| MSF0 | Uniform fusion of three aligned stages | 5,656,353 |
| MSF1 | Per-image, per-channel softmax selection across stages | 5,730,561 |
| MSFC | Per-image channel recalibration, uniform across stages | 5,730,561 |

MSFC and MSF1 are capacity matched. MSFC cannot select one stage over another.

## Initial multistage screen

| Comparison | Delta Macro | Delta Hard | Delta bottom-three | Delta Worst |
|---|---:|---:|---:|---:|
| BE2G -> MSF0 | +2.68 | +2.61 | +0.74 | +0.00 |
| BE2H -> MSF0 | +0.60 | +0.33 | -3.70 | -10.26 |
| BE2G -> MSF1 | +3.66 | +3.57 | +5.69 | +6.06 |
| BE2H -> MSF1 | +1.59 | +1.28 | +1.25 | -4.20 |
| MSF0 -> MSF1 | +0.98 | +0.95 | +4.95 | +6.06 |

The screen passed and authorized only the capacity control.

## Capacity-matched result

| Comparison | Macro-F1 | Hard-F1 | Bottom-three F1 | Worst-F1 |
|---|---:|---:|---:|---:|
| BE2G -> MSFC | 88.71 -> 92.37 (+3.65) | 82.20 -> 86.89 (+4.68) | 73.79 -> 78.57 (+4.78) | 66.67 -> 66.67 (+0.00) |
| BE2H -> MSFC | 90.79 -> 92.37 (+1.58) | 84.49 -> 86.89 (+2.40) | 78.23 -> 78.57 (+0.34) | 76.92 -> 66.67 (-10.26) |
| MSF0 -> MSFC | 91.39 -> 92.37 (+0.97) | 84.82 -> 86.89 (+2.07) | 74.53 -> 78.57 (+4.04) | 66.67 -> 66.67 (+0.00) |
| MSFC -> MSF1 | 92.37 -> 92.38 (+0.01) | 86.89 -> 85.77 (-1.12) | 78.57 -> 79.48 (+0.91) | 66.67 -> 72.73 (+6.06) |

Frozen capacity gate for `MSFC -> MSF1` required:

1. Macro-F1 improvement;
2. Hard-F1 improvement;
3. bottom-three F1 decline no greater than one point.

MSF1 failed criterion 2. The final capacity decision is **FAIL**.

## Interpretation boundary

The permitted interpretation is:

> On Coffee17 validation seed 42, direct multistage fusion was useful and
> per-image channel recalibration improved its Macro-F1 and Hard-F1. Adaptive
> selection among stages did not outperform a capacity-matched uniform-stage
> channel control.

The result does **not** establish MSFC as the final model. MSFC had the best
Macro-F1 and Hard-F1 in this group, but lost 10.26 points of Worst-F1 against
BE2H. The validation set is small and per-class supports are discrete. A
post-hoc per-class/rescue-harm audit and paired stratified bootstrap are
allowed because they require no training and do not inspect test.

No further seed or test evaluation is authorized by protocol v1.

## Post-hoc per-class audit: MSFC versus MSF1

**Status: DIAGNOSTIC ONLY.** Audit membaca prediction validation yang sudah
ada; tidak melakukan training dan tidak membuka test.

Dalam perbandingan ini MSFC adalah baseline dan MSF1 adalah candidate.

| Class | n | MSFC F1 | MSF1 F1 | Delta | Rescued by MSF1 | Harmed by MSF1 |
|---|---:|---:|---:|---:|---:|---:|
| Partial Sour | 5 | 90.91 | 80.00 | -10.91 | 0 | 1 |
| Cut | 7 | 93.33 | 87.50 | -5.83 | 0 | 0 |
| Full Sour | 8 | 93.33 | 87.50 | -5.83 | 0 | 0 |
| Fade | 3 | 100.00 | 100.00 | +0.00 | 0 | 0 |
| Full Black | 4 | 100.00 | 100.00 | +0.00 | 0 | 0 |
| Floater | 5 | 100.00 | 100.00 | +0.00 | 0 | 0 |
| Fungus Damage | 8 | 93.33 | 93.33 | +0.00 | 0 | 0 |
| Dry Cherry | 5 | 100.00 | 100.00 | +0.00 | 0 | 0 |
| Partial Black | 6 | 90.91 | 90.91 | +0.00 | 0 | 0 |
| Husk | 5 | 100.00 | 100.00 | +0.00 | 0 | 0 |
| Parchment | 5 | 90.91 | 90.91 | +0.00 | 0 | 0 |
| Immature | 7 | 85.71 | 85.71 | +0.00 | 0 | 0 |
| Shell | 6 | 100.00 | 100.00 | +0.00 | 0 | 0 |
| Severe Insect Damage | 6 | 90.91 | 90.91 | +0.00 | 1 | 1 |
| Withered | 5 | 66.67 | 72.73 | +6.06 | 0 | 0 |
| Slight Insect Damage | 6 | 83.33 | 90.91 | +7.58 | 0 | 0 |
| Broken | 6 | 90.91 | 100.00 | +9.09 | 1 | 0 |

Ringkasan audit:

- 11 dari 17 kelas mempunyai F1 yang sama;
- MSF1 menyelamatkan dua keputusan yang salah pada MSFC;
- MSF1 juga merusak dua keputusan yang benar pada MSFC;
- kenaikan F1 `Withered` dan `Slight Insect Damage`, serta penurunan `Cut`
  dan `Full Sour`, dapat terjadi tanpa perubahan benar/salah pada sampel dari
  kelas aktual tersebut karena F1 juga dipengaruhi false positive dari kelas
  lain;
- support per kelas hanya 3--8 gambar, sehingga perubahan satu keputusan dapat
  menggeser F1 kelas secara besar.

Audit ini menguatkan putusan capacity control: MSF1 melakukan redistribusi
kesalahan, bukan perbaikan bersih atas MSFC. Tidak ada dasar untuk membuka seed
123/2026 atau test.
