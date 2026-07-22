# Master log eksperimen klasifikasi biji kopi

Terakhir diperbarui: **21 Juli 2026**.

Dokumen ini mengonsolidasikan hasil yang sebelumnya tersebar di report, output
Colab/Kaggle, README, dan percakapan eksperimen. Tujuannya adalah mencegah
hasil negatif hilang dan mencegah angka screening diperlakukan sebagai bukti
final.

## Cara membaca status

- **LOCKED TEST**: evaluasi test yang telah dikunci dan boleh menjadi bukti
  utama sesuai batas klaimnya.
- **CONFIRMED**: diulang pada tiga seed atau protokol konfirmasi, tetapi tetap
  dibaca sesuai split yang tertulis.
- **SCREENING**: validation/fail-fast; dipakai untuk keputusan melanjutkan atau
  menghentikan metode, bukan klaim final.
- **ARCHIVED**: arah eksperimen telah dihentikan atau berada di luar fokus
  closed-set fine-grained classification saat ini.
- **NON-INFORMATIVE**: hasil tidak dapat membedakan metode, misalnya dataset
  terlalu mudah atau mempunyai masalah evaluasi.

`Hard-F1` adalah rerata F1 kelas/kelompok sulit yang ditetapkan pada protokol.
`Worst-F1` adalah F1 kelas terburuk. Semua delta adalah poin persentase.

## Ringkasan keputusan

| Arah | Putusan | Bukti ringkas |
|---|---|---|
| HBP pada Coffee17 bersih | Dipertahankan sebagai temuan utama lama | Test tiga seed: Macro +1,32; Hard +2,48; Worst +7,64 |
| Manfaat HBP dipengaruhi granularitas | Didukung, belum universal | Fine gain +3,03 dengan CI positif; difference-in-differences CI masih melintasi nol |
| HBP universal lintas dataset | Ditolak | Kalah pada USK; hasil CBD bergantung seed/protokol |
| SPPF, spatial HBP, MoE, ArcFace, EMA, crop | Ditolak | Tidak konsisten atau menurunkan metrik utama |
| Auxiliary hierarchy | Ditolak pada gate | Macro naik, Hard turun |
| Compact MPN-COV | Ditolak | Macro naik kecil, Hard dan Worst memburuk |
| Decoupled GAP-HBP pada CBD | Sinyal positif kecil | Menang atas HBP dan capacity control, tetapi mahal dan margin kecil |
| Stacking GAP-HBP pada CBD | Lolos sebagai ensemble | Macro +0,86 atas model terkalibrasi terbaik; memerlukan dua model |
| LMMD/synthetic robustness | Diarsipkan | Sinyal pada illumination, gagal cross-shift dan bukan validasi nyata |
| OSR | Dihentikan/diarsipkan | HBP dan ARPL fail-fast gagal |
| OMSL taxonomy contrastive | Dihentikan | Delta dataset-Macro hanya +0,08 |

---

## 1. Coffee17: bukti utama GAP versus HBP

**Status: LOCKED TEST.** Dataset Coffee17 content-clean, grouped split, augmentasi
online hanya pada train, seed 42/123/2026.

| Metrik | M0 MobileNetV3-GAP | M1 MobileNetV3-HBP | Delta berpasangan |
|---|---:|---:|---:|
| Accuracy | 85,26 ± 1,26 | **86,93 ± 3,14** | **+1,68 ± 4,33** |
| Balanced accuracy | 85,39 ± 1,40 | **86,63 ± 3,20** | **+1,24 ± 4,60** |
| Macro-F1 | 85,46 ± 1,31 | **86,78 ± 3,20** | **+1,32 ± 4,51** |
| Hard-F1 | 81,52 ± 2,13 | **84,01 ± 4,58** | **+2,48 ± 6,55** |
| Worst-F1 | 55,64 ± 2,60 | **63,27 ± 3,33** | **+7,64 ± 5,76** |

Efisiensi CUDA batch-1 pada input 224:

| Model | Parameter | FP32 | Latency |
|---|---:|---:|---:|
| M0 GAP | 2.988.289 | 11,40 MB | 5,922 ms |
| M1 HBP | 3.562.305 | 13,59 MB | 7,425 ms |

Interpretasi yang diizinkan: HBP meningkatkan mean semua metrik dalam protokol
ini, terutama kelas terlemah, tetapi variabilitas antarseed tinggi. Jangan
menulis bahwa HBP menang pada setiap seed atau universal.

Dokumen rinci: [FINAL_HBP_RESULTS.md](FINAL_HBP_RESULTS.md).

## 2. Reproduksi protokol Arwatchananukul et al.

**Status: LOCKED TEST untuk keterbandingan paper, bukan bukti generalisasi
utama.** Enam rotasi sebelum split membuat protokol rentan identity leakage.

| Metrik | P0 GAP | P1 HBP | Delta | Seed membaik |
|---|---:|---:|---:|---:|
| Accuracy | 87,51 | **93,93** | **+6,41 ± 2,26** | 3/3 |
| Macro-F1 | 87,07 | **93,68** | **+6,61 ± 2,33** | 3/3 |
| Hard-F1 | 83,42 | **91,00** | **+7,58 ± 2,59** | 3/3 |
| Worst-F1 | 65,43 | **81,41** | **+15,98 ± 6,13** | 3/3 |

Besarnya gain tidak boleh menggantikan hasil clean grouped pada bagian 1.

## 3. Uji granularitas Fine-17 versus Coarse-9

### Validation tiga seed

**Status: CONFIRMED VALIDATION.** Gambar dan split sama; yang berubah hanya
granularitas label.

| Perbandingan | Macro-F1 baseline | Macro-F1 HBP | Gain HBP |
|---|---:|---:|---:|
| Fine-17 | 88,34 | 88,95 | +0,62 ± 1,77 |
| Coarse-9 | 95,54 | 93,76 | -1,78 ± 0,86 |

Difference-in-differences per seed: seed 42 `+0,27`, seed 123 `+3,38`, seed
2026 `+3,52`; mean `+2,39 ± 1,84`, positif 3/3.

### Test dan paired stratified bootstrap

**Status: LOCKED TEST.** Tiga seed, 199 sampel per seed.

| Perbandingan | Baseline Macro-F1 | HBP Macro-F1 | Gain HBP |
|---|---:|---:|---:|
| Fine-17 | 84,95 | 87,97 | **+3,03 ± 1,65** |
| Coarse-9 | 89,56 | 89,98 | +0,42 ± 1,74 |

- Fine gain: point `+3,03`, CI 95% `[+0,57; +5,69]`, P(gain > 0) `0,992`.
- Coarse gain: point `+0,42`, CI `[-2,35; +3,22]`, P(gain > 0) `0,607`.
- Granularity effect: point `+2,60`, CI `[-0,88; +6,25]`, P(effect > 0)
  `0,931`.
- Hierarchical seed-and-sample effect: CI `[-2,74; +8,19]`, P(effect > 0)
  `0,839`.

Kesimpulan: manfaat HBP pada Fine-17 didukung; klaim bahwa gain HBP pasti
meningkat karena granularitas belum signifikan pada CI 95%.

## 4. Screening backbone kontemporer

**Status: SCREENING VALIDATION, seed 123.** Angka ini hanya boleh dibandingkan
di dalam tabel ini.

| Rank | Kode | Backbone | Head | Macro-F1 | Hard-F1 | Worst-F1 |
|---:|---|---|---|---:|---:|---:|
| 1 | BE2H | EfficientNetV2-B0 | HBP | **88,65** | 80,36 | 66,67 |
| 2 | BE2G | EfficientNetV2-B0 | GAP | 88,24 | **81,59** | **66,67** |
| 3 | BSHG | SHViT-S1 | GAP | 83,69 | 75,13 | 66,67 |
| 4 | BV4G | MobileNetV4-Conv-Medium | GAP | 83,28 | 72,20 | 36,36 |
| 5 | BV4H | MobileNetV4-Conv-Medium | HBP | 83,28 | 69,52 | 28,57 |
| 6 | BP2G | PVTv2-B0 | GAP | 83,10 | 73,00 | 57,14 |
| 7 | BC2G | ConvNeXtV2-Atto | GAP | 82,46 | 72,85 | 54,55 |
| 8 | BSHH | SHViT-S1 | HBP | 81,50 | 67,13 | 0,00 |
| 9 | BP2H | PVTv2-B0 | HBP | 77,13 | 62,96 | 0,00 |
| 10 | BC2H | ConvNeXtV2-Atto | HBP | 71,01 | 62,79 | 0,00 |

Temuan: EfficientNetV2-B0 paling kuat pada screening ini. Implementasi HBP
tidak otomatis kompatibel/unggul pada seluruh feature hierarchy backbone.

## 5. Cross-dataset HBP

### USK-Coffee empat kelas

**Status: CONFIRMED VALIDATION.**

| Model | Accuracy | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|---:|
| U1 MobileNetV2-GAP | 93,50 | 93,50 | 93,17 | 91,88 |
| U2 MobileNetV3-GAP | **95,62** | **95,65** | **94,80** | **93,91** |
| U3 MobileNetV3-HBP | 94,75 | 94,77 | 94,12 | 93,71 |

HBP kalah dari GAP MobileNetV3: Macro `-0,88`, Hard `-0,68`, Worst `-0,19`.
Hasil ini menolak klaim HBP universal.

### Coffee roast dataset

**Status: NON-INFORMATIVE.** MobileNetV3-GAP dan HBP sama-sama mencapai
Accuracy/Macro/Hard/Worst `100%`. Dataset/split ini terlalu mudah untuk menguji
kontribusi HBP dan sebelumnya juga memerlukan audit duplicate lintas split.

### Roboflow CBD-Multiclassify

Dataset bersih: 4.012 gambar berlabel, delapan kelas; train/val/test
2.408/803/801. Dua unlabeled dan satu stray `Insect` dikeluarkan; tidak ada
exact duplicate group atau identity leak yang ditemukan oleh preparer.

Screening CE awal pada validation:

| Model | Accuracy | Macro-F1 | Defect-F1 | Worst-F1 |
|---|---:|---:|---:|---:|
| CBD0 GAP | **94,15** | **91,84** | **90,75** | **80,00** |
| CBD1 HBP | 93,52 | 90,95 | 89,71 | 77,84 |

Balanced Softmax juga gagal: GAP Macro `90,27`; HBP Macro `87,19`.

## 6. Ablasi fine-grained Coffee17

Hasil di bawah adalah screening kecuali dinyatakan lain.

| Metode | Pembanding | Delta Macro | Delta Hard | Delta Worst | Putusan |
|---|---|---:|---:|---:|---|
| Resolusi HBP 320 (F1) | HBP 224 (M1) | -3,49 | -4,40 | -6,06 | Gagal |
| ArcFace pada GAP (A2) | GAP+CE (M0) | +0,10 | -0,41 | -9,79 | Gagal |
| ArcFace pada HBP (A3) | HBP+CE (M1) | -4,43 | -5,81 | -11,86 | Gagal |
| Spatial HBP 14×14 (M1s) | HBP 7×7 (M1) | +0,57 | +1,21 | -6,67 | Gagal |
| Global-local MoE (E1), konfirmasi | HBP (M1) | -1,01 | -0,49 | -4,16 | Gagal |
| Object-centric crop (O1) | HBP tanpa crop (M1) | -6,96 | -11,78 | -15,87 | Gagal |
| EMA (M1e), 3 seed | raw HBP (M1) | -0,43 ± 0,73 | +0,51 ± 2,77 | -2,78 ± 4,81 | Gagal |
| SPPF-HBP (S1), final test | HBP (M1) | -1,53 | -0,72 | -6,20 | Gagal |
| Capacity residual (C1), konfirmasi | HBP (M1) | -0,30 | -0,39 | -4,04 | Gagal |
| Full-family hierarchy H0f | GAP M0 | +1,30 | -0,16 | +11,54 | Gagal gate |
| Compact MPN-COV, 3 seed | GAP COV0 | +1,08 ± 1,01 | -0,09 ± 2,20 | -13,54 ± 17,29 | Gagal |

Catatan hierarchy lama H0: Macro `+1,22`, Hard `-0,34`, Worst `+16,67`.
Perbaikan terutama terjadi pada kelompok insect damage (`+17,42`), tetapi
shape/withered turun `-12,52`; karena Hard-F1 turun, hierarchy tidak diteruskan.

### SPPF faktorial

Validation tiga seed:

| Model | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| M0 GAP | 88,55 | 81,34 | 64,32 |
| M1 HBP | **89,65** | 81,86 | 66,46 |
| S0 SPPF-GAP | 88,06 | 80,05 | **70,71** |
| S1 SPPF-HBP | 89,50 | **82,57** | **70,71** |

S1 tampak melindungi Worst-F1 pada validation tetapi gagal pada test, sehingga
tidak dipilih.

### Compact MPN-COV per seed

| Seed | GAP Macro/Hard/Worst | MPN-COV Macro/Hard/Worst |
|---:|---|---|
| 42 | 88,71 / 82,20 / 66,67 | 90,70 / 84,31 / 72,73 |
| 123 | 88,24 / 81,59 / 66,67 | 89,51 / 81,54 / 40,00 |
| 2026 | 89,36 / 83,32 / 80,00 | 89,34 / 81,02 / 60,00 |

Seed 42 memberi sinyal kuat, tetapi tidak bereplikasi.

## 7. Ciri klasik, complementarity, dan ensemble

### Ablasi warna, bentuk, tekstur

| Fitur | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| Color+Shape+Texture | **71,74** | **65,95** | **49,52** |
| Color+Shape | 69,10 | 61,93 | 30,77 |
| Color+Texture | 65,79 | 61,64 | 41,18 |
| Color | 60,60 | 55,62 | 36,00 |
| Shape+Texture | 54,85 | 48,70 | 25,35 |
| Texture | 44,94 | 40,71 | 9,52 |
| Shape | 30,58 | 24,10 | 8,33 |

Pada 979 sampel OOF, CST dan HBP mempunyai oracle `89,79%`: keduanya benar 653,
hanya CST benar 51, hanya HBP benar 175, keduanya salah 100. Namun disagreement
gate memilih CST hanya pada delapan sampel dan menghasilkan Macro `84,30%`
dibanding HBP `84,41%` (`-0,11`), sehingga gate ditolak.

### OOF GAP-HBP historis

**Status: HISTORICAL; dataset 979 sebelum clean final.** GAP Macro `82,42`, HBP
`84,41` (`+1,99`); GAP Hard `74,59`, HBP `77,37`; GAP Worst `56,84`, HBP
`54,12`. Oracle GAP/HBP `89,17%`, menunjukkan komplementaritas tetapi juga 106
sampel yang sama-sama salah. Weighted ensemble terbaik meningkatkan Worst ke
`56,84` tetapi menurunkan Macro menjadi `83,48` dari HBP `84,41`.

## 8. CBD: stacking dan decoupled dual branch

### Stacking tiga seed

**Status: CONFIRMED TEST untuk protokol CBD stacking.**

| Model | Macro-F1 | Worst-F1 |
|---|---:|---:|
| GAP raw | 90,66 ± 1,01 | 76,31 ± 0,76 |
| HBP raw | 90,44 ± 0,26 | 75,47 ± 1,46 |
| GAP calibrated | 91,46 ± 0,26 | 75,54 ± 1,27 |
| HBP calibrated | 90,85 ± 0,77 | 77,03 ± 0,30 |
| **Stacking** | **92,32 ± 0,21** | **77,37 ± 2,89** |

Fusion margin terhadap model terkalibrasi terbaik: Macro `+0,86`, Worst
`+0,34`; keputusan `PASS`. Kekurangannya adalah dua model dan meta-classifier.

KD seed 42 hanya `SCREEN_ONLY`: GAP raw Macro/Worst `90,01/72,83`, KD control
`92,11/75,00`, stacking teacher `91,17/75,82`, stacking-KD `91,35/76,09`.
Stacking-KD kalah `-0,76` Macro dari KD calibration control.

### Learned decoupled GAP-HBP

**Status: CONFIRMED VALIDATION DAN TEST.**

| Split | CBD1 HBP Macro/Hard/Worst | CBDD2 dual Macro/Hard/Worst | Delta dual |
|---|---|---|---|
| Validation | 90,46 / 89,21 / 77,32 | **91,48 / 90,37 / 80,98** | +1,01 / +1,16 / +3,66 |
| Test | 89,60 / 88,17 / 74,35 | **90,83 / 89,58 / 76,62** | +1,23 / +1,42 / +2,27 |

Capacity-matched CBDC1 versus CBDD2:

| Split | CBDC1 Macro/Hard/Worst | CBDD2 Macro/Hard/Worst | Delta dual |
|---|---|---|---|
| Validation | 90,92 / 89,71 / 79,34 | 91,48 / 90,37 / 80,98 | +0,56 / +0,65 / +1,64 |
| Test | 90,26 / 88,94 / 76,62 | 90,83 / 89,58 / 76,62 | +0,57 / +0,65 / +0,00 |

Efisiensi:

| Model | Parameter | FP32 | Latency b1 | Throughput b32 |
|---|---:|---:|---:|---:|
| CBD1 HBP | 3.548.472 | 13,54 MB | 6,909 ms | 1.141,7 img/s |
| CBDC1 capacity HBP | 5.735.706 | 21,88 MB | 6,908 ms | 1.052,7 img/s |
| CBDD2 dual | 5.736.234 | 21,88 MB | 8,737 ms | 1.004,6 img/s |

Kesimpulan: dual branch membawa sinyal yang melampaui penambahan kapasitas,
tetapi margin kecil dan biaya deployment naik. Ini bukan pengganti otomatis
model ringan.

## 9. XAI final M0 versus M1

**Status: ANALISIS POST-HOC.** Sebanyak 36 sampel test: sembilan per outcome
`rescued_by_hbp`, `harmed_by_hbp`, `both_correct`, dan `both_wrong`.

| Metode | Delta foreground | Delta leakage | Delta relative confidence drop |
|---|---:|---:|---:|
| LayerCAM | -2,62 | +2,62 | +0,1381 |
| Finer-LayerCAM | -2,42 | +2,42 | +0,1345 |

HBP tidak lebih terkonsentrasi pada foreground. Pada rescued samples,
Finer-LayerCAM relative drop `+0,6174`; pada harmed samples `-0,2764`. Temuan
yang diizinkan adalah region HBP lebih diskriminatif pada sampel yang berhasil
diselamatkan, bukan klaim bahwa HBP selalu lebih fokus pada objek.

## 10. Eksperimen robustness/domain adaptation

**Status: ARCHIVED; bukan fokus fine-grained saat ini dan bukan validasi dunia
nyata.**

Controlled synthetic combined domain, seed 123:

| Model | Target Macro | Hard | Worst |
|---|---:|---:|---:|
| M0 source-only GAP | 21,05 | 12,92 | 0,00 |
| M2 MMD-GAP | **31,39** | **24,32** | 0,00 |
| M3 LMMD-GAP | 29,98 | 23,64 | 0,00 |

Illumination M1 versus M5w01, konfirmasi seed 42 dan 2026:

- target Macro delta `+15,49 ± 1,99`;
- Hard delta `+16,51 ± 0,50`;
- Worst delta `+17,56 ± 4,16`;
- source delta `-2,40 ± 2,79`;
- keputusan illumination `PASS` 2/2 seed.

Cross-shift gagal karena source retention:

| Domain | Delta Macro | Delta Hard | Delta Worst | Delta Source | Putusan |
|---|---:|---:|---:|---:|---|
| Sensor | +10,62 | +8,91 | +0,00 | -10,29 | FAIL |
| Background | +26,24 | +25,68 | +11,11 | -20,25 | FAIL |

Kesimpulan: bobot LMMD 0,1 membantu shift illumination sintetis tetapi tidak
robust lintas jenis shift dan tidak membuktikan performa dunia nyata.

## 11. OMSL multi-source heterogeneous labels

**Status: ARCHIVED/STOP.**

| Model | Dataset-Macro | Balanced | Coffee17 Macro/Acc | CBD Macro/Acc |
|---|---:|---:|---:|---:|
| OMSL0 | 90,77 | 90,75 | 90,16 / 89,69 | **91,38 / 93,77** |
| OMSL1 + taxonomy contrastive | **90,85** | **91,23** | **90,71 / 90,72** | 90,98 / **94,02** |

Delta dataset-Macro hanya `+0,08`; terlalu kecil dan berpindah trade-off antar
dataset. Taxonomy contrastive tidak diteruskan.

## 12. Open-set recognition

**Status: ARCHIVED/STOP atas keputusan penelitian. Jangan dicampur dengan fokus
closed-set fine-grained classification.**

Baseline post-hoc tiga seed:

| Tier | Skor terbaik menurut OSCR | OSCR | AUROC | FPR95 | Known Macro-F1 |
|---|---|---:|---:|---:|---:|
| Near | MSP | 71,56 ± 1,33 | 77,47 ± 1,10 | 85,71 | 87,72 ± 1,70 |
| Medium | Prototype | 67,18 ± 3,69 | 72,32 ± 4,88 | 85,71 | 85,84 ± 1,64 |
| Far | MSP | 66,68 ± 2,42 | 77,44 ± 3,26 | 79,37 | 79,47 ± 2,01 |

Fail-fast HBP versus GAP:

| Tier | Delta AUROC | Delta OSCR | Delta known Macro | Putusan |
|---|---:|---:|---:|---|
| Near | -18,12 | -16,38 | +3,37 | FAIL |
| Medium | -20,94 | -18,27 | +3,25 | FAIL |

Fail-fast ARPL-no-CS:

| Tier | Delta AUROC | Delta OSCR | Delta known Macro | Putusan |
|---|---:|---:|---:|---|
| Near | -3,30 | -7,82 | -5,36 | FAIL |
| Medium | -6,07 | -5,83 | -2,56 | FAIL |

OSR dihentikan; runner dipertahankan hanya untuk reproduktibilitas sejarah.

## 13. Keterbatasan arsip

1. Hanya dua artefak hasil statis lama yang sebelumnya committed:
   `FINAL_HBP_RESULTS.md` dan `PAPER_REPRODUCTION_RESULTS.json`. Angka lain di
   master log ini ditranskripsi dari output eksperimen yang pernah ditampilkan.
2. Checkpoint dan report Colab/Kaggle yang tidak disalin ke persistent storage
   dapat hilang setelah runtime reset. Karena itu setiap eksperimen baru wajib
   menyimpan `metrics.json`, `predictions.csv`, config, seed, dan commit hash.
3. Baris `SCREENING` tidak boleh dipindahkan ke tabel hasil utama tesis tanpa
   konfirmasi split/seed yang telah didaftarkan.
4. Beberapa eksperimen historis memakai dataset 979 sebelum audit duplicate;
   bagian tersebut telah ditandai dan tidak menjadi bukti utama.
5. Hasil test yang sudah dibuka tidak boleh digunakan untuk tuning ulang model.

## 14. Progressive multi-granularity EfficientNetV2

### Screening validation seed 123

**Status: SCREENING PASS untuk E2; E3 dihentikan.** Seluruh model memakai
EfficientNetV2-B0, input 224, dan split benchmark backbone yang sama.

| Model | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| E0/BE2G GAP | 88,24 | 81,59 | 66,67 |
| E1/BE2H HBP | 88,65 | 80,36 | 66,67 |
| **E2 progressive multi-granularity** | **90,46** | **87,14** | **76,92** |
| E3 E2 + category consistency | 90,06 | 83,25 | 72,73 |

E2 versus GAP: Macro `+2,21`, Hard `+5,54`, Worst `+10,26`. E2 versus HBP:
Macro `+1,81`, Hard `+6,78`, Worst `+10,26`. E2 lolos seluruh gate.

E3 versus E2 menurunkan Macro `-0,40`, Hard `-3,89`, dan Worst `-4,20`.
Category consistency ditolak dan tidak dibawa ke seed konfirmasi. Hasil ini
masih satu seed validation; test belum boleh dibuka.

### Keputusan konfirmasi

**Status: KONFIRMASI VALIDATION SELESAI — E2 gagal melampaui HBP dan
dihentikan.** E2 dikunci tanpa tuning tambahan, lalu E0/E1/E2 dikonfirmasi pada
seed 42/123/2026. Test tidak dibuka.

| Model | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---:|---:|---:|
| E0/BE2G GAP | 88,77 | 82,37 | 71,11 |
| **E1/BE2H HBP** | **89,93** | 82,47 | 70,09 |
| E2 progressive multi-granularity | 89,44 | **83,50** | **72,11** |

| Perbandingan | Metrik | Delta mean | Std delta | Naik/seed |
|---|---|---:|---:|---:|
| E2 vs GAP | Macro-F1 | +0,67 | 1,84 | 2/3 |
| E2 vs GAP | Hard-F1 | +1,13 | 4,32 | 2/3 |
| E2 vs GAP | Worst-F1 | +0,99 | 8,81 | 1/3 |
| E2 vs HBP | Macro-F1 | -0,48 | 2,11 | 1/3 |
| E2 vs HBP | Hard-F1 | +1,03 | 5,01 | 1/3 |
| E2 vs HBP | Worst-F1 | +2,02 | 10,84 | 2/3 |

Menurut gate numerik, E2 **PASS versus GAP** karena rerata Macro/Hard naik dan
Worst tidak turun. Namun, peningkatannya kecil dibanding simpangan antar-seed;
Worst hanya naik pada 1/3 seed. E2 **FAIL versus HBP** karena rerata Macro-F1
turun 0,48 poin dan hanya naik pada 1/3 seed. Dengan demikian hasil seed 123
adalah screening yang terlalu optimistis, bukan bukti superioritas stabil.

Putusan akhir: E2 tidak menjadi metode utama, E3 tetap ditolak, adaptasi
progressive multi-granularity dihentikan, dan test tetap tertutup. Artefak
ringkas yang dapat diproses mesin disimpan pada
`docs/results/PROGRESSIVE_MULTIGRANULARITY_CONFIRMATION.json`.

## 15. Arah aktif setelah pembekuan log

Fokus aktif adalah **closed-set fine-grained classification**, bukan OSR,
deteksi, atau domain adaptation. Kandidat selanjutnya harus merupakan algoritma
fine-grained yang mempunyai mekanisme dan baseline terkontrol, bukan penambahan
head secara acak. Jalur progressive multi-granularity (E2) dan
category-consistency (E3) telah ditutup sebagai hasil negatif setelah
konfirmasi tiga seed. Eksperimen berikutnya tidak boleh mengulang tuning E2/E3
atau membuka test untuk menyelamatkan hasil.

## 16. Confusion-aware pairwise learning

**Status: SCREENING FAIL — DIHENTIKAN.** Pola eksperimen sebelumnya
menunjukkan bahwa penambahan statistik, kapasitas, attention, atau granularitas
umumnya memindahkan kesalahan antarkelas dan tidak memberi superioritas stabil.
Hipotesis baru secara langsung menargetkan batas antar pasangan kelas yang
sering tertukar.

| Kode | Model | Objective |
|---|---|---|
| BE2G | EfficientNetV2-B0 GAP | CE, checkpoint lama |
| BE2H | EfficientNetV2-B0 HBP | CE, baseline terkuat lama |
| CP1 | EfficientNetV2-B0 GAP | CE + vanilla SupCon |
| CP2 | EfficientNetV2-B0 GAP | CE + train-only confusion-aware SupCon |

Screening dikunci pada validation seed 123. CP2 harus mengalahkan CP1 untuk
membuktikan kontribusi pembobotan confusion dan mengalahkan BE2H untuk menjadi
kandidat utama. Projection head hanya untuk training dan dibuang dari
checkpoint inference. Confusion validation hanya untuk audit diagnostik;
sampler CP2 dibangun dinamis dari prediction train. Test belum boleh dibuka.

Putusan yang dilaporkan runner:

| Perbandingan | Putusan |
|---|---|
| BE2G vs CP1 | FAIL |
| BE2H vs CP1 | FAIL |
| BE2G vs CP2 | FAIL |
| BE2H vs CP2 | FAIL |
| CP1 vs CP2 | FAIL |
| CP2 final | FAIL |

Angka metrik tidak tersedia pada output yang diarsipkan dan tidak diisi ulang
secara spekulatif. CP1/CP2 dihentikan pada screening; tiga seed dan test tidak
dijalankan.

Protokol lengkap:
`docs/protocols/CONFUSION_AWARE_PAIRWISE.md`.

## 17. Adaptasi klasifikasi Hong: DSConv x SPPF-Attention

**Status: SCREENING SELESAI — KOMBINASI FAIL; DSConv-ONLY LOLOS.** Eksperimen ini mengadaptasi dua
komponen representasi Hong et al. ke EfficientNetV2-B0 untuk klasifikasi
Coffee17. Tidak ada YOLO, PConv, HBP, bounding-box loss, atau klaim detector.

| Kode | DSConv | SPPF-Attention | Head |
|---|---:|---:|---|
| BE2G | tidak | tidak | GAP + CE, baseline lama |
| HCD1 | ya | tidak | GAP + CE |
| HCS1 | tidak | ya | GAP + CE |
| HCDS1 | ya | ya | GAP + CE |

DSConv adalah Distribution Shifting Convolution VQK/KDS/CDS 4-bit, block size
128, pada lima full spatial convolution stage awal/menengah. Forward PyTorch
merupakan simulasi quantization-aware; tidak ada klaim speedup atau ukuran
checkpoint integer. SPPF mengikuti tiga max-pool 5x5 berurutan, channel/spatial
attention, residual, lalu GAP.

Screening dikunci pada validation seed 123. HCDS1 final harus mengalahkan BE2H,
HCD1, dan HCS1 dengan Macro/Hard naik serta Worst tidak turun lebih dari satu
poin. Jika gagal, tiga seed dan test tidak dijalankan.

Putusan yang dilaporkan runner:

| Perbandingan | Putusan |
|---|---|
| BE2G vs HCD1 | PASS |
| BE2H vs HCD1 | PASS |
| BE2G vs HCS1 | FAIL |
| BE2H vs HCS1 | FAIL |
| BE2G vs HCDS1 | FAIL |
| BE2H vs HCDS1 | PASS |
| HCD1 vs HCDS1 | FAIL |
| HCS1 vs HCDS1 | PASS |
| HCDS1 final | FAIL |

Kesimpulan screening: kontribusi yang menjanjikan berasal dari **DSConv saja
(HCD1)**. SPPF-Attention gagal sebagai modul tunggal dan, ketika ditambahkan
ke HCD1, membuat kombinasi kalah dari HCD1. Karena itu HCS1 dan HCDS1
dihentikan. HCD1 belum menjadi hasil final: ia hanya kandidat untuk protokol
konfirmasi multi-seed terpisah. Test tetap terkunci dan tidak ada klaim runtime
DSConv. Angka delta metrik belum diarsipkan dan tidak direkonstruksi secara
spekulatif dari putusan PASS/FAIL.

Protokol lengkap:
`docs/protocols/HONG_CLASSIFICATION_PROTOCOL.md`.

### 17.1 Konfirmasi HCD1

**Status: KONFIRMASI FAIL — DSCONV DIHENTIKAN.** Hanya HCD1 diteruskan pada seed
42, 123, dan 2026. HCD1 harus mengalahkan BE2G dan BE2H pada rata-rata
Macro-F1 dan Hard-F1, meningkat minimal 2/3 seed untuk keduanya, serta menjaga
rata-rata Worst-F1 dalam toleransi satu poin. Test tetap terkunci.

| Perbandingan | Macro baseline → HCD1 | Delta Macro | Delta Hard | Delta Worst |
|---|---:|---:|---:|---:|
| BE2G vs HCD1 | 88,77 → 88,28 | -0,49 ± 1,85 | -1,65 ± 3,43 | -7,98 ± 19,31 |
| BE2H vs HCD1 | 89,93 → 88,28 | -1,65 ± 2,04 | -1,75 ± 3,34 | -6,95 ± 8,67 |

Kedua perbandingan FAIL. Sinyal screening seed 123 tidak bertahan lintas seed;
HCD1 menurunkan mean Macro, Hard, dan Worst. DSConv, SPPF, serta kombinasinya
dihentikan. Test tidak dibuka dan DSConv+HBP tidak dijalankan sebagai tuning
post-hoc.

Audit per-seed mengonfirmasi bahwa seed 123 tidak berubah: HCD1 masih memberi
Macro `+0,90` vs GAP dan `+0,50` vs HBP. Namun seed 2026 jatuh `-2,59/-3,57`
poin Macro, `-5,60/-4,86` Hard, dan `-30,00/-16,67` Worst terhadap GAP/HBP.
Dengan demikian kegagalan agregat berasal dari ketidakstabilan nyata, bukan
perubahan checkpoint seed 123.

Protokol lengkap: `docs/protocols/HONG_DSCONV_CONFIRMATION.md`.

## 18. Chang-Liu multiscale defect extraction

**Status: SCREENING FAIL -- DIHENTIKAN.** Adaptasi ini menempatkan cabang
konvolusi standar 3x3 dan 5x5 secara paralel pada feature terdalam
EfficientNetV2-B0, melakukan fusi 1x1 dan residual add, lalu GAP + CE. MDE0
adalah kontrol residual pointwise tanpa receptive field spasial. Selisih total
parameter MDE0/MDE1 hanya `0,0022%`.

| Perbandingan | Delta Macro | Delta Hard | Delta Worst | Putusan |
|---|---:|---:|---:|---|
| BE2G vs MDE0 | -1,07 | -0,91 | +0,00 | FAIL |
| BE2H vs MDE0 | -3,15 | -3,19 | -10,26 | FAIL |
| BE2G vs MDE1 | +0,65 | -0,66 | +0,00 | FAIL |
| BE2H vs MDE1 | -1,43 | -2,94 | -10,26 | FAIL |
| MDE0 vs MDE1 | +1,72 | +0,25 | +0,00 | PASS |

MDE1 mengalahkan kontrol kapasitas, sehingga operator spasial multiscale
memiliki sinyal kausal pada screening ini. Namun MDE1 menurunkan Hard-F1
terhadap GAP dan tertinggal dari HBP pada semua metrik. Sesuai gate yang
dibekukan, seed 123/2026 tidak dijalankan dan test tidak dibuka.

Protokol dan angka lengkap:
`docs/protocols/CHANG_LIU_MDE_PROTOCOL.md` dan
`docs/results/CHANG_LIU_MDE_SCREENING.json`.
