# Hasil final MobileNetV3-GAP vs MobileNetV3-HBP

Tanggal penguncian hasil: **16 Juli 2026**.

Dokumen ini adalah catatan hasil final untuk eksperimen source-only Coffee-17.
Hasil screening lama, OOF yang masih mengandung exact duplicate, eksperimen
domain sintetis, dan benchmark USK-Coffee tidak menggantikan angka test yang
dicatat di sini.

## Protokol yang dikunci

- Dataset: Coffee Green Bean with 17 Defects, versi content-clean.
- Split: `data/coffee17_hierarchy_clean/folds/fold_1`.
- Training memakai augmentasi online hanya pada train.
- Evaluasi final memakai split test yang tidak digunakan untuk tuning.
- Seed: 42, 123, dan 2026.
- M0: MobileNetV3-Large + GAP + CE.
- M1: MobileNetV3-Large + HBP + CE.
- Semua angka berikut adalah mean dan sample standard deviation tiga seed.

## Hasil test final

| Metrik | M0 GAP (%) | M1 HBP (%) | Delta berpasangan M1-M0 (poin) |
|---|---:|---:|---:|
| Accuracy | 85,26 ± 1,26 | **86,93 ± 3,14** | **+1,68 ± 4,33** |
| Balanced accuracy | 85,39 ± 1,40 | **86,63 ± 3,20** | **+1,24 ± 4,60** |
| Macro-F1 | 85,46 ± 1,31 | **86,78 ± 3,20** | **+1,32 ± 4,51** |
| Hard-class F1 | 81,52 ± 2,13 | **84,01 ± 4,58** | **+2,48 ± 6,55** |
| Worst-class F1 | 55,64 ± 2,60 | **63,27 ± 3,33** | **+7,64 ± 5,76** |

M1 mengungguli M0 pada mean seluruh metrik yang ditetapkan. Keuntungan terbesar
muncul pada Worst-class F1, sehingga bukti utama manfaat HBP adalah pengurangan
kegagalan ekstrem pada kelas lemah, bukan hanya kenaikan rata-rata.

Variabilitas M1 antarseed lebih tinggi daripada M0. Standard deviation delta
Macro-F1 dan Hard-class F1 juga lebih besar daripada mean deltanya. Dengan hanya
tiga seed, hasil ini tidak boleh ditulis sebagai bukti signifikansi statistik
atau klaim bahwa HBP menang pada setiap seed. Formulasi yang diizinkan adalah
"secara rata-rata pada tiga seed".

## Efisiensi batch-1

Benchmark dilakukan pada CUDA yang sama, input 224, batch 1, 20 warmup, dan
100 iterasi. Angka ini mengukur forward model saja, bukan pipeline conveyor
end-to-end.

| Metrik | M0 GAP | M1 HBP | Perubahan M1 |
|---|---:|---:|---:|
| Parameter | 2.988.289 | 3.562.305 | +574.016 (+19,2%) |
| Estimasi ukuran FP32 | 11,40 MB | 13,59 MB | +2,19 MB (+19,2%) |
| Latency batch-1 | 5,922 ms | 7,425 ms | +1,503 ms (+25,4%) |
| Throughput model teoretis | 168,9 gambar/detik | 134,7 gambar/detik | -20,3% |

Throughput tersebut tidak memasukkan akuisisi kamera, deteksi biji, crop,
preprocessing, transfer data, dan postprocessing. Karena itu angka 134,7
gambar/detik tidak boleh disebut sebagai FPS conveyor atau perangkat edge.

## Keputusan model

**M1 (MobileNetV3-Large + HBP + CE) dikunci sebagai model final Coffee-17.**

Biaya absolut HBP adalah sekitar 2,19 MB FP32 dan 1,50 ms per gambar pada GPU
pengujian. Trade-off ini diterima karena mean seluruh metrik meningkat dan
Worst-class F1 naik 7,64 poin. M0 tetap menjadi baseline serta opsi yang lebih
ringan dan lebih stabil terhadap seed.

Klaim ini khusus untuk protokol Coffee-17. Pada benchmark USK-Coffee empat
kelas yang terpisah, MobileNetV3-GAP memperoleh Macro-F1 validation 95,65%,
sedangkan MobileNetV3-HBP 94,77%. Hasil tersebut mencegah klaim bahwa HBP
universal; manfaatnya bergantung pada karakter fine-grained dataset.

## Ablasi SPPF-Attention

Ablasi validation faktorial menghasilkan:

| Model | Macro-F1 (%) | Hard-F1 (%) | Worst-F1 (%) |
|---|---:|---:|---:|
| M0: GAP | 88,55 | 81,34 | 64,32 |
| M1: HBP | **89,65** | 81,86 | 66,46 |
| S0: SPPF-Attention + GAP | 88,06 | 80,05 | **70,71** |
| S1: SPPF-Attention + HBP | 89,50 | **82,57** | **70,71** |

S1 tampak melindungi kelas terburuk pada validation, tetapi gagal pada test:

| Metrik test | M1 HBP (%) | S1 SPPF-HBP (%) | Delta S1-M1 (poin) |
|---|---:|---:|---:|
| Macro-F1 | **86,78** | 85,25 | -1,53 |
| Hard-F1 | **84,01** | 83,29 | -0,72 |
| Worst-F1 | **63,27** | 57,07 | -6,20 |

SPPF-Attention ditolak sebagai komponen model final. Hasil test tidak digunakan
untuk men-tuning ulang S1; S0, S1, dan capacity-matched C1 dipertahankan sebagai
ablasi negatif.

## Artefak pelaporan

Runner `python -m bilinear_lmmd.run_final_hbp_report` menghasilkan:

- `final_summary.json`: angka lengkap dan benchmark efisiensi;
- `per_seed.csv`: hasil serta delta berpasangan setiap seed;
- `per_class.csv`: mean F1 per kelas, delta, dan jumlah seed yang membaik;
- `FINAL_HBP_REPORT.md`: ringkasan siap baca.

Report per kelas dan confusion matrix harus tetap disertakan pada lampiran tesis;
angka agregat di dokumen ini tidak menggantikannya.

## XAI model final

Runner `python -m bilinear_lmmd.run_final_hbp_xai` membandingkan M0 dan M1 pada
sampel test yang dipilih deterministik dari empat outcome: `rescued_by_hbp`,
`harmed_by_hbp`, `both_correct`, dan `both_wrong`. Setiap panel memuat input,
raw LayerCAM heatmap, overlay LayerCAM, raw Finer-LayerCAM heatmap, dan overlay
Finer-LayerCAM untuk kedua model. Target penjelasan adalah kelas aktual.

XAI hanya dipakai setelah model dikunci. Heatmap tidak digunakan untuk memilih
checkpoint atau tuning. Interpretasi wajib menyandingkan visual dengan
foreground mass, background leakage, dan relative confidence drop; visual yang
menarik saja tidak membuktikan model memakai fitur secara kausal.
