# Reproduksi Arwatchananukul et al. (2024)

## Tujuan

Menguji efek HBP+CE terhadap MobileNetV3 standar di bawah protokol yang sejauh
mungkin mengikuti Arwatchananukul et al. (2024). Eksperimen ini hanya untuk
keterbandingan dengan paper; protokol M0/M1 bebas leakage tetap menjadi bukti
utama penelitian.

## Komponen yang dilaporkan paper

- 979 gambar asli, background putih, crop manual 500 x 500;
- enam rotasi: 45, 90, 135, 180, 225, dan 270 derajat;
- total 6.853 gambar termasuk gambar asli;
- augmentasi dilakukan sebelum pembagian 70% train, 20% validation, 10% test;
- ImageNet pretrained MobileNetV3;
- learning rate 0,01;
- konfigurasi terbaik 3 epoch;
- test berisi 686 gambar;
- hasil referensi: accuracy 88,63% dan Macro-F1 89,04%.

## Asumsi operasional

Paper tidak melaporkan optimizer, batch size, ukuran input, fungsi loss, seed,
varian MobileNetV3, atau detail implementasi fold. Diagram menyebut backbone
dibekukan, sedangkan narasi juga memakai istilah fine-tuning.

Reproduksi ini menetapkan:

- `mobilenetv3_large_100` dari timm;
- input 224 x 224 dan batch 32;
- AdamW, CE tanpa label smoothing, weight decay 0;
- LR konstan 0,01 selama 3 epoch;
- backbone frozen dalam mode evaluasi; head tetap trainable;
- split acak non-stratified menggunakan seed yang dilaporkan runner.

Asumsi ini harus disebutkan saat melaporkan hasil dan tidak boleh dipresentasikan
sebagai reproduksi identik.

## Model

| Kode | Model | Parameter yang dilatih |
|---|---|---|
| P0 | frozen MobileNetV3 + GAP + CE | classifier |
| P1 | frozen MobileNetV3 + HBP + CE | HBP dan classifier |

## Leakage yang disengaja untuk reproduksi

Variant rotasi dibuat sebelum split. Audit mencatat berapa identitas gambar asli
yang muncul pada lebih dari satu split. Ini mereproduksi kelemahan protokol paper,
bukan praktik yang direkomendasikan.

Nilai P0/P1 tidak boleh digabungkan dengan hasil clean M0/M1 sebagai satu tabel
tanpa kolom protokol yang jelas.

## Menjalankan di Kaggle

```bash
python -u -m bilinear_lmmd.experiments.run_paper_reproduction \
  --raw-root /kaggle/input/PATH_DATASET_COFFEE17 \
  --data-root /kaggle/working/coffee17-paper-protocol \
  --output-root /kaggle/working/paper-reproduction-results \
  --seeds 42
```

Runner otomatis:

1. membuat 6.853 variant jika data belum tersedia;
2. memastikan test berisi 686 gambar;
3. mencetak audit identity leakage;
4. melatih P0 dan P1;
5. mengevaluasi test dan menyimpan agregat P0 vs P1.

## Interpretasi bersama protokol bersih

- P1 > P0 dan M1 > M0: dukungan HBP konsisten pada kedua protokol.
- P1 > P0 tetapi M1 tidak > M0: peningkatan paper-style mungkin didorong leakage.
- P1 tidak > P0 tetapi M1 > M0: HBP membantu generalisasi asli, bukan memorisasi
  variant rotasi.
- Keduanya gagal: HBP tidak didukung sebagai kontribusi utama.

## Hasil terkunci tiga seed

Eksperimen test dijalankan pada seed 42, 123, dan 2026. Angka delta adalah
mean dan sample standard deviation dari selisih berpasangan P1-P0 per seed.

| Metrik | P0 GAP (%) | P1 HBP (%) | Delta P1-P0 (poin) | Seed membaik |
|---|---:|---:|---:|---:|
| Accuracy | 87,51 | **93,93** | **+6,41 ± 2,26** | **3/3** |
| Macro-F1 | 87,07 | **93,68** | **+6,61 ± 2,33** | **3/3** |
| Hard-class F1 | 83,42 | **91,00** | **+7,58 ± 2,59** | **3/3** |
| Worst-class F1 | 65,43 | **81,41** | **+15,98 ± 6,13** | **3/3** |

P1 mengungguli P0 pada semua metrik dan pada ketiga seed. Dampak terbesar
terlihat pada Worst-class F1, sehingga temuan utamanya adalah HBP membantu
kelas cacat yang paling lemah, bukan hanya menaikkan akurasi rata-rata.

Sebagai pemeriksaan kewajaran reproduksi, P0 memperoleh accuracy 87,51% dan
Macro-F1 87,07%, dekat dengan laporan paper sebesar 88,63% dan 89,04%.
Selisih tetap wajar karena optimizer, batch size, ukuran input, varian
MobileNetV3, dan detail implementasi lain tidak dilaporkan paper dan harus
ditetapkan sebagai asumsi operasional di atas.

Hasil P1 tidak boleh disebut sebagai perbandingan identik dengan angka paper.
Protokol ini membuat variant rotasi sebelum split sehingga identitas gambar
asli dapat muncul lintas train, validation, dan test. Akibatnya, besarnya gain
paper-style mungkin optimistis dan bukan estimasi generalisasi ke biji baru.

## Penemuan lintas protokol

| Protokol | Delta Macro-F1 | Delta Hard-F1 | Delta Worst-F1 |
|---|---:|---:|---:|
| Paper-style P1-P0 | +6,61 | +7,58 | +15,98 |
| Clean grouped M1-M0 | +1,32 | +2,48 | +7,64 |

Arah peningkatan HBP sama pada kedua protokol dan paling besar pada metrik
kelas terburuk. Namun, magnitude pada paper-style jauh lebih besar. Kesimpulan
yang diizinkan adalah bahwa dukungan terhadap HBP muncul pada dua rancangan
evaluasi, sementara protokol clean grouped tetap menjadi bukti utama. Hasil ini
tidak membuktikan HBP selalu unggul pada semua dataset atau setiap seed.

Ringkasan terstruktur hasil ini disimpan di
`docs/results/PAPER_REPRODUCTION_RESULTS.json`.
