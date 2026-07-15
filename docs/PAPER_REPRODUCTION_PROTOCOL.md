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
python -u -m bilinear_lmmd.run_paper_reproduction \
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
