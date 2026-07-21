# EfficientNetV2 progressive multi-granularity protocol

## Pertanyaan penelitian

Apakah progressive learning pada detail lokal, menengah, dan global meningkatkan
klasifikasi fine-grained Coffee17 dibanding EfficientNetV2-B0 dengan GAP/HBP,
dan apakah category-consistency antar gambar sekelas memberi kontribusi di luar
progressive training itu sendiri?

## Pijakan literatur

- Du et al., *Fine-Grained Visual Classification via Progressive
  Multi-Granularity Training of Jigsaw Patches*, ECCV 2020:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3399_ECCV_2020_paper.php
- Implementasi resmi PMG:
  https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training
- Du et al., *Progressive Learning of Category-Consistent Multi-Granularity
  Features for Fine-Grained Visual Classification*, TPAMI 2021/2022,
  DOI 10.1109/TPAMI.2021.3126668.
- Implementasi resmi PMG-V2: https://github.com/PRIS-CV/PMG-V2

## Batas adaptasi

Eksperimen ini **bukan reproduksi identik PMG-V2**. PMG-V2 resmi menyisipkan
consistent block convolution (CCBC) ke internal stage ResNet50 pada input 448.
Repository ini memakai EfficientNetV2-B0, input 224, dan endpoint timm reduction
4/16/32. Memaksakan nama PMG-V2 tanpa batas ini akan menjadi klaim yang salah.

Komponen yang dipertahankan:

1. tiga branch feature hierarchy;
2. progressive jigsaw grid 8x8, 4x4, dan 2x2 dari PMG;
3. satu optimizer update per granularitas dan satu update concat per batch;
4. combined inference dari penjumlahan tiga branch logits dan concat logits;
5. E3 mengambil independently augmented positive image sekelas;
6. descriptor consistency mengikuti urutan PMG-V2: average pool blok 7x7,
   global max, lalu min-max channel normalization;
7. dynamic balancing `CE / MSE` dengan bobot stage 0,01/0,05/0,1.

Komponen yang tidak diklaim:

- ResNet CCBC/split-concat internal;
- input 448 dan hyperparameter resmi CUB;
- reproduksi angka benchmark CUB/Cars/Aircraft.

Nama yang benar adalah **EfficientNetV2 progressive multi-granularity
adaptation**, dengan category-consistency pada E3.

## Model terkontrol

| Kode | Backbone | Metode |
|---|---|---|
| E0 = BE2G | EfficientNetV2-B0 | GAP + CE, checkpoint lama |
| E1 = BE2H | EfficientNetV2-B0 | HBP + CE, checkpoint lama |
| E2 | EfficientNetV2-B0 | progressive three-branch PMG adaptation |
| E3 | EfficientNetV2-B0 | E2 + same-class category consistency |

E0/E1 tidak dilatih ulang jika checkpoint untuk seed dan protokol identik sudah
tersedia. E2 dan E3 memakai image 224, split bersih yang sama, augmentasi
standard yang sama, 50 epoch, AdamW 3e-4, weight decay 1e-4, label smoothing
0,1, dan cosine schedule seperti benchmark backbone.

E2/E3 menambah tiga projection block dan classifier. Karena itu E2 versus E0
mengukur keseluruhan algoritma progressive, bukan parameter-matched effect.
E3 versus E2 adalah isolasi utama kontribusi category consistency karena
arsitektur dan parameter keduanya identik.

## Training dan inferensi

Setiap batch E2/E3 menjalankan:

```text
grid 8x8 -> endpoint reduction 4  -> fine branch CE        -> update 1
grid 4x4 -> endpoint reduction 16 -> medium branch CE      -> update 2
grid 2x2 -> endpoint reduction 32 -> coarse branch CE      -> update 3
full image -> concat classifier CE x 2                     -> update 4
```

Pada E3, tiga update branch memakai batch asli dan positive batch sekelas.
Consistency MSE diterapkan pada descriptor endpoint yang berpasangan. Update
concat tetap memakai gambar asli. Pada validation/test tidak ada jigsaw dan
tidak diperlukan pasangan; prediksi adalah:

```text
logits = fine + medium + coarse + concat
```

## Gate dan urutan evaluasi

Validation harus dijalankan lebih dahulu. Untuk setiap perbandingan, kandidat
lolos hanya bila:

1. Macro-F1 naik;
2. Hard-F1 naik;
3. Worst-F1 tidak turun lebih dari satu poin.

Perbandingan wajib:

- E2 versus E0: nilai progressive learning terhadap GAP;
- E3 versus E0: nilai metode lengkap terhadap GAP;
- E3 versus E2: kontribusi category consistency;
- E2/E3 versus E1: apakah kandidat melampaui HBP.

Test tidak dibuka untuk tuning grid, dimensi branch, atau bobot consistency.

## Menjalankan

```bash
python -u -m bilinear_lmmd.experiments.run_progressive_multigranularity \
  --data-root data/coffee17-clean/folds/fold_1 \
  --baseline-root outputs/backbone-results \
  --output-root outputs/progressive-efficientnet \
  --seeds 123 \
  --models E2 E3 \
  --evaluation-split val
```

Runner membutuhkan `best.pt` E0/E1 pada `baseline-root`, melatih E2/E3 secara
end-to-end, mendukung resume, menampilkan progress bar, mengevaluasi seluruh
model, dan menyimpan `progressive_decision.json`.

Setelah screening menolak E3, konfirmasi tidak boleh menjalankan E3 kembali:

```bash
python -u -m bilinear_lmmd.experiments.run_progressive_multigranularity \
  --data-root data/coffee17-clean/folds/fold_1 \
  --baseline-root outputs/backbone-results \
  --output-root outputs/progressive-efficientnet \
  --seeds 42 123 2026 \
  --models E2 \
  --evaluation-split val
```

## Keterbatasan

- Empat optimizer update per batch membuat biaya training tidak setara GAP/HBP.
- Positive-pair sampling menggunakan label train dan sah untuk supervised FGVC,
  tetapi tidak boleh mengambil validation/test.
- Coffee17 kecil; hasil satu seed adalah screening dan belum cukup untuk klaim.
- Jika E3 tidak mengalahkan E2, category consistency ditolak walaupun keduanya
  mungkin mengalahkan GAP.
