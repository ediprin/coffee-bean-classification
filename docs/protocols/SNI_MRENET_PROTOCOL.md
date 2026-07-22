# Protokol SNI-MRENet v1

## Putusan metode

Metode final v1 adalah **SNI Ontology-Guided Multi-Resolution Expert
Network (SNI-MRENet)** untuk klasifikasi oracle crop ke satu dari 21 label
canonical. Sistem ini tidak mengeluarkan jumlah nilai cacat atau kelas mutu
SNI. Nilai ekuivalen cacat tetap metadata dan bukan bobot loss.

Metode menyerang tiga sumber kesulitan yang berbeda:

1. detail kecil dapat hilang bila classifier hanya memakai feature map akhir;
2. 21 label mencampur kondisi biji, material, dan ukuran;
3. interaksi orde kedua relevan untuk kondisi permukaan biji, tetapi tidak
   mempunyai alasan yang sama untuk seluruh kelas material dan ukuran.

## Ontologi yang dibekukan

Urutan kelas bersumber tunggal dari
`src/bilinear_lmmd/data/sni_ontology.py` dan harus sama dengan urutan alfabetis
`ImageFolder`:

- `kondisi_biji`: 12 kelas;
- `kulit_kopi`: 3 ukuran;
- `kulit_tanduk`: 3 ukuran;
- `benda_asing`: 3 ukuran.

Dataset bersama tidak mempertahankan identitas tanah, batu, dan ranting.
Ketiganya dipetakan menjadi `tanah_batu_ranting_{besar,kecil,sedang}` karena
dataset Faruq memang menyediakan label gabungan. Model tidak boleh diklaim
membedakan ketiga jenis material tersebut.

## Arsitektur

Backbone adalah `tf_efficientnetv2_b0.in1k`. Empat keluaran stage
`[1, 2, 3, 4]` diproyeksikan ke 128 channel dan difusikan secara top-down.
Embedding global merupakan concatenation GAP dari keempat feature map hasil
fusi.

Router memprediksi empat kelompok ontologi. Setiap expert memprediksi kelas
lokal di kelompoknya. Probabilitas leaf adalah:

```text
P(leaf | x) = P(group | x) * P(leaf | group, x)
```

Keempat distribusi lokal dikalikan dengan probabilitas router dan
di-concatenate menjadi distribusi 21 kelas yang ternormalisasi.

Cabang `kondisi_biji` menggunakan concatenation embedding global dan HBP
lintas tiga feature map resolusi tertinggi. Cabang lainnya hanya memakai
embedding GAP multiresolusi. Implementasi berada di
`src/bilinear_lmmd/modeling/sni_mrenet.py`.

## Ablation yang dibekukan

| Kode | Model | Pertanyaan |
|---|---|---|
| SNIB0 | EfficientNetV2 final-stage GAP + flat CE | baseline |
| SNIB1 | multiresolution fusion + flat CE | efek preservasi/fusi backbone |
| SNIB2 | multiresolution + ontology experts + projected hierarchical GAP | efek ontologi dan kontrol orde pertama |
| SNIB3 | SNI-MRENet, HBP selektif pada expert biji | efek interaksi orde kedua |

SNIB2 dan SNIB3 memiliki jumlah parameter serta dimensi embedding yang sama.
`ProjectedHierarchicalGAP` dan `HierarchicalBilinearPooling` memakai proyeksi
yang identik; satu-satunya perbedaan representasional adalah pooling orde
pertama versus perkalian fitur berpasangan.

Tidak ada attention, metric loss, domain adaptation, atau grading SNI pada
protokol v1. Komponen tersebut tidak boleh ditambahkan selama ablation ini.

## Optimasi komputasi yang dibekukan

Keempat konfigurasi memakai AMP FP16 dan layout `channels_last` pada CUDA,
serta transfer batch non-blocking dari DataLoader yang memakai pinned memory.
Ini adalah optimasi eksekusi, bukan perubahan metode: resolusi 224, batch 32,
50 epoch, optimizer, scheduler, augmentasi, split, dan seluruh parameter model
tetap sama. Proyeksi convolutional boleh memakai autocast, tetapi perkalian
orde kedua, spatial mean, signed-square-root, dan L2 normalization pada HBP
selalu dihitung dalam FP32. Kontrol `ProjectedHierarchicalGAP` memakai kebijakan
presisi FP32 yang sama agar perbandingan SNIB2--SNIB3 tetap adil.

Checkpoint `last.pt` menyimpan state `GradScaler`. Checkpoint FP32 lama tetap
dapat dilanjutkan; bila state scaler belum ada, scaler AMP diinisialisasi baru
tanpa membuang state model, optimizer, scheduler, atau epoch.

## Data dan evaluasi

Gunakan output lengkap dari protokol `SNI_INSTANCE_CROP_PROTOCOL.md`:

- unit split adalah grup foto sumber, bukan crop;
- training dan seleksi checkpoint hanya memakai train/validation;
- test tetap terkunci;
- input classifier adalah oracle crop tanpa mask;
- augmentasi dilakukan online hanya pada train.

Metrik utama adalah Macro-F1. Metrik pendamping adalah balanced accuracy,
worst-class F1, per-class F1, dan hard-group F1 untuk variasi hitam, jumlah
lubang, ukuran kulit, serta ukuran benda asing. Hasil juga harus dilaporkan per
dataset asal untuk mendeteksi shortcut domain.

Screening awal memakai seed 42 secara fail-fast. Stage `backbone` menjalankan
SNIB0/SNIB1, stage `ontology` hanya menjalankan SNIB2 setelah stage pertama
PASS, dan stage `bilinear` hanya menjalankan SNIB3 setelah stage kedua PASS.
Metode final yang lolos baru boleh dikonfirmasi dengan seed 42, 123, dan 2026.
Test dibuka sekali setelah putusan validation final.

## Runner validation-only

```bash
python -u -m bilinear_lmmd.experiments.run_sni_mrenet_screening \
  --data-root /content/sni-instance-crops \
  --output-root /content/sni-mrenet-results \
  --seeds 42 \
  --stage backbone \
  --evaluation-split val
```

Runner memverifikasi urutan 21 kelas, mengaudit capacity match SNIB2/SNIB3,
menampilkan progress epoch dari engine training, mengevaluasi validation, dan
tidak mempunyai opsi untuk membuka test.

Notebook Colab resumable tersedia di
`notebooks/sni_mrenet_failfast_colab.ipynb`. Dataset dipulihkan dari shard
Google Drive yang dibuat notebook preparasi; checkpoint dan report ditulis
langsung ke Drive agar runtime reset tidak menghapus hasil.
