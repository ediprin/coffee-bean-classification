# Rancangan Eksperimen Fine-Grained HBP dan ArcFace

## Tujuan

Menguji apakah detail spasial yang lebih tinggi, interaksi fitur lintas-layer,
dan angular margin membantu klasifikasi langsung 17 cacat biji kopi pada
dataset kecil yang sudah dibersihkan dari exact duplicate.

Eksperimen ini tidak mengklaim sertifikasi SNI. Sebagian label dapat dipetakan
ke SNI 2907:2008, tetapi target utama tetap taksonomi 17 kelas dari dataset.

## Hipotesis

- H1: input 320 meningkatkan Macro-F1 HBP dibanding input 224 karena detail
  bercak, lubang, retakan, dan tekstur lebih terjaga.
- H2: pada input yang sama, HBP meningkatkan Macro-F1 dan Hard-F1 dibanding GAP.
- H3: pada pooling yang sama, ArcFace memperbaiki pemisahan kelas yang mirip.
- H4: HBP + ArcFace tidak boleh meningkatkan rata-rata dengan mengorbankan
  Worst-F1 secara besar atau tidak konsisten.

## Tahapan

### Screening resolusi

| Kode | Model | Input | Loss |
|---|---|---:|---|
| M1 | MobileNetV3 + HBP | 224 | CE + label smoothing |
| F1 | MobileNetV3 + HBP | 320 | CE + label smoothing |

Gunakan satu clean fold dan seed 123. Resolusi 320 diterima jika Macro-F1 atau
Hard-F1 membaik tanpa penurunan Worst-F1 yang material.

Hasil screening clean fold 1 seed 123:

| Metrik | HBP 224 | HBP 320 | Delta |
|---|---:|---:|---:|
| Macro-F1 | 90.57% | 87.08% | -3.49% |
| Hard-F1 | 87.94% | 83.54% | -4.40% |
| Worst-F1 | 72.73% | 66.67% | -6.06% |

Keputusan: H1 ditolak pada screening dan eksperimen utama memakai input 224.

### Ablation 2 x 2 pada 224

| Kode | Pooling | Classifier |
|---|---|---|
| M0 | GAP | linear + CE |
| M1 | HBP | linear + CE |
| A2 | GAP | ArcFace, scale 30, margin 0.3 |
| A3 | HBP | ArcFace, scale 30, margin 0.3 |

Perbandingan utama setelah keputusan resolusi:

- M1 - M0: efek HBP di bawah CE.
- A2 - M0: efek ArcFace di bawah GAP.
- A3 - M1: efek ArcFace di bawah HBP.
- A3 - A2: efek HBP di bawah ArcFace.

### Matrix 2 x 2 pada 320 (tidak dilanjutkan)

| Kode | Pooling | Classifier |
|---|---|---|
| F0 | GAP | linear + CE |
| F1 | HBP | linear + CE |
| F2 | GAP | ArcFace, scale 30, margin 0.3 |
| F3 | HBP | ArcFace, scale 30, margin 0.3 |

Perbandingan yang diinterpretasikan:

- F1 - F0: efek HBP di bawah CE.
- F2 - F0: efek ArcFace di bawah GAP.
- F3 - F1: efek ArcFace di bawah HBP.
- F3 - F2: efek HBP di bawah ArcFace.

Konfigurasi dipertahankan untuk audit, tetapi tidak dijalankan setelah resolusi
320 gagal pada ketiga metrik screening.

ArcFace mengganti linear softmax selama training dan menggunakan scaled cosine
logits saat inferensi. Model tetap melakukan klasifikasi 17 kelas secara
langsung. Tidak ada pembentukan pasangan dan tidak ada risiko pair-level split.

## Evaluasi

Urutan keputusan:

1. Screening semua kandidat dengan seed 123.
2. Pilih maksimal dua kandidat berdasarkan Macro-F1, Hard-F1, dan Worst-F1.
3. Uji kandidat dengan seed 42, 123, dan 2026.
4. Jalankan grouped 5-fold OOF pada 965 gambar unik untuk hasil utama.

Laporkan accuracy, balanced accuracy, Macro-F1, Hard-F1, Worst-F1, F1 per kelas,
confusion matrix, delta berpasangan, dan konsistensi lintas seed. Klaim metode
diterima hanya jika peningkatan tidak bergantung pada satu seed dan tidak hanya
berasal dari accuracy agregat.
