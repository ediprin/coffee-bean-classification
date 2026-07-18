# Protokol Spatially Preserved HBP

## Batas klaim

HBP dan interaksi pasangan lintas-layer mengikuti kerangka Yu et al. (ECCV
2018). Penyelarasan ke grid tengah 14 x 14 adalah modifikasi penelitian ini,
bukan komponen yang diklaim berasal langsung dari paper tersebut.

Paper HBP memakai aktivasi `relu5_1`, `relu5_2`, dan `relu5_3` dari VGG-16 yang
berada pada tahap spasial yang sama. Implementasi MobileNetV3 dalam repositori
ini memakai endpoint dengan reduction 4, 16, dan 32. Pada input 224 x 224,
ukurannya adalah 56 x 56, 14 x 14, dan 7 x 7. Baseline M1 mengecilkan semuanya
ke ukuran terdalam 7 x 7 sebelum interaksi. M1s menguji apakah mempertahankan
grid 14 x 14 membantu cacat lokal tanpa menambah parameter.

## Ablasi terkendali

| Kode | Head | Grid interaksi | Loss |
|---|---|---:|---|
| M1 | HBP | 7 x 7 | CE |
| M1s | SP-HBP | 14 x 14 | CE |

Backbone, pretrained weights, `out_indices`, dimensi proyeksi, classifier,
dropout, augmentasi, optimizer, split, dan seed harus sama. Kedua model memiliki
jumlah parameter serta embedding 1536-D yang sama. Perubahan biaya komputasi
tetap dilaporkan karena M1s melakukan produk pada grid empat kali lebih besar.

## Tahapan keputusan

1. Screening dijalankan pada validation dengan protokol yang ditetapkan sebelum
   melihat test.
2. Kandidat diteruskan hanya jika Macro-F1 dan Hard-F1 validation meningkat dan
   Worst-F1 tidak turun lebih dari satu poin.
3. Setelah lolos, jalankan seed 42, 123, dan 2026 pada split identik, lalu
   evaluasi test sekali untuk konfirmasi.
4. Klaim keberhasilan memerlukan mean Macro-F1 dan Hard-F1 lebih tinggi, delta
   Macro-F1 positif minimal dua dari tiga seed, serta pelaporan Worst-F1 dan
   latency tanpa disembunyikan.

Attention belum digabungkan pada tahap ini. Jika M1s berhasil, attention spasial
residual dapat diuji sebagai ablasi berikutnya agar efek preservasi resolusi dan
attention tetap dapat dipisahkan.
