# SHViT Backbone Screening Note

## Status

SHViT dicatat sebagai kandidat ablation backbone, bukan upgrade yang sudah
terbukti dan bukan bagian novelty model saat ini. Eksperimen ini baru boleh
dijalankan setelah konfirmasi granularity selesai.

## Dasar pertimbangan

SHViT (Yun dan Ro, CVPR 2024) memakai macro design tiga stage, konvolusi pada
stage awal, dan single-head self-attention parsial pada stage akhir. Implementasi
`timm>=1.0.20` menyediakan pretrained `shvit_s1` sampai `shvit_s4` dan mendukung
`features_only=True`.

Pada `shvit_s1`, tiga output stage memiliki channel `[128, 224, 320]` dan
reduction `[16, 32, 64]`. Secara struktural ini kompatibel dengan HBP melalui
`out_indices: [0, 1, 2]`. Namun penyelarasan ke feature map terdalam membuat
interaksi HBP berlangsung pada grid sekitar 4 x 4 untuk input 224. MobileNetV3
yang sekarang dipakai menghasilkan reduction terdalam 32 atau grid sekitar
7 x 7. Grid SHViT yang lebih kasar berpotensi menghilangkan cacat lokal kecil.

Hasil ImageNet-1K resmi tidak memberikan alasan kuat bahwa SHViT-S1 otomatis
mengungguli MobileNetV3-Large:

| Backbone | Params | FLOPs | Top-1 |
|---|---:|---:|---:|
| MobileNetV3-Large | 5,4 M | 217 M | 75,2% |
| SHViT-S1 | 6,3 M | 241 M | 72,8% |
| SHViT-S2 | 11,4 M | 366 M | 75,2% |
| SHViT-S3 | 14,2 M | 601 M | 77,4% |

SHViT-S1 lebih tepat diperlakukan sebagai kandidat efisiensi. SHViT-S3 memberi
prior akurasi yang lebih baik, tetapi jauh lebih berat dan lebih berisiko
overfit pada Coffee-17 yang kecil.

## Screening yang diperbolehkan

Gunakan validation seed 123 terlebih dahulu:

| Kode konseptual | Model |
|---|---|
| SH0 | SHViT-S1 + GAP + CE |
| SH1 | SHViT-S1 + low-dimensional HBP + CE |

Bandingkan SH0/SH1 dengan M0/M1 pada split, augmentasi, optimizer, dan seed
identik. Laporkan Macro-F1, Worst-F1, parameter, latency batch-1, dan throughput.
Jangan menjalankan multi-seed jika SHViT gagal pada screening awal. Perbandingan
lintas-backbone tidak boleh disebut efek arsitektur murni karena jumlah parameter
berbeda.

## Framing metode

HBP yang digunakan adalah hierarchical factorized bilinear pooling:

```text
project per layer -> pairwise Hadamard product -> spatial mean
-> signed square-root -> L2 normalization -> concatenate three pairs
```

Dimensi proyeksi bukan otomatis rank aljabar efektif. Low-rank/factorized
bilinear pooling sudah ada sebelum penelitian ini dan tidak diklaim sebagai
novelty baru. Potensi kontribusi penelitian berada pada adaptasi ringan,
protokol evaluasi terkontrol, dan analisis batas manfaat berdasarkan
granularitas.

## Referensi primer

- Yun, S. dan Ro, Y. (2024), *SHViT: Single-Head Vision Transformer with
  Memory Efficient Macro Design*, CVPR.
- Yu, C. et al. (2018), *Hierarchical Bilinear Pooling for Fine-Grained Visual
  Recognition*, ECCV.
- Kim, J.-H. et al. (2017), *Hadamard Product for Low-Rank Bilinear Pooling*,
  ICLR.
