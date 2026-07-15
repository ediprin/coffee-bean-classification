# Protokol SPPF-Attention sebelum HBP

## Dasar metode dan batas klaim

Hong et al. (2026) menempatkan SPPF-Attention pada feature map tingkat tinggi
YOLOv10. Modulnya menggunakan tiga max-pooling `5 x 5`, stride 1, padding 2
secara berurutan; concatenation multi-skala; channel attention berbasis GAP dan
MLP; spatial attention berbasis average/max channel pooling dan convolution
`7 x 7`; lalu residual connection.

S1 mengadaptasi modul tersebut untuk klasifikasi, bukan mengklaim
mereproduksi detector Hong. Hanya feature map terdalam MobileNetV3 yang
diperbaiki sebelum interaksi HBP. Feature map dangkal/menengah, formulasi HBP,
dan classifier 17 kelas tetap sama.

## Ablasi terkendali

| Kode | Jalur model | Loss |
|---|---|---|
| M1 | MobileNetV3 -> HBP | CE |
| S1 | MobileNetV3 -> SPPF-Attention pada deep feature -> HBP | CE |

Backbone, pretrained weights, endpoint, dimensi HBP, classifier, dropout,
augmentasi, optimizer, split, epoch, dan seed sama. Inisialisasi modul baru
tidak menggeser RNG sehingga encoder, HBP, classifier, dan urutan batch M1/S1
tetap sebanding pada seed yang sama. S1 memiliki parameter dan FLOPs tambahan;
jika lolos, capacity-matched control dan latency harus diuji sebelum klaim
bahwa peningkatan berasal dari desain attention.

## Keputusan yang dikunci

1. Screening memakai validation seed 42.
2. Lolos bila Macro-F1 dan Hard-F1 meningkat serta Worst-F1 tidak turun lebih
   dari satu poin.
3. Jika gagal, S1 dihentikan tanpa tuning reduction atau bobot loss.
4. Jika lolos, konfirmasi seed 42, 123, dan 2026 dilakukan sebelum test dibuka.
5. Test hanya dievaluasi sekali setelah kandidat dan capacity control dikunci.
