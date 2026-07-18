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
| C1 | MobileNetV3 -> capacity-matched pointwise residual -> HBP | CE |
| S0 | MobileNetV3 -> SPPF-Attention pada deep feature -> GAP | CE |
| S1 | MobileNetV3 -> SPPF-Attention pada deep feature -> HBP | CE |

Backbone, pretrained weights, endpoint, dimensi HBP, classifier, dropout,
augmentasi, optimizer, split, epoch, dan seed sama. Inisialisasi modul baru
tidak menggeser RNG sehingga encoder, HBP, classifier, dan urutan batch M1/S1
tetap sebanding pada seed yang sama. S1 memiliki parameter dan FLOPs tambahan;
jika lolos, capacity-matched control dan latency harus diuji sebelum klaim
bahwa peningkatan berasal dari desain attention.

C1 memakai dua pointwise convolution dengan hidden dimension 1259 dan residual
connection. Ia tidak memiliki spatial pooling, channel attention, atau spatial
attention. Pada MobileNetV3-Large, C1 memiliki 5,984,023 parameter dan S1
5,984,483 parameter (selisih 460 atau kurang dari 0.01%). Pointwise FLOPs pada
grid deep `7 x 7` juga sebanding dengan dua proyeksi utama S1.

S0 melengkapi ablasi faktorial `2 x 2`: M0/M1 mengisolasi efek HBP tanpa SPPF,
M0/S0 mengisolasi efek SPPF sebelum GAP, M1/S1 mengisolasi efek SPPF sebelum
HBP, dan S0/S1 mengisolasi efek HBP ketika SPPF aktif. S0 hanya memakai endpoint
terdalam yang sama dengan M0; inisialisasi backbone dan classifier dijaga identik
pada seed yang sama.

## Keputusan yang dikunci

1. Screening memakai validation seed 42.
2. Lolos bila Macro-F1 dan Hard-F1 meningkat serta Worst-F1 tidak turun lebih
   dari satu poin.
3. Jika S1 lolos terhadap M1, jalankan C1 pada seed yang sama. Efek spesifik
   SPPF-Attention hanya diterima bila S1 mengungguli C1 pada Macro-F1 dan
   Hard-F1, sementara Worst-F1 tidak turun lebih dari satu poin.
4. Jika gagal, S1 dihentikan tanpa tuning reduction atau bobot loss.
5. Jika lolos, konfirmasi seed 42, 123, dan 2026 dilakukan sebelum test dibuka.
6. Test hanya dievaluasi sekali setelah kandidat dan capacity control dikunci.

## Hasil dan keputusan final

Pada agregasi validation tiga seed, S1 terhadap M1 mengubah Macro-F1 -0,14
poin, Hard-F1 +0,71 poin, dan Worst-F1 +4,24 poin. Ablasi S0 menunjukkan bahwa
SPPF-Attention tanpa HBP juga menaikkan Worst-F1 validation, tetapi menurunkan
Macro-F1 dan Hard-F1.

Keuntungan validation tersebut tidak bertahan pada test. Dibanding M1, S1
menurunkan Macro-F1 1,53 poin, Hard-F1 0,72 poin, dan Worst-F1 6,20 poin.
S1 karena itu ditolak dan tidak dituning ulang menggunakan test. Angka lengkap
dan keputusan model dicatat di [FINAL_HBP_RESULTS.md](../results/FINAL_HBP_RESULTS.md).
