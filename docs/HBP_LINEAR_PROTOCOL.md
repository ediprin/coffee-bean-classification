# Audit dan Protokol HBP Linear

## Mengapa varian baru diperlukan

`head: hbp` lama tetap dipertahankan agar seluruh checkpoint dan hasil yang
sudah diperoleh tetap dapat direproduksi. Head tersebut memakai proyeksi
`Conv1x1 -> BatchNorm -> ReLU`. Operasi pasangan, pooling, signed square-root,
normalisasi L2, dan concatenation-nya benar, tetapi ReLU membuat proyeksi dan
interaksi tidak lagi bertanda.

Yu et al. (ECCV 2018) merumuskan proyeksi linear sebelum Hadamard product.
Karena itu ditambahkan head terpisah:

```text
head: hbp_linear
feature_i -> Conv1x1 linear -> align grid
          -> pairwise Hadamard -> spatial mean
          -> signed sqrt -> L2 per pair -> concat
```

Bias proyeksi dapat dipelajari tetapi diinisialisasi nol, mengikuti implementasi
Caffe yang dirilis penulis. Dimensi tetap `3 x 512 = 1536`. Nama yang benar
adalah **Yu-style linear-projection, low-dimensional cross-stage HBP**; ini
belum reproduksi literal karena feature map backbone modern berasal dari stage
dengan resolusi berbeda dan masih disejajarkan ke grid terdalam.

## Kontrol eksperimen pertama

Eksperimen pertama hanya mengganti proyeksi. `out_indices`, backbone,
pretraining, input, CE, augmentasi, optimizer, scheduler, dan seed tetap sama
dengan `head: hbp` lama. Kode baru tidak pernah memakai direktori checkpoint
lama:

| Keluarga | HBP lama | HBP linear |
|---|---|---|
| MobileNetV3 | M1 | M1L |
| MobileNetV4 | BV4H | BV4L |
| EfficientNetV2 | BE2H | BE2L |
| ConvNeXtV2 | BC2H | BC2L |
| PVTv2 | BP2H | BP2L |
| SHViT | BSHH | BSHL |

Bias nol dipasang tanpa mengambil bilangan acak tambahan. Dengan seed yang sama,
HBP lama dan HBP linear memperoleh bobot awal backbone, proyeksi, classifier,
serta state RNG sebelum iterasi DataLoader yang identik. Hal ini dikunci dengan
unit test agar ablation proyeksi benar-benar mengubah satu faktor saja.

## Screening minimal

Jangan mengulang semua model dahulu. EfficientNetV2 dipilih karena HBP lama
hampir menyamai GAP, sedangkan PVTv2 dipilih sebagai stress test karena HBP lama
kolaps. Jalankan hanya validation seed 123:

```bash
python -u -m bilinear_lmmd.run_backbone_screening \
  --data-root /content/coffee17-resumable-v1/clean/folds/fold_1 \
  --output-root /content/hbp-linear-results \
  --backbones EV2 PV2 \
  --heads hbp hbp_linear \
  --seeds 123 \
  --evaluation-split val \
  --hf-repo NAMA_USER/coffee-backbone-checkpoints \
  --hf-sync-every 2
```

Runner akan memulihkan BE2H/BP2H lama bila tersedia dan hanya melatih BE2L serta
BP2L. Jangan membuka test.

## Aturan keputusan

1. Jika `hbp_linear` tidak memperbaiki EV2 maupun PVTv2, hentikan cabang ini.
2. Jika EV2 membaik tetapi PVTv2 tetap kolaps, simpulkan bahwa proyeksi linear
   membantu CNN tetapi adapter cross-stage belum cocok untuk Transformer.
3. Jika PVTv2 pulih jelas, lanjutkan satu seed pada MobileNetV3 sebagai anchor.
4. Multi-seed hanya untuk kandidat yang menang Macro-F1 tanpa menjatuhkan
   Hard-F1/Worst-F1 secara material.

SHViT tidak dipakai pada screening minimal karena feature grid-nya berakhir pada
sekitar 4x4 untuk input 224. Perbandingan HBP kanonik pada semua keluarga
memerlukan adapter hook tiga layer akhir dengan resolusi sama dan merupakan
eksperimen terpisah.

## Verifikasi kode

Tes mencakup oracle numerik manual, pelestarian interaksi negatif, larangan
BatchNorm/ReLU pada proyeksi baru, kontrol inisialisasi same-seed, serta forward
seluruh backbone. Tes bentuk tensor saja tidak dianggap bukti rumus.
