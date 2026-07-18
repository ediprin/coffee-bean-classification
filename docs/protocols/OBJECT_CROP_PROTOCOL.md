# Protokol object-centric crop Coffee-17

## Pertanyaan penelitian

Apakah normalisasi posisi dan skala biji melalui object-centric crop memperbaiki
MobileNetV3-GAP dan MobileNetV3-HBP pada klasifikasi 17 cacat green coffee
bean?

Eksperimen ini mengambil prinsip preprocessing Jiao et al. (2025), bukan
arsitektur Swin-HSSAM. Mask hanya dipakai untuk menemukan bounding box dan
tidak menjadi input CNN.

## Preprocessing

Untuk setiap gambar RGB:

1. resize sementara sisi terpanjang menjadi 256 piksel untuk segmentasi cepat;
2. estimasi warna background dari border gambar dalam ruang Lab;
3. Otsu threshold pada jarak warna terhadap background;
4. opening, closing, largest connected component, dan hole filling;
5. proyeksikan bounding box kembali ke resolusi asli;
6. tambahkan margin 10% dari sisi bounding box terpanjang;
7. ambil crop dari gambar RGB asli;
8. pad menjadi persegi memakai median warna border;
9. jalankan augmentasi/resize/normalisasi yang sama dengan baseline.

Mask tidak disimpan, tidak diberikan ke classifier, dan tidak memerlukan
anotasi manual. Segmentasi yang gagal menghentikan eksperimen agar sampel yang
bermasalah tidak disembunyikan oleh fallback.

## Factorial ablation

| Kode | Input | Head | Tujuan |
|---|---|---|---|
| M0 | gambar asli | GAP | baseline pooling biasa |
| M1 | gambar asli | HBP | baseline final HBP |
| O0 | object-centric crop | GAP | efek crop pada GAP |
| O1 | object-centric crop | HBP | efek crop pada HBP |

Kontras yang dilaporkan:

- `O0 - M0`: pengaruh crop pada GAP;
- `O1 - M1`: pengaruh crop pada HBP;
- `O1 - O0`: pengaruh HBP setelah background dikendalikan;
- `M1 - M0`: pengaruh HBP pada gambar asli.

Semua pasangan memakai split, seed, backbone, optimizer, augmentasi, dan jumlah
epoch yang sama. Parameter model O0 sama dengan M0; parameter O1 sama dengan
M1 karena preprocessing tidak menambah parameter inferensi CNN.

## Tahapan keputusan

Screening pertama memakai validation split dan satu seed yang telah ditentukan:

```bash
python -u -m bilinear_lmmd.experiments.run_finegrained_screening \
  --data-root DATA_ROOT_SATU_FOLD \
  --output-root /kaggle/working/object-crop-results \
  --stage object_crop \
  --seeds 123 \
  --evaluation-split val
```

Lanjutkan ke tiga seed hanya jika O1:

- tidak menurunkan Macro-F1 terhadap M1;
- meningkatkan Hard-F1;
- tidak menurunkan Worst-F1;
- tidak menunjukkan kegagalan segmentasi.

Konfirmasi tiga seed:

```bash
python -u -m bilinear_lmmd.experiments.run_finegrained_screening \
  --data-root DATA_ROOT_SATU_FOLD \
  --output-root /kaggle/working/object-crop-results \
  --stage object_crop \
  --seeds 42 123 2026 \
  --evaluation-split val
```

Test atau grouped cross-validation hanya dijalankan setelah keputusan dan
hyperparameter margin dikunci. Margin 10% tidak boleh dituning memakai test.

## Interpretasi

- `O1 > M1`: normalisasi objek membantu HBP; background/skala sebelumnya
  mungkin mengganggu.
- `O1 ~= M1`: background bukan bottleneck penting; leakage CAM kemungkinan
  berasal dari kontur dan resolusi feature map.
- `O1 < M1`: crop membuang konteks tepi atau memperbesar artefak; pertahankan
  M1 gambar asli.
- `O0` dan `O1` sama-sama naik: manfaat terutama berasal dari preprocessing,
  bukan interaksi khusus dengan HBP.

## Inference

Checkpoint menyimpan konfigurasi `object_crop`, sehingga evaluasi menggunakan
preprocessing yang sama. Pada aplikasi, pengguna tetap memasukkan gambar
mentah:

```text
raw RGB -> auto-segmentation -> RGB crop + margin -> CNN -> kelas
```

Untuk conveyor multi-biji, bounding box detector menggantikan segmentasi
background putih; classifier tetap menerima crop RGB, bukan mask.
