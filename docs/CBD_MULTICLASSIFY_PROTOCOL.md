# Benchmark Roboflow CBD-Multiclassify

## Tujuan

Menguji apakah efek HBP terhadap MobileNetV3 juga muncul pada dataset publik
green coffee bean lain yang labelnya lebih kasar. Dataset sumber adalah proyek
klasifikasi [`asdasd-zsar1/cbd-multiclassify`](https://universe.roboflow.com/asdasd-zsar1/cbd-multiclassify)
di Roboflow Universe, yang pada halaman indeks dilaporkan berisi sekitar 4,01
ribu gambar.

Eksperimen ini adalah benchmark terpisah. Gambar tidak dicampurkan ke Coffee-17
karena ontologi labelnya berbeda.

## Ontologi

Sepuluh kelas berlabel yang dipakai:

- Black
- Broken
- Cherry
- Damage
- Dried
- Floater
- Fungus
- Good
- Insect
- Sour

Folder `Unlabeled` dikeluarkan sebelum split dan jumlahnya dicatat di
`audit.json`. Nama label yang lebih kasar tidak dipetakan paksa ke `Full Sour`,
`Partial Sour`, `Slight Insect Damage`, atau `Severe Insect Damage` Coffee-17.

## Audit dan split

Preparer melakukan langkah berikut:

1. menemukan struktur ekspor klasifikasi Roboflow `train/valid/test/Kelas`;
2. mengeluarkan `Unlabeled`;
3. membuang exact duplicate berlabel sama dan mengarantina konflik label;
4. mengambil identitas gambar dari nama sebelum suffix `.rf.<hash>`;
5. membuat ulang split 60/20/20 secara class-stratified dan identity-grouped.

Split arsip tidak dipertahankan karena variant Roboflow dari gambar asal yang
sama dapat tersebar lintas split. Audit melaporkan overlap identitas pada arsip
dan memastikan overlap pada split baru bernilai nol. Aturan nama hanya menangkap
format `.rf.<hash>`; audit manual gallery tetap dianjurkan untuk mendeteksi
near-duplicate yang namanya telah berubah total.

## Model

| Kode | Model |
|---|---|
| CBD0 | MobileNetV3-Large + GAP + CE |
| CBD1 | MobileNetV3-Large + HBP + CE |

Keduanya memakai input 224 dan konfigurasi training yang sama. `Defect-F1`
adalah macro-F1 sembilan kelas selain `Good`; Macro-F1 dan Worst-class F1 tetap
menjadi metrik utama.

## Protokol keputusan

1. Screening validation seed 42.
2. Jika CBD1 tidak menaikkan Macro-F1 dan Worst-class F1, HBP tidak dilanjutkan
   pada dataset ini.
3. Jika lolos, jalankan seed 42, 123, dan 2026 pada test.
4. Klaim akhir harus memuat mean, sample standard deviation delta berpasangan,
   dan jumlah seed yang membaik.

Hasil ini menguji generalitas efek HBP pada dataset publik kedua, tetapi bukan
external validation model Coffee-17 karena jumlah dan makna kelas berbeda.
Lisensi versi Roboflow yang diunduh harus diverifikasi dan dicatat sebelum
dataset dipakai dalam publikasi.

## Runner

```bash
python -u -m bilinear_lmmd.run_cbd_multiclassify_screening \
  --raw-root /kaggle/input/PATH_CBD_MULTICLASSIFY \
  --data-root /kaggle/working/cbd-multiclassify-prepared \
  --output-root /kaggle/working/cbd-hbp-results \
  --seeds 42 \
  --evaluation-split val
```

Perintah dapat dijalankan ulang. Training yang memiliki `history.json` lengkap
dan `best.pt` akan dilewati.
