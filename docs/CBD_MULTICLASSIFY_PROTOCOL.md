# Benchmark Roboflow CBD-Multiclassify

## Tujuan

Menguji apakah efek HBP terhadap MobileNetV3 juga muncul pada dataset publik
green coffee bean lain yang labelnya lebih kasar. Dataset sumber adalah proyek
klasifikasi [`asdasd-zsar1/cbd-multiclassify`](https://universe.roboflow.com/asdasd-zsar1/cbd-multiclassify)
di Roboflow Universe. Ekspor version 1 yang diaudit berisi 4.015 gambar.

Eksperimen ini adalah benchmark terpisah. Gambar tidak dicampurkan ke Coffee-17
karena ontologi labelnya berbeda.

## Ontologi

Delapan kelas berlabel yang dipakai:

- Black
- Broken
- Dried Cherry
- Floater
- Fungus Damage
- Good
- Insect Damage
- Sour

Nama folder arsip dinormalisasi dari `Cherry Dried`, `Damage Fungus`, dan
`Damage Insect` menjadi nama di atas. Dua gambar `Unlabeled` dikeluarkan sebelum
split. Satu gambar pada folder terpisah `Insect` juga dikeluarkan karena label
ini inkonsisten dengan `Damage Insect` yang memiliki 455 gambar dan n=1 tidak
memungkinkan split/evaluasi kelas yang valid. Semua pengeluaran dicatat di
`audit.json`.

Nama label yang lebih kasar tidak dipetakan paksa ke `Full Sour`, `Partial Sour`,
`Slight Insect Damage`, atau `Severe Insect Damage` Coffee-17.

## Audit dan split

Preparer melakukan langkah berikut:

1. menemukan struktur ekspor klasifikasi Roboflow `train/valid/test/Kelas`;
2. mengeluarkan `Unlabeled` dan stray class `Insect` dengan n=1;
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
| CBD2 | MobileNetV3-Large + GAP + Balanced Softmax |
| CBD3 | MobileNetV3-Large + HBP + Balanced Softmax |

Keduanya memakai input 224, 25 epoch, dan konfigurasi training yang sama.
Budget 25 epoch dipilih untuk screening karena dataset ini jauh lebih besar
daripada Coffee-17. `Defect-F1` adalah macro-F1 tujuh kelas selain `Good`;
Macro-F1 dan Worst-class F1 tetap menjadi metrik utama. CBD2/CBD3 menerapkan
Balanced Softmax dari Ren et al. (NeurIPS 2020): logit training ditambah log
jumlah sampel kelas, sementara logit inference tidak diubah. Natural sampling
tetap dipakai; tidak ada penghapusan kelas mayoritas atau oversampling fisik.

## Protokol keputusan

1. Screening CE validation seed 42 menunjukkan CBD1 tidak mengungguli CBD0.
2. Karena rasio `Good` terhadap `Floater` sekitar 24:1, dilakukan satu follow-up
   faktorial terkontrol dengan Balanced Softmax pada seed/split yang sama.
3. CBD3 harus menaikkan Macro-F1 dan Worst-class F1 terhadap CBD2 agar HBP
   dinyatakan lolos di bawah penanganan imbalance.
4. Jika salah satu tidak naik, eksperimen berhenti tanpa tiga seed test.
5. Jika lolos, jalankan seed 42, 123, dan 2026 pada test.
6. Klaim akhir harus memuat mean, sample standard deviation delta berpasangan,
   dan jumlah seed yang membaik.

Referensi: Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual
Recognition*, NeurIPS 2020,
https://papers.nips.cc/paper_files/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html.

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

Perintah dapat dijalankan ulang. Training CBD0/CBD1 yang sudah lengkap akan
dilewati; setelah update runner hanya CBD2/CBD3 yang dilatih. Training yang
memiliki `history.json` lengkap dan `best.pt` akan dilewati.
