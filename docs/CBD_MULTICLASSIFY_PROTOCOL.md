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

## Konfirmasi logistic stacking

Eksperimen seed 42 menunjukkan bahwa concatenated log-probability GAP/HBP yang
diproses `StandardScaler` dan `LogisticRegression(C=1)` mengungguli GAP mentah
serta kontrol kalibrasi single-model. Pipeline stacking kemudian dikunci tanpa
tuning tambahan dan dikonfirmasi pada seed 42, 123, dan 2026.

Untuk setiap seed, meta-model dilatih pada prediction validation dan dievaluasi
sekali pada prediction test. Kontrol `GAP_CAL` dan `HBP_CAL` memakai meta-model
yang sama tetapi hanya menerima log-probability satu model. Dengan demikian,
keuntungan stacking di luar kontrol terbaik dapat diatribusikan pada fusion,
bukan sekadar kalibrasi kelas.

Kriteria PASS yang dikunci:

1. mean Macro-F1 stacking minimal 0,3 poin di atas calibrated control terbaik;
2. mean Worst-F1 tidak lebih rendah dari calibrated control terbaik;
3. Macro-F1 stacking mengungguli masing-masing kontrol pada minimal 2/3 seed.

```bash
python -u -m bilinear_lmmd.run_cbd_stacking_confirmation \
  --data-root /kaggle/working/cbd-multiclassify-prepared \
  --output-root /kaggle/working/cbd-hbp-results \
  --seeds 42 123 2026
```

Hasil tetap merupakan benchmark sekunder satu split dataset CBD, bukan bukti
bahwa stacking atau HBP unggul universal.

### Hasil konfirmasi stacking

Konfirmasi test pada seed 42, 123, dan 2026 menghasilkan:

| Model | Macro-F1 | Worst-F1 |
|---|---:|---:|
| GAP raw | 90,66% +/- 1,01% | 76,31% +/- 0,76% |
| HBP raw | 90,44% +/- 0,26% | 75,47% +/- 1,46% |
| GAP calibrated | 91,46% +/- 0,26% | 75,54% +/- 1,27% |
| HBP calibrated | 90,85% +/- 0,77% | 77,03% +/- 0,30% |
| GAP-HBP stacking | **92,32% +/- 0,21%** | **77,37% +/- 2,89%** |

Fusion margin terhadap calibrated control terbaik adalah +0,86 poin
Macro-F1 dan +0,34 poin Worst-F1, sehingga keputusan pra-registrasi adalah
**PASS**. Secara teknis ini adalah *holdout probability stacking/blending*:
meta-model dilatih dari validation prediction, bukan dari prediction
out-of-fold seluruh training set. Worst-F1 stacking juga lebih bervariasi dan
turun pada seed 2026; variasi ini tetap dilaporkan.

## Knowledge distillation stacking ke satu CNN

Karena stacking membutuhkan GAP CNN dan HBP CNN saat inference, tahap berikutnya
menguji apakah soft target stacking dapat dipindahkan ke satu student
MobileNetV3-GAP. Teacher tetap dibangun per seed dari checkpoint GAP/HBP dan
logistic meta-model yang hanya di-fit pada validation prediction. Teacher dan
student menerima augmentasi online yang sama pada training image. Test tidak
dipakai untuk fitting atau pemilihan checkpoint.

Konfigurasi dikunci sebelum hasil KD dilihat:

- student: MobileNetV3-Large + GAP;
- epoch dan augmentasi: sama dengan CBD0;
- temperature: 2;
- loss: `0.5 * CE + 0.5 * T^2 * KL`;
- `GAP_KD_CONTROL`: distilasi dari GAP calibrated untuk mengontrol efek
  kalibrasi;
- `STACKING_KD`: distilasi dari GAP-HBP stacking untuk menguji transfer fusion;
- inference student hanya memakai satu GAP CNN; kedua teacher CNN tidak dibawa
  ke deployment.

Kriteria konfirmasi tiga seed:

1. mean Macro-F1 `STACKING_KD` minimal +0,3 poin atas GAP raw;
2. mean Worst-F1 tidak lebih rendah dari GAP raw;
3. mean Macro-F1 `STACKING_KD` lebih tinggi dari `GAP_KD_CONTROL`;
4. Macro-F1 mengungguli GAP raw pada minimal 2/3 seed.

Screening seed 42 dijalankan dahulu untuk menghemat GPU:

```bash
python -u -m bilinear_lmmd.run_cbd_kd_confirmation \
  --data-root /kaggle/working/cbd-multiclassify-prepared \
  --output-root /kaggle/working/cbd-hbp-results \
  --seeds 42
```

Jika screening menjanjikan, konfirmasi dilanjutkan memakai perintah yang sama
dengan `--seeds 42 123 2026`. Run seed 42 yang lengkap akan dilewati.
