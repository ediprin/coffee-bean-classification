# Protokol USK-Coffee

## Tujuan

USK-Coffee dipakai untuk dua pertanyaan terpisah:

1. apakah kontribusi HBP berlaku pada dataset kopi publik lain; dan
2. apakah pretraining domain kopi kelak membantu klasifikasi Coffee17.

Dokumen ini mengatur pertanyaan pertama. Empat label asli dipertahankan:
`Defect`, `Longberry`, `Peaberry`, dan `Premium`. Label `Defect` tidak dipecah
atau dipetakan ke taksonomi 17 cacat.

## Referensi paper

Febriana et al. (CyberneticsCom 2022) melaporkan 8.000 gambar seimbang,
masing-masing 2.000 per kelas. Gambar diubah ke 256 x 256. Hasil test yang
dilaporkan adalah:

| Model | Accuracy | Precision | Recall | Waktu |
|---|---:|---:|---:|---:|
| ResNet-18 | 81,13% | 84,14% | 81,12% | 5.930 s |
| MobileNetV2 | 81,31% | 81,42% | 81,31% | 1.948 s |

Teks paper menyebut rasio 55:25:20, sedangkan tabel berisi 4.800/1.600/1.600,
atau 60:20:20. Runner mempertahankan split arsip bila tersedia. Jika arsip
tidak memiliki split, importer membuat 60:20:20 mengikuti jumlah pada tabel dan
mencatatnya sebagai split baru; hasilnya tidak boleh disebut reproduksi exact.

## Audit data

Paper menyatakan setiap biji diposisikan dari sisi depan dan belakang sebelum
difoto, tetapi tidak menjelaskan grouping saat split. Karena itu:

- exact duplicate dengan label sama hanya dipertahankan satu;
- exact duplicate lintas label dikeluarkan;
- pasangan filename dengan akhiran `front/back`, `depan/belakang`,
  `side1/side2`, atau `view1/view2` harus berada pada split sama;
- bila konvensi filename berbeda, `audit.json` dan sampel nama file diperiksa
  sebelum klaim utama.

`--allow-pair-leakage` hanya boleh digunakan untuk reproduksi tambahan yang
dilabeli berpotensi bocor, bukan hasil utama tesis.

## Model dan kontrol

| Kode | Model | Head |
|---|---|---|
| U0 | ResNet-18 | GAP |
| U1 | MobileNetV2 | GAP |
| U2 | MobileNetV3-Large | GAP |
| U3 | MobileNetV3-Large | HBP |

Semua konfigurasi memakai input 256, CE, pretrained ImageNet, 25 epoch, split,
augmentasi, dan seed yang sama. Optimizer pipeline repositori dibuat sama agar
perbandingan U0-U3 terkontrol. Karena pipeline ini tidak identik dengan paper,
angka 81,31% hanya merupakan pembanding langsung bila data dan split asli benar-
benar sama; efek HBP ditentukan terutama oleh U2 vs U3.

## Tahapan keputusan

1. Screening validation seed 42 dilakukan untuk U1, U2, dan U3.
2. U3 diteruskan bila Macro-F1 dan Hard-F1 lebih tinggi dari U2, serta
   Worst-F1 tidak turun lebih dari satu poin.
3. Kandidat yang lolos dikonfirmasi pada seed 42, 123, dan 2026.
4. Test dibuka setelah kandidat dikunci. Semua model terpilih dievaluasi pada
   test yang sama.
5. "Mengalahkan paper" secara deskriptif memerlukan test accuracy di atas
   81,31% pada split asli. Klaim kontribusi metode memerlukan U3 mengungguli U2
   pada mean Macro-F1 minimal dua dari tiga seed, disertai Hard-F1, Worst-F1,
   parameter, dan latency.

Pasangan Peaberry-Premium dilaporkan sebagai hard group karena paper menunjukkan
kebingungan visual terbesar pada kedua kelas tersebut.
