# Protokol Efek HBP pada Coffee Bean Roast Dataset

## Batas klaim

Dataset publik `gpiosenka/coffee-bean-dataset-resized-224-x-224` adalah
klasifikasi empat tingkat roasting (`Dark`, `Green`, `Light`, `Medium`). Dataset
ini digunakan Jiao et al. (2025) sebagai uji generalisasi tambahan dan berbeda
dari dataset utama 10.378 green coffee beans untuk grading/cacat.

Hasil eksperimen ini hanya menjawab apakah HBP membantu klasifikasi roast-level.
Hasilnya tidak boleh diklaim sebagai bukti klasifikasi sembilan cacat green bean.

## Ablasi terkendali

| Kode | Backbone | Head | Input | Loss |
|---|---|---|---:|---|
| R0 | MobileNetV3-Large | GAP | 224 | CE |
| R1 | MobileNetV3-Large | HBP | 224 | CE |

Pretrained weights, augmentasi, optimizer, scheduler, epoch, batch size, split,
dan seed sama. Satu-satunya perubahan substantif adalah GAP menjadi HBP.

## Integritas split

Test bawaan Kaggle dipertahankan. Jika arsip tidak menyediakan validation,
20% bagian train per kelas dipindahkan secara deterministik ke validation.
Exact duplicate diaudit berdasarkan SHA-256. Runner berhenti jika gambar identik
muncul lintas split atau mempunyai label berbeda.

## Tahapan keputusan

1. Screening validation seed 42.
2. R1 lolos bila Macro-F1 dan F1 hard-group `Light/Medium` meningkat, sementara
   Worst-F1 tidak turun lebih dari satu poin.
3. Bila lolos, konfirmasi validation seed 42, 123, dan 2026.
4. Test dibuka setelah keputusan dikunci.
5. Klaim stabil membutuhkan mean Macro-F1 R1 lebih tinggi dan delta positif
   minimal dua dari tiga seed.

Swin-HSSAM melaporkan test Accuracy 96,50% dan F1 96,48% pada dataset publik
ini. Angka tersebut hanya konteks eksternal; kontribusi HBP ditentukan dari
perbandingan terkontrol R0 versus R1 pada split yang sama.
