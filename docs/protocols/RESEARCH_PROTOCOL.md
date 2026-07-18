# Protokol dataset dan eksperimen

## Sumber yang diverifikasi

- Dataset: [Coffee Green Bean with 17 Defects
  (original)](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original)
- Paper: S. Arwatchananukul et al., *Implementing a deep learning model for
  defect classification in Thai Arabica green coffee beans*, Smart
  Agricultural Technology 9 (2024) 100680,
  [DOI 10.1016/j.atech.2024.100680](https://doi.org/10.1016/j.atech.2024.100680).
- Salinan lokal: `docs/Arwatchananukul et al. - 2024 - Implementing a deep
  learning model for defect classification in Thai Arabica green coffee
  beans.pdf`.

## Fakta baseline dari paper

- Dataset asli: 979 gambar JPEG, masing-masing 500×500 piksel.
- Kondisi akuisisi: latar putih dan pencahayaan seragam.
- Kelas: Broken, Cut, Dry Cherry, Fade, Floater, Full Black, Full Sour, Fungus
  Damage, Husk, Immature, Parchment, Partial Black, Partial Sour, Severe Insect
  Damage, Shell, Slight Insect Damage, dan Withered.
- Arsip Kaggle salah mengeja dua folder sebagai `Fungus Damange` dan
  `Severe Insect Damange`; script menormalkannya menjadi `Damage` agar sesuai
  dengan paper.
- Augmentasi: rotasi 45°, 90°, 135°, 180°, 225°, dan 270°, menghasilkan 5.874
  citra tambahan dan total 6.853 citra.
- Split yang dilaporkan: train 70%, validation 20%, test 10%.
- Backbone yang dibandingkan: MobileNetV2, MobileNetV3, EfficientNetV2,
  InceptionV2, dan ResNetV2 dengan bobot ImageNet.
- MobileNetV3 memperoleh validation accuracy 90,19% pada perbandingan model.
- Setelah tuning dan cross-validation, evaluasi test yang dilaporkan adalah
  accuracy 88,63% dan macro F1 89,04% pada 686 sampel.

## Perubahan yang disengaja dalam repositori ini

### 1. Split sebelum augmentasi

Script persiapan membagi 979 gambar asli terlebih dahulu. Rotasi dilakukan
online hanya untuk train. Ini menjaga independensi gambar asli antar-split.
Angka hasil tidak boleh diklaim sebagai reproduksi identik paper karena detail
urutan augmentasi dan split paper tidak sepenuhnya cukup untuk merekonstruksi
identitas setiap sampel.

### 2. Macro-F1 sebagai metrik utama

Jumlah sampel per kelas tidak seimbang (35-78 gambar asli per kelas), sehingga
macro-F1 target dipakai bersama accuracy. Confusion matrix dan F1 per kelas
perlu dilaporkan, khususnya untuk Partial Sour, Slight Insect Damage, dan kelas
lain yang sering tertukar.

### 3. Kaggle sebagai source, bukan otomatis sebagai target

Dataset Kaggle hanya membentuk satu domain terkontrol. Split train/test dari
dataset yang sama bukan domain adaptation. Eksperimen LMMD yang sah memerlukan
target dengan:

- 17 nama kelas yang sama;
- kamera, pencahayaan, latar, lokasi, atau lot yang berbeda;
- label target train tidak digunakan saat optimisasi;
- label target validation/test hanya digunakan untuk evaluasi.

Jika target nyata belum tersedia, eksperimen dapat dimulai dari B0-B4 dan
M0-M1. Domain sintetis boleh dipakai untuk debugging kode, tetapi harus diberi
label sebagai simulasi dan bukan bukti utama efektivitas UDA.

## Matriks minimum

1. B0-B4: pemilihan backbone pada protokol identik.
2. M0 vs M1: kontribusi HBP.
3. M2 vs M3: global MMD dibanding class-wise LMMD.
4. M4 vs M5: DANN dibanding LMMD pada HBP.
5. Tiga atau lebih seed; laporkan mean dan standard deviation.
6. Parameter, ukuran model, latency batch-1, dan peak memory pada perangkat yang
   sama.
