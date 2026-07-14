# Hipotesis dan uji kontribusi HBP

## Apa yang benar-benar ditunjukkan paper

Table 12 Arwatchananukul et al. melaporkan test accuracy 88,63% dan macro-F1
89,04% setelah tuning MobileNetV3. Nilai keseluruhan tersebut tidak merata pada
17 kelas:

| Kelas | F1 (%) |
|---|---:|
| Broken | 84,78 |
| Cut | 85,98 |
| Dry Cherry | 100,00 |
| Fade | 86,27 |
| Floater | 90,32 |
| Full Black | 100,00 |
| Full Sour | 88,07 |
| Fungus Damage | 89,66 |
| Husk | 100,00 |
| Immature | 85,37 |
| Parchment | 100,00 |
| Partial Black | 87,85 |
| Partial Sour | 77,78 |
| Severe Insect Damage | 89,13 |
| Shell | 84,09 |
| Slight Insect Damage | 78,38 |
| Withered | 86,08 |

Empat kelas sudah mencapai F1 100%, tetapi Partial Sour dan Slight Insect
Damage berada di bawah 80%. Ini mendukung adanya ekor kelas sulit, bukan klaim
bahwa semua kelas sulit.

## Kelompok kelas sulit yang ditetapkan sebelum training

Kelompok berikut berasal dari confusion matrix dan pembahasan paper, bukan dari
hasil HBP kita:

- Sour/black: Partial Black, Partial Sour, Full Sour.
- Shape/withered: Withered, Immature, Cut.
- Insect damage: Slight Insect Damage, Severe Insect Damage.

Berdasarkan angka Table 12:

| Metrik turunan paper | Nilai (%) |
|---|---:|
| Sour/black mean F1 | 84,57 |
| Shape/withered mean F1 | 85,81 |
| Insect damage mean F1 | 83,76 |
| Gabungan 8 hard classes | 84,83 |
| Worst-class F1 | 77,78 |
| Rata-rata 9 kelas lainnya | 92,78 |

Selisih sekitar 7,95 poin antara hard subset dan sembilan kelas lainnya adalah
alasan yang lebih tepat untuk menguji representasi fine-grained.

### Diagnosis kesalahan dari Table 12

Angka precision, recall, dan support memungkinkan jumlah kesalahan utama
diturunkan:

| Kelas | TP | FN | FP | Diagnosis |
|---|---:|---:|---:|---|
| Partial Sour | 28 | 7 | 9 | precision dan recall sama-sama rendah |
| Slight Insect Damage | 29 | 7 | 9 | menerima banyak false positive dan kehilangan 7 sampel |
| Full Sour | 48 | 10 | 3 | prediksi relatif presisi, tetapi banyak sampel kelas terlewat |
| Severe Insect Damage | 41 | 7 | 3 | pola serupa Full Sour: recall lebih lemah |
| Withered | 34 | 7 | 4 | Fig. 11 menunjukkan empat sampel bergeser ke Immature |
| Immature | 35 | 6 | 6 | batas keputusan dua arah masih lemah |
| Partial Black | 47 | 6 | 7 | menerima dan menghasilkan confusion dengan kelas sekelompok |

Pada Fig. 11, pola paling informatif bukan hanya skor terendah: Full Sour
kehilangan 10 dari 58 sampel, Withered diprediksi sebagai Immature pada 4 dari
41 sampel, dan Partial Sour hanya benar 28 dari 35. HBP seharusnya mengurangi
kesalahan terstruktur ini; menaikkan kelas yang sudah 100% tidak cukup.

## Hipotesis

> Dengan backbone, split, augmentasi, optimizer, dan seed yang sama,
> MobileNetV3-HBP meningkatkan hard-class macro-F1 dan mengurangi confusion pada
> pasangan kelas sulit dibanding MobileNetV3-GAP.

Paper hanya membentuk hipotesis ini. Paper tidak dapat membuktikan HBP karena
tidak menjalankan HBP.

## Eksperimen terkontrol

| Kode | Head | Fungsi kontrol |
|---|---|---|
| M0 | GAP | baseline orde pertama |
| M0b | factorized bilinear pada lapisan terakhir | apakah interaksi orde kedua saja cukup |
| M1 | HBP tiga kedalaman | apakah interaksi lintas lapisan memberi nilai tambahan |

Dengan 17 kelas, implementasi saat ini memiliki sekitar 2,988 juta parameter
untuk M0, 3,468 juta untuk M0b, dan 3,562 juta untuk M1. M0b dan M1 sengaja
dibuat berdekatan agar perbandingan bilinear satu lapis vs hierarkis tidak
didominasi perbedaan kapasitas. M0 tetap diperlukan sebagai baseline GAP
standar.

Semua model memakai gambar asli yang dibagi sebelum augmentasi. Rotasi hanya
dilakukan pada train untuk menghindari turunan biji yang sama tersebar ke
train/test.

Jalankan minimal tiga seed. Pilih checkpoint memakai source validation, lalu
evaluasi test satu kali. Simpan prediksi per sampel agar delta dapat diperiksa
dan paired bootstrap dapat dilakukan.

Split 70/20/10 dari 979 citra hanya menghasilkan sekitar 98 citra test, sehingga
dukungan per kelas berkisar beberapa sampel saja. Tiga-seed holdout pada
notebook dipakai sebagai **screening awal**. Jika HBP tampak unggul, bukti
konfirmatori harus memakai stratified 5-fold pada 979 identitas gambar asli,
dengan augmentasi hanya di train fold, lalu menggabungkan out-of-fold
predictions. Ini memberi setiap gambar kesempatan menjadi test tepat satu kali
tanpa augmentation leakage.

## Endpoint dan keputusan

Endpoint primer:

1. hard-class F1 gabungan delapan kelas;
2. macro-F1 seluruh kelas;
3. worst-class F1.

Endpoint sekunder:

1. F1 setiap hard group;
2. confusion antaranggota kelompok;
3. parameter, latency, dan ukuran model.

HBP dipertahankan hanya bila peningkatan hard-class F1 muncul konsisten pada
mayoritas seed, tidak dibayar dengan penurunan material macro-F1, dan confusion
yang ditargetkan ikut turun. Kenaikan satu run atau kenaikan hanya pada kelas
mudah bukan bukti yang cukup.

Jika M0b setara dengan M1, manfaatnya kemungkinan berasal dari statistik orde
kedua, bukan struktur hierarkis. Jika M0, M0b, dan M1 setara dalam variasi antar
seed, HBP harus dibuang dari model final dan fokus dipindahkan ke generalisasi
lintas domain.

## Batas perbandingan dengan angka paper

Hasil repositori ini tidak boleh disebut peningkatan langsung atas 89,04% paper
karena protokol split bersih berbeda dari urutan augmentasi/split yang
dilaporkan paper. Klaim yang sah adalah perbandingan M0, M0b, dan M1 di bawah
protokol kita yang sama. Angka paper dipakai sebagai motivasi dan referensi
kelas sulit.
