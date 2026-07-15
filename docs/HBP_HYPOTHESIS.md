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
untuk M0, 3,551 juta untuk M0b, dan 3,562 juta untuk M1. M0b dan M1 sengaja
dibuat berdekatan agar perbandingan bilinear satu lapis vs hierarkis tidak
didominasi perbedaan kapasitas. M0 tetap diperlukan sebagai baseline GAP
standar.

M0b memakai bilinear rank 160 dari feature map akhir, lalu ekspansi linear ke
embedding 1.536 dimensi. Dengan demikian, M0b dan M1 mempunyai dimensi embedding
1.536, classifier 1.536-ke-17 yang identik, dan perbedaan jumlah parameter hanya
sekitar 0,32%. Perbedaan utamanya adalah satu feature map pada M0b versus tiga
kedalaman pada M1.

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

## Hasil screening tiga seed

Hasil berikut memakai split asli 70/20/10 yang sama, augmentasi hanya pada
train, dan seed 42, 123, serta 2026.

| Metrik | GAP mean (%) | HBP mean (%) | Mean delta (poin) | Arah per seed |
|---|---:|---:|---:|---|
| Macro-F1 | 85,49 | 87,90 | +2,41 | naik 3/3 |
| Hard-class F1 | 85,49 | 87,02 | +1,53 | naik 2/3 |
| Worst-class F1 | 66,67 | 66,67 | 0,00 | tetap 3/3 |
| Sour/black F1 | 84,86 | 86,51 | +1,64 | tidak konsisten |
| Shape/withered F1 | 84,46 | 90,03 | +5,57 | naik 3/3 |
| Insect-damage F1 | 87,97 | 83,27 | -4,70 | turun 2/3 |

### Audit Macro-F1 dan hard-class F1 GAP

Kedua mean kebetulan sama setelah pembulatan. Nilainya berbeda pada setiap seed:

| Seed | Macro-F1 GAP (%) | Hard-class F1 GAP (%) |
|---:|---:|---:|
| 42 | 85,50 | 86,45 |
| 123 | 86,86 | 86,61 |
| 2026 | 84,11 | 83,42 |

Implementasi metrik menggabungkan prediksi seluruh test set sebelum menghitung
F1, bukan merata-ratakan F1 per batch. Macro-F1 memakai daftar eksplisit 17
kelas dan hard-class F1 memakai union delapan kelas yang ditetapkan di dokumen
ini. `zero_division=0` diterapkan. Test otomatis juga mengunci keanggotaan hard
subset agar tidak berubah setelah hasil dilihat.

### Putusan sementara

Hipotesis didukung sebagian. HBP meningkatkan macro-F1 secara konsisten dan
hard-class F1 pada mayoritas seed, terutama pada shape/withered. HBP belum
memperbaiki worst-class dan cenderung merugikan insect damage. Karena itu HBP
berstatus **kandidat didukung empiris awal**, bukan komponen final.

M0b tetap menjadi eksperimen kausal wajib. Screening holdout ini juga harus
dikonfirmasi dengan stratified grouped 5-fold pada identitas citra asli sebelum
kesimpulan final.

Pipeline konfirmasi tersedia melalui `prepare_grouped_folds` dan
`run_grouped_cv`. Outer test fold mencakup sekitar 20% citra asli, validation
diambil sekitar 10% dari keseluruhan data, dan sisanya menjadi train. Setiap
identitas muncul pada outer test tepat satu kali. Runner menggabungkan lima
test fold menjadi 979 out-of-fold predictions sebelum menghitung metrik akhir.

## Hasil grouped 5-fold sebelum deduplikasi konten

Audit SHA-256 lanjutan menemukan 13 kelompok exact duplicate: 12 pasangan
berlabel sama dan satu pasangan byte-identik dengan label Fade/Partial Sour.
Sebelas kelompok tersebar pada outer fold berbeda. Karena split ini mengelompokkan
filename, bukan hash konten, hasil 979 OOF di bawah masih berguna sebagai
screening tetapi belum merupakan konfirmasi content-clean.

Agregasi 979 out-of-fold predictions mengonfirmasi kenaikan agregat HBP, tetapi
juga memperjelas trade-off pada kelas terburuk.

| Metrik | GAP (%) | HBP (%) | Delta HBP (poin) |
|---|---:|---:|---:|
| Macro-F1 | 82,42 | 84,41 | +1,99 |
| Hard-class F1 | 74,59 | 77,37 | +2,78 |
| Worst-class F1 | 56,84 | 54,12 | -2,72 |

Audit pasangan prediksi menemukan 66 sampel yang hanya benar oleh HBP dan 45
yang hanya benar oleh GAP. Sebanyak 106 sampel salah oleh keduanya. Oracle yang
memilih prediksi benar menggunakan label mencapai akurasi 89,17%; angka ini
hanya batas atas, bukan model yang sah.

Ensemble probabilitas dengan satu alpha yang dipilih pada validation setiap
fold tidak memanfaatkan batas atas tersebut. Macro-F1 ensemble 83,48% dan
hard-class F1 75,56%, keduanya di bawah HBP. Worst-class F1 pulih menjadi
56,84%. Alpha antar-fold `[0,30; 1,00; 0,00; 0,55; 0,15]` menunjukkan bahwa
satu bobot global tidak stabil dan tidak cukup untuk komplementaritas yang
bergantung pada kelas/sampel.

## Eksperimen feature fusion yang dipraregistrasikan

Dua model baru diuji lebih dulu pada holdout dengan seed 42, 123, dan 2026:

| Kode | Representasi | Peran kausal |
|---|---|---|
| M1 | HBP 1536-D + classifier linear | baseline HBP |
| M1c | HBP + proyeksi nonlinear | kontrol tambahan kapasitas |
| M1f | proyeksi HBP + proyeksi GAP, lalu konkatenasi | uji informasi GAP |

M1c dan M1f berbeda kurang dari 0,1% parameter. Karena itu perbandingan primer
untuk klaim fusion adalah **M1f vs M1c**, bukan hanya M1f vs M1. Fusion dianggap
layak dibawa ke grouped 5-fold bila mean macro-F1 dan hard-class F1 mengungguli
M1c serta delta macro-F1 positif pada minimal dua dari tiga seed. Worst-class
F1 tetap dilaporkan sebagai constraint; peningkatan agregat tidak boleh disebut
solusi worst class jika metrik tersebut turun.

### Hasil screening fusion terkompresi

| Metrik | M1c (%) | M1f (%) | Delta M1f (poin) |
|---|---:|---:|---:|
| Macro-F1 | 83,37 | 85,55 | +2,18 |
| Hard-class F1 | 82,75 | 84,29 | +1,54 |
| Worst-class F1 | 65,28 | 49,40 | -15,88 |

M1f meningkatkan macro-F1 terhadap kontrol pada 3/3 seed dan hard-class F1
pada 2/3 seed. Sebelas kelas memperoleh mean F1 lebih tinggi, termasuk Shell,
Withered, Immature, serta kedua kelas insect damage. Namun Full Sour turun
25,13 poin dan tidak membaik pada satu pun seed; pada seed 123 F1 Full Sour
hanya 20,00%. M1f juga tetap di bawah M1 asli pada macro-F1 dan hard-class F1.

Hasil ini mendukung adanya informasi komplementer pada GAP, tetapi menolak M1f
sebagai model final. M1c dan M1f sama-sama mengganti embedding HBP 1536-D dengan
bottleneck 672-D/512-D. Penurunan M1c terhadap M1 menunjukkan bahwa perubahan
representasi tersebut sendiri merupakan confound yang merugikan.

### Residual fusion tanpa kompresi yang dipraregistrasikan

| Kode | Representasi |
|---|---|
| M1rc | HBP asli 1536-D + auxiliary HBP 80-D |
| M1r | HBP asli 1536-D + residual GAP 128-D |

Bobot proyeksi tambahan keduanya sama-sama 122.880 dan total parameter berbeda
sekitar 0,03%. Komponen HBP 1536-D dipertahankan tanpa perubahan. Seed 123
dipakai lebih dulu hanya sebagai stress test rekayasa karena merupakan failure
case M1f, bukan sebagai bukti final. Jika collapse Full Sour berkurang, seed 42
dan 2026 wajib dijalankan sebelum kesimpulan. Klaim manfaat residual fusion
tetap memerlukan M1r mengungguli M1rc pada agregasi tiga seed dan tidak kalah
dari M1 asli pada endpoint primer.

## Status komponen

> Pembaruan 16 Juli 2026: hasil test content-clean tiga seed dan keputusan model
> final tersedia di [FINAL_HBP_RESULTS.md](FINAL_HBP_RESULTS.md). Hasil final
> tersebut menjadi acuan utama jika bertentangan dengan screening/OOF lama di
> bawah.

| Komponen | Status saat ini |
|---|---|
| MobileNetV3 | Baseline tervalidasi |
| HBP | Diterima: mean test Macro +1,32; Hard +2,48; Worst +7,64 poin vs GAP |
| SPPF-Attention + HBP | Ditolak: seluruh metrik utama turun pada test |
| LMMD | Belum diuji dalam hasil ini |
| HBP + LMMD | Belum boleh diklaim sinergis |
| M0b single-layer bilinear | Screening selesai; tidak mengungguli HBP secara umum |
| GAP-HBP probability ensemble | Selesai; kalah dari HBP pada Macro/Hard-F1 |
| M1c/M1f feature fusion | Selesai; sinyal GAP ada, tetapi Worst-F1 collapse |
| M1rc/M1r residual fusion | Stress test seed 123 selesai; belum bukti tiga seed |
| Filename-grouped 5-fold (979) | Selesai untuk GAP dan HBP; mengandung exact duplicate |
| Content-clean grouped 5-fold (965) | Fold siap; GAP/HBP perlu dikonfirmasi ulang |
| XAI | Tahap analisis pola kelas setelah konfirmasi |

## Ablasi atribut tanpa HBP

Untuk menguji apakah ciri khas kopi yang lebih eksplisit dapat menjelaskan
kesalahan HBP, seluruh kombinasi warna (`C`), bentuk (`S`), dan tekstur (`T`)
diuji: `C`, `S`, `T`, `CS`, `CT`, `ST`, dan `CST`. Eksperimen menggunakan outer
grouped 5-fold yang sama dengan GAP/HBP. Hyperparameter RBF-SVM dipilih hanya
dari validation pada masing-masing fold, kemudian seluruh outer-test digabung
menjadi 979 prediksi OOF.

Hipotesis atribut dinilai dalam dua tahap:

1. Ranking tujuh kombinasi menentukan atribut yang membawa informasi paling
   kuat tanpa deep feature.
2. Prediksi kombinasi terbaik dipasangkan dengan prediksi OOF HBP untuk
   menghitung `attribute_only`, `hbp_only`, dan `both_wrong`.

Model atribut tidak harus mengungguli HBP agar berguna. Kandidat hybrid hanya
layak bila terdapat cukup sampel `attribute_only`, khususnya pada kelas yang
lemah di HBP. Jika kesalahannya hampir sama, menambah cabang atribut hanya
menambah kompleksitas tanpa bukti informasi komplementer.

### Hasil ablasi atribut dan audit komplementaritas

| Fitur | Macro-F1 (%) | Hard-F1 (%) | Worst-F1 (%) |
|---|---:|---:|---:|
| C | 60,60 | 55,62 | 36,00 |
| S | 30,58 | 24,10 | 8,33 |
| T | 44,94 | 40,71 | 9,52 |
| CS | 69,10 | 61,93 | 30,77 |
| CT | 65,79 | 61,64 | 41,18 |
| ST | 54,85 | 48,70 | 25,35 |
| CST | **71,74** | **65,95** | **49,52** |

CST tidak menggantikan HBP (Macro-F1 84,41%), tetapi pasangan prediksi OOF
menemukan 51 sampel yang hanya benar oleh CST, 175 yang hanya benar oleh HBP,
653 yang benar oleh keduanya, dan 100 yang salah oleh keduanya. Oracle akurasi
HBP-CST adalah 89,79%; ini batas atas dan bukan hasil model yang sah. CST
menyelamatkan 51 dari 151 kesalahan HBP. Rescue terbesar muncul pada Partial
Sour (11), Immature (8), Withered (8), dan Slight Insect Damage (5), tetapi
Partial Sour adalah satu-satunya kelas dengan `CST_only > HBP_only`.

Karena manfaat bergantung pada sampel, eksperimen berikutnya adalah
cross-fitted disagreement gate. Untuk setiap outer fold, aturan pasangan
prediksi dipelajari dari empat fold lain; `min_support` dan margin dipilih lewat
inner leave-one-fold validation. Pasangan jarang/default tetap memakai HBP.
Screening dinilai berhasil hanya bila Macro-F1 dan Hard-F1 gate mengungguli HBP
tanpa penurunan Worst-F1. Karena memakai hard prediction OOF yang sudah ada,
hasil ini tetap eksploratif dan harus dikonfirmasi dengan calibration prediction
validation atau test independen sebelum klaim final.
