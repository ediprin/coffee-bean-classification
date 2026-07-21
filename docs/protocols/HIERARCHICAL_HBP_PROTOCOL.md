# Protokol faktorial auxiliary hierarchy: GAP dan HBP

## Hipotesis

Coffee17 memisahkan tingkat cacat yang masih memiliki satu induk visual:
`Full/Partial Black`, `Full/Partial Sour`, serta `Severe/Slight Insect
Damage`. H0 dan H1 menguji apakah supervisi parent membantu representasi GAP
atau HBP membangun struktur antarkelas tanpa mengganti prediksi akhir 17 kelas.

Pemetaan `Black` dan `Sour` mengikuti penggabungan kategori pada Hong et al.
(2026). Pemetaan `Insect Damage` adalah penggabungan tingkat keparahan yang
analog. Sebelas kelas lain menjadi parent tunggal agar seluruh 17 kelas
membentuk partisi lengkap dan tidak dimasukkan ke kategori visual yang
arbitrer.

## Ablasi terkendali

| Kode | Backbone/head | Objective |
|---|---|---|
| M0 | MobileNetV3-Large + GAP | `CE_fine` |
| H0 | MobileNetV3-Large + GAP | `CE_fine + 0.2 CE_parent` |
| M1 | MobileNetV3-Large + HBP | `CE_fine` |
| H1 | MobileNetV3-Large + HBP | `CE_fine + 0.2 CE_parent` |

H0/H1 menambahkan classifier parent 14 kelas hanya untuk supervisi auxiliary.
Prediksi dan seluruh metrik tetap berasal dari classifier fine 17 kelas.
Di dalam setiap pasangan M0/H0 dan M1/H1, backbone, pretrained weights, pooling,
classifier fine, dropout, optimizer,
augmentasi, split, epoch, dan seed tidak berubah. Inisialisasi auxiliary head
tidak menggeser RNG sehingga initial weights dan urutan batch baseline/kandidat tetap
sebanding pada seed yang sama.

Bobot `0.2` ditetapkan sebelum screening agar fine objective tetap dominan;
tidak dilakukan pencarian lambda pada test.

## Keputusan

1. Screening H0 pertama memakai validation seed 42, sama dengan screening H1.
2. Lolos bila Macro-F1 dan Hard-F1 meningkat serta Worst-F1 tidak turun lebih
   dari satu poin.
3. Kandidat yang lolos dikonfirmasi pada seed 42, 123, dan 2026.
4. Test dibuka satu kali setelah kandidat dikunci. Klaim keberhasilan
   memerlukan mean Macro-F1 dan Hard-F1 lebih tinggi serta delta Macro-F1
   positif pada sedikitnya dua dari tiga seed.

Screening `M0` versus `H0` mengisolasi hierarchy tanpa HBP. Jika lolos, H0
dikonfirmasi pada seed 42, 123, dan 2026. Faktorial `M0/H0/M1/H1` dapat dipakai
untuk mengukur apakah efek hierarchy bergantung pada pooling. SPPF-Attention
tidak dimasukkan sehingga penyebab tetap dapat ditentukan.
