# Protokol Global-Local HBP Mixture-of-Experts

## Asal metode dan batas klaim

E1 terinspirasi oleh *Mixture of Granularity-Specific Experts* (MGE-CNN),
Zhang et al., ICCV 2019. Paper aslinya memakai beberapa ConvNet terpisah,
Grad-CAM untuk membuat crop bergranularitas makin halus, gate prediksi, dan
diversity loss. E1 **bukan reproduksi MGE-CNN**. E1 adalah adaptasi ringan
untuk menguji prinsip global-local expert pada MobileNetV3-HBP:

- satu backbone bersama, bukan beberapa ConvNet;
- expert global berupa HBP baseline M1;
- expert lokal memakai proyeksi 1x1 dan global max pooling dari endpoint tengah
  14 x 14;
- gate per sampel menggabungkan logit kedua expert;
- tidak memakai Grad-CAM crop saat training maupun inferensi.

Referensi resmi: https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Learning_a_Mixture_of_Granularity-Specific_Experts_for_Fine-Grained_Categorization_ICCV_2019_paper.html

## Ablasi terkendali

| Kode | Expert | Fusion | Objektif |
|---|---|---|---|
| M1 | HBP global | - | CE |
| E1 | HBP global + local GMP | gate per sampel | CE fusion + 0,3 CE tiap expert + 0,05 diversity |

Backbone, pretrained weights, endpoint, HBP, augmentasi, optimizer, split,
epoch, dan seed sama. Gate dimulai dengan prior 0,8 untuk HBP dan 0,2 untuk
expert lokal agar kandidat tidak merusak baseline pada awal training.

Diversity loss memisahkan distribusi non-target kedua expert. Kelas teratas
bersama dimask terlebih dahulu, sehingga loss tidak secara langsung meminta
expert berbeda pada prediksi utama.

## Pelaporan wajib

Selain Macro-F1, Hard-F1, Worst-F1, dan F1 per kelas, laporan harus memuat:

- jumlah parameter dan latency dengan batch/perangkat yang sama;
- rata-rata bobot gate HBP/lokal;
- fraksi sampel yang memilih masing-masing expert dan entropy gate;
- hasil per seed, bukan hanya seed terbaik.

## Keputusan yang ditetapkan sebelum test

1. Screening pertama memakai validation seed 42.
2. E1 lolos bila Macro-F1 dan Hard-F1 naik, Worst-F1 tidak turun lebih dari
   satu poin, serta rata-rata bobot expert lokal sekurangnya 0,02.
3. Jika lolos, ulangi seed 42, 123, dan 2026 lalu evaluasi test.
4. Kandidat diklaim stabil hanya bila delta Macro-F1 positif minimal dua dari
   tiga seed dan mean Macro-F1/Hard-F1 lebih tinggi tanpa penurunan besar pada
   Worst-F1.
5. Bila gagal, E1 dihentikan; penambahan expert, attention, atau backbone tidak
   dilakukan pada eksperimen yang sama.

Dataset 17 kelas saat ini dipakai sebagai screening arsitektur. Validasi lintas
dataset publik dilakukan setelah kandidat dikunci dan harus mempertahankan
protokol split bebas duplikat pada tingkat objek.
