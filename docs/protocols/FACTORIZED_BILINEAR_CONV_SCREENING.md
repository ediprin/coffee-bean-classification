# Factorized Bilinear Conv fail-fast

## Pertanyaan

Apakah interaksi kuadratik lokal dari Conv-FBN meningkatkan klasifikasi
fine-grained Coffee17 dibandingkan EfficientNetV2-B0 GAP dan kontrol dengan
jumlah parameter identik?

Rujukan utama adalah Li et al., *Factorized Bilinear Models for Image
Recognition*, ICCV 2017:
https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Factorized_Bilinear_Models_ICCV_2017_paper.pdf

## Model yang dikunci

| Kode | Backbone | Transformasi sebelum global pooling |
|---|---|---|
| BE2G | EfficientNetV2-B0 | linear classifier setelah GAP |
| BE2H | EfficientNetV2-B0 | HBP, referensi bilinear lintas-layer |
| FB0 | EfficientNetV2-B0 | linear factor paths, tanpa kuadrat |
| FB1 | EfficientNetV2-B0 | linear + factorized quadratic paths |

FB0 dan FB1 memakai parameter yang persis sama. Satu-satunya perbedaan adalah
`projected` pada FB0 dan `projected.square()` pada FB1. Karena FB0 masih
merupakan parametrization linear redundan, ia adalah kontrol jumlah parameter
dan optimisasi, bukan kandidat deployment.

FB1 mengikuti Conv-FBN paper pada empat keputusan utama:

- FB Conv `1x1` pada setiap lokasi feature map terakhir;
- rank/factor `k=20`;
- `Tanh` sebelum FB layer;
- DropFactor keep probability `p=0.5`;
- slow-start LR FB layer dari `0.1x` menuju `1.0x` selama tiga epoch.

Paper tidak menyediakan detail initialization PyTorch yang dapat diverifikasi.
Adaptasi ini menginisialisasi factor dengan `std=1/sqrt(C*k)` agar jumlah rank
paths tetap berorde satu. Keputusan tersebut adalah detail implementasi riset
ini, bukan klaim berasal dari paper.

## Data dan evaluasi

- Coffee17 clean grouped fold 1;
- validation saja, test tetap terkunci;
- screening seed `42`;
- image 224, CE, optimizer/scheduler/augmentasi sama dengan BE2G;
- metrik: Macro-F1, Hard-F1, Worst-F1.

FB1 hanya boleh masuk konfirmasi tiga seed apabila **keduanya** PASS:

1. `BE2G vs FB1`, membuktikan manfaat total terhadap GAP;
2. `FB0 vs FB1`, mengisolasi manfaat operasi kuadratik dari jumlah parameter.

Untuk setiap PASS, Macro-F1 dan Hard-F1 harus naik, sedangkan Worst-F1 tidak
boleh turun lebih dari satu poin persentase. Perbandingan BE2H vs FB1 bersifat
deskriptif dan tidak menjadi syarat screening karena HBP menjawab mekanisme
lintas-layer yang berbeda.

## Batas klaim

- PASS satu seed hanya mengizinkan konfirmasi; bukan hasil tesis final.
- FAIL menghentikan FB Conv tanpa tuning rank, keep probability, atau gabungan
  HBP post-hoc.
- Tidak ada klaim efisiensi kernel khusus karena implementasi memakai einsum
  PyTorch, bukan kernel Conv-FBN teroptimasi.
- Tidak ada test evaluation pada screening.

## Perintah

```bash
python -u -m bilinear_lmmd.experiments.run_factorized_bilinear_conv_screening \
  --data-root data/coffee17-clean/folds/fold_1 \
  --baseline-root outputs/backbone-results \
  --output-root outputs/factorized-bilinear-conv \
  --seeds 42 \
  --evaluation-split val
```

## Konfirmasi tiga seed

Konfirmasi hanya boleh dijalankan setelah report screening seed 42 berstatus
`PASS`. Artefak seed 42 dipakai ulang dan hanya seed 123 serta 2026 yang perlu
dilatih. Tidak ada perubahan rank, DropFactor, slow-start, augmentasi, atau
hyperparameter setelah melihat hasil screening.

FB1 dinyatakan terkonfirmasi hanya jika `BE2G vs FB1` dan `FB0 vs FB1`
memenuhi seluruh syarat berikut pada validation:

- rata-rata delta Macro-F1 positif dan naik pada minimal 2/3 seed;
- rata-rata delta Hard-F1 positif dan naik pada minimal 2/3 seed;
- rata-rata delta Worst-F1 tidak turun lebih dari satu poin persentase.

Perbandingan terhadap BE2H tetap dilaporkan, tetapi bukan syarat utama karena
tujuan mekanistik FB1 adalah menguji operasi kuadratik terhadap GAP dan kontrol
linear capacity-matched. Test tetap terkunci apa pun hasil konfirmasi.

```bash
python -u -m bilinear_lmmd.experiments.run_factorized_bilinear_conv_confirmation \
  --data-root data/coffee17-clean/folds/fold_1 \
  --baseline-root outputs/backbone-results \
  --output-root outputs/factorized-bilinear-conv \
  --seeds 42 123 2026 \
  --evaluation-split val
```
