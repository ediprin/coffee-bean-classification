# Protokol EMA untuk MobileNetV3-HBP

## Tujuan

Hasil test final Coffee-17 menunjukkan M1 meningkatkan mean seluruh metrik
terhadap GAP, tetapi memiliki standard deviation antarseed lebih tinggi.
Eksperimen M1e menguji apakah Exponential Moving Average (EMA) dapat menstabilkan
bobot HBP tanpa mengubah arsitektur atau biaya inferensi.

## Ablasi terkendali

| Kode | Training | Bobot evaluasi |
|---|---|---|
| M1 | MobileNetV3-HBP + CE | `best_raw.pt`, terbaik di validation |
| M1e | MobileNetV3-HBP + CE, trajectory yang sama | `best.pt`, EMA terbaik di validation |

M1e memakai decay `0.995` dan mulai setelah lima epoch penuh. Saat EMA mulai,
bobot disalin dari model aktif; update berikutnya memakai
`ema = 0.995 * ema + 0.005 * current`. Parameter BatchNorm dirata-ratakan,
sedangkan running statistics/buffer disalin dari model aktif. EMA tidak ikut
backpropagation dan tidak mengubah RNG, optimizer, loss, atau urutan batch.

M1 dan M1e memiliki parameter, ukuran deployment, FLOPs, dan latency yang sama.
EMA hanya menambah memori selama training.

Runner khusus melatih satu trajectory per seed dan menyimpan checkpoint raw
serta EMA secara terpisah. Ini memangkas waktu dibanding dua training dan
menghilangkan perbedaan trajectory sebagai confound.

## Larangan test leakage

Split test `fold_1` yang menghasilkan hasil final M0/M1 telah dibuka dan tidak
boleh dipakai untuk memilih decay, start epoch, atau menerima M1e. Screening
pertama harus memakai validation fold baru, misalnya `fold_2`. Jika M1e lolos,
konfirmasi dilakukan pada fold lain atau dataset eksternal yang belum dipakai
untuk tuning.

## Kriteria yang dikunci

Pada seed 42, 123, dan 2026, M1e lolos screening bila:

1. mean Macro-F1 validation tidak lebih rendah daripada M1;
2. mean Hard-F1 meningkat;
3. mean Worst-F1 tidak turun lebih dari satu poin;
4. delta Macro-F1 positif pada sedikitnya dua dari tiga seed; dan
5. standard deviation Macro-F1 M1e lebih kecil daripada M1.

Jika gagal, decay dan start epoch tidak dituning pada fold yang sama. EMA
dicatat sebagai ablation negatif. Test lama tetap hanya melaporkan M1 yang sudah
terkunci.

Jalankan dengan `python -m bilinear_lmmd.run_ema_screening`; runner hanya
mengizinkan evaluation split `val`.
