# Konfirmasi tiga seed Hong DSConv-only

## Hipotesis terkunci

HCD1 (EfficientNetV2-B0 dengan lima Distribution Shifting Convolution awal,
GAP, linear classifier, dan CE) meningkatkan klasifikasi fine-grained Coffee17
dibanding BE2G (GAP) dan BE2H (HBP). Hipotesis ini berasal dari ablasi
faktorial seed 123; tidak dilakukan perubahan bit, block size, stage, loss,
augmentasi, atau epoch setelah melihat validation.

## Model dan data

- kandidat: HCD1 saja;
- baseline: checkpoint BE2G dan BE2H yang sama dengan benchmark backbone;
- seed: 42, 123, 2026;
- split: validation bersih/grouped;
- test tetap terkunci;
- checkpoint HCD1 seed 123 dipakai ulang, bukan dilatih ulang.

HCS1 dan HCDS1 tidak diteruskan karena sudah FAIL. PConv tetap tidak diadaptasi
karena kontribusinya berada pada detection head dan occlusion handling.

## Gate konfirmasi

HCD1 harus PASS terhadap **kedua** baseline. Untuk setiap perbandingan:

1. rata-rata delta Macro-F1 positif dan naik minimal 2/3 seed;
2. rata-rata delta Hard-F1 positif dan naik minimal 2/3 seed;
3. rata-rata Worst-F1 tidak turun lebih dari satu poin.

Jika salah satu perbandingan FAIL, HCD1 dihentikan dan test tidak dibuka. Jika
keduanya PASS, statusnya menjadi kandidat terkonfirmasi pada validation; test
baru boleh dibuka melalui protokol final terpisah.

Tidak ada klaim runtime atau kompresi aktual karena DSConv masih berupa
simulasi quantization-aware PyTorch.

## Hasil

**FINAL: FAIL.** Konfirmasi tiga seed menghasilkan:

| Perbandingan | Delta Macro-F1 | Delta Hard-F1 | Delta Worst-F1 | Putusan |
|---|---:|---:|---:|---|
| BE2G vs HCD1 | -0,49 ± 1,85 | -1,65 ± 3,43 | -7,98 ± 19,31 | FAIL |
| BE2H vs HCD1 | -1,65 ± 2,04 | -1,75 ± 3,34 | -6,95 ± 8,67 | FAIL |

Angka menggunakan poin persentase. HCD1 gagal terhadap kedua baseline dan
Worst-F1 turun besar. Sinyal positif seed 123 dinyatakan tidak stabil. Sesuai
gate, test tidak dibuka dan kombinasi DSConv+HBP tidak dijalankan sebagai
penyelamatan post-hoc.
