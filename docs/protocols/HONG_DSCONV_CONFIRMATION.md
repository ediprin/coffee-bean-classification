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
