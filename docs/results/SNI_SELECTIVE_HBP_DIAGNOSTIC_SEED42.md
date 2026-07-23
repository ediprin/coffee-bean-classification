# Hasil selective residual HBP diagnostic seed 42

**Status: SELESAI -- STOP, TEST TETAP TERKUNCI.**

Diagnostik ini dimulai dari checkpoint SNIB1 seed 42. Encoder EfficientNetV2,
fusi multiresolusi, dan classifier dasar dibekukan. Dua residual head dengan
jumlah parameter identik dilatih selama 10 epoch pada train split dan dipilih
hanya menggunakan validation:

- `SNIDG`: projected hierarchical GAP residual;
- `SNIDH`: hierarchical bilinear pooling residual.

Keduanya hanya dapat mengubah 12 logit kondisi biji. Sembilan logit kulit dan
benda asing tetap berasal dari classifier SNIB1.

## Putusan praregistrasi

| Perbandingan | Macro-F1 meningkat | Hard-F1 meningkat | Worst-F1 terjaga | Putusan |
|---|---:|---:|---:|---|
| `SNIB1_vs_SNIDH` | Tidak | Tidak | Ya | **FAIL** |
| `SNIDG_vs_SNIDH` | Tidak | Ya | Tidak | **FAIL** |

Putusan final runner: **STOP**. Residual HBP tidak mengalahkan baseline SNIB1
dan juga tidak mengalahkan residual projected-GAP secara menyeluruh. Karena itu:

- seed 123 dan 2026 tidak dijalankan;
- split test tidak dibuka;
- residual HBP selektif dihentikan;
- hasil tidak boleh dipresentasikan sebagai peningkatan model.

## Batas pencatatan

Keluaran yang tersedia saat pencatatan berisi keputusan dan boolean kriteria,
bukan angka metrik/delta lengkap. Angka tersebut tidak direkonstruksi atau
diperkirakan. Artefak lengkap tetap berada di Google Drive:

`MyDrive/sni-selective-hbp-diagnostic-v1/val_reports/selective_residual_diagnostic.json`

Apabila artefak itu kemudian dimasukkan ke repo, dokumen ini dapat dilengkapi
dengan Macro-F1, Hard-F1, Worst-F1, dan delta numeriknya tanpa mengubah putusan.
