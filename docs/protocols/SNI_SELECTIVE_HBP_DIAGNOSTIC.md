# SNI selective residual HBP diagnostic v1

## Tujuan

Diagnostik ini menguji satu pertanyaan sempit setelah ontology expert SNIB2
gagal: apakah statistik orde kedua membantu **khusus 12 kelas kondisi biji**
tanpa mengganti keputusan SNIB1 untuk kelas kulit dan benda asing?

Ini bukan konfirmasi model final. Eksperimen dibatasi pada validation seed 42
agar kandidat yang tidak menjanjikan dapat dihentikan tanpa menghabiskan tiga
seed dan test.

## Model yang dibandingkan

- `SNIB1`: EfficientNetV2-B0 + fusi empat resolusi + flat GAP classifier.
- `SNIDG`: SNIB1 + residual projected hierarchical GAP pada logit kelas 0–11.
- `SNIDH`: SNIB1 + residual HBP pada logit kelas 0–11.

`SNIDG` dan `SNIDH` memakai tiga feature map hasil fusi, proyeksi 128 channel,
embedding 384 dimensi, dan classifier residual 12 kelas. Jumlah parameter total
serta parameter yang dilatih harus identik. Perbedaannya hanya statistik:
first-order projected GAP versus interaksi perkalian HBP.

## Warm start dan pembekuan

Kedua kandidat dimulai dari `SNIB1_seed42/best.pt`. Tiga prefix wajib dimuat
secara lengkap: `encoder`, `fusion`, dan `flat_classifier`. Classifier residual
diinisialisasi nol, sehingga sebelum update pertama seluruh 21 logit persis
sama dengan SNIB1.

Selama 10 epoch:

- encoder dibekukan dan berada dalam mode evaluasi;
- fusi multiresolusi dibekukan dan berada dalam mode evaluasi;
- flat classifier SNIB1 dibekukan;
- hanya projection pool dan classifier residual yang dilatih;
- dropout residual boleh aktif, sedangkan dropout dasar tetap nonaktif;
- loss tetap cross-entropy 21 kelas;
- split test tidak dibaca.

Residual 12 dimensi di-pad nol ke 21 dimensi. Karena urutan kelas SNI bersifat
kontigu, residual hanya dapat mengubah `SNI_CLASSES[0:12]`. Logit kelas kulit
kopi, kulit tanduk, dan benda asing (`12:21`) tetap berasal dari SNIB1.

## Aturan keputusan yang dibekukan

`SNIDH` hanya masuk konfirmasi penuh bila **dua** perbandingan berikut PASS:

1. `SNIDH` versus SNIB1;
2. `SNIDH` versus kontrol berkapasitas sama SNIDG.

Setiap perbandingan harus memenuhi semuanya pada validation:

- delta Macro-F1 lebih besar dari nol;
- delta Hard-F1 lebih besar dari nol;
- delta Worst-F1 minimal -1 poin persentase.

Jika salah satu gagal, hasil akhir `STOP`: jangan menjalankan seed tambahan,
jangan membuka test, dan jangan mengklaim HBP selektif lebih baik. Jika keduanya
lolos, hasilnya hanya memberi izin merancang konfirmasi tiga seed; belum menjadi
bukti final.

## Menjalankan

Gunakan `notebooks/sni_selective_hbp_diagnostic_colab.ipynb`. Notebook meminta:

- backup shard dataset `MyDrive/coffee-sni-instance-crop-v1`;
- checkpoint dan report validation SNIB1 seed 42 di
  `MyDrive/sni-mrenet-v1`.

Artefak diagnostik ditulis langsung ke
`MyDrive/sni-selective-hbp-diagnostic-v1`, sehingga dapat dilanjutkan setelah
runtime Colab reset.

## Hasil seed 42

**Status: SELESAI -- STOP.** Pada validation seed 42:

- `SNIB1_vs_SNIDH`: FAIL karena Macro-F1 dan Hard-F1 tidak meningkat;
  Worst-F1 masih memenuhi batas preservasi.
- `SNIDG_vs_SNIDH`: FAIL karena Macro-F1 tidak meningkat dan Worst-F1 tidak
  terjaga, walaupun Hard-F1 meningkat.

Keputusan akhir sesuai aturan praregistrasi adalah menghentikan residual HBP.
Seed 123/2026 dan split test tidak dijalankan. Detail keputusan tersedia di
`docs/results/SNI_SELECTIVE_HBP_DIAGNOSTIC_SEED42.md`.
