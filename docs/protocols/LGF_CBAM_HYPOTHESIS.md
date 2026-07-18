# Hipotesis LGF-CBAM tanpa HBP

## Pertanyaan

Apakah attention channel-spatial dengan Learnable Gated Fusion (LGF-CBAM)
dapat memperbaiki klasifikasi 17 cacat green coffee bean menggunakan
MobileNetV3 dan GAP, tanpa HBP?

Referensi: Techie-Menson et al. (2026), *Enhanced convolutional block attention
module with Learnable Gated Fusion (LGF-CBAM) for cocoa pod disease
identification*, PLOS One, DOI 10.1371/journal.pone.0348147.

## Catatan kritis terhadap paper

Paper tidak sepenuhnya konsisten sehingga implementasi tidak boleh disalin
tanpa definisi operasional:

1. Narasi halaman 10 menyebut channel dan spatial pathway paralel dari feature
   map asli, tetapi pseudocode dan Persamaan 8-11 pada halaman 11 membuat
   spatial attention memproses `Fch`. Implementasi ini mengikuti pseudocode:
   `F -> Fch -> Fsp`, kemudian memadukan `Fch` dan `Fsp`.
2. Persamaan 6 menulis gate dari `MLP(GAP(F))`, sedangkan uraian dan Persamaan
   10 memakai `MLP([GAP(Fch), GAP(Fsp)])`. Implementasi mengikuti Persamaan 10.
3. Table 11 melaporkan Proposed System accuracy 97,56%, lebih rendah daripada
   ResNet-101-CBAM 98,34%, walaupun narasi menyatakan proposed system terbaik.
   Table 12 melaporkan angka proposed model lain, 98,95%. Karena kontradiksi
   ini, angka peningkatan paper tidak dipakai sebagai ekspektasi hasil kita.
4. Paper menggunakan tiga kelas cocoa dan random ShuffleSplit. Eksperimen kopi
   memiliki 17 kelas fine-grained dan wajib dinilai dengan Macro-F1,
   Hard-class F1, dan Worst-class F1, bukan accuracy saja.

## Definisi model

Semua model memakai backbone, augmentasi, optimizer, split, dan classifier yang
sama. Hanya head setelah feature map terdalam MobileNetV3 yang berubah.

| Kode | Head | Peran |
|---|---|---|
| M0 | GAP | baseline tanpa attention |
| M0a | `0.5 Fch + 0.5 Fsp`, lalu GAP | kontrol attention/fusion tetap |
| M0lgf | `alpha Fch + beta Fsp`, lalu GAP | uji learnable fusion |

Untuk M0lgf, descriptor `GAP(Fch)` dan `GAP(Fsp)` dikonkatenasi dan dilewatkan
ke MLP. Softmax menjamin `alpha + beta = 1` untuk setiap sampel. Layer terakhir
gate diinisialisasi nol, sehingga `alpha = beta = 0.5` pada awal training.
Reduction ratio 16 mengikuti pilihan paper. Pembuatan gate tidak menggeser RNG
global, sehingga untuk seed yang sama M0a dan M0lgf memperoleh inisialisasi
encoder, attention pathway, dan classifier yang identik.

Parameter model:

| Model | Parameter |
|---|---:|
| M0 | 2.988.289 |
| M0a | 3.103.587 |
| M0lgf | 3.218.969 |

M0lgf memiliki 115.382 parameter gate lebih banyak daripada M0a (sekitar 3,7%
dari M0a). Oleh sebab itu kenaikan kecil M0lgf vs M0a masih harus ditafsirkan
dengan hati-hati sebagai gabungan adaptivitas gate dan tambahan kapasitas.

## Protokol keputusan

Seed 123 dipakai sebagai screening pertama untuk menghemat komputasi.
Perbandingan primer adalah M0lgf vs M0a; M0 digunakan untuk memastikan bahwa
attention sendiri bermanfaat.

- **Lulus screening:** M0lgf meningkatkan Macro-F1 dan Hard-F1 terhadap M0a,
  tidak menurunkan Worst-F1, serta tidak kalah dari M0 pada endpoint primer.
- **Attention berguna, LGF belum terbukti:** M0a dan M0lgf mengungguli M0,
  tetapi M0lgf tidak mengungguli M0a.
- **Ditolak:** kedua model attention kalah dari M0 atau terjadi collapse kelas.

Hanya model yang lulus screening seed 123 yang dilanjutkan ke seed 42 dan 2026.
Grouped 5-fold baru dijalankan bila pola tiga seed konsisten. Hasil satu seed
tidak boleh menjadi klaim final.
