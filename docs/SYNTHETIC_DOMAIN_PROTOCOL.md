# Controlled Synthetic Domain-Shift Protocol

## Tujuan

Protokol ini menguji apakah MMD/LMMD membantu ketika distribusi visual target
diubah secara terkontrol. Hasilnya adalah **uji robustness sintetis dan
sanity-check UDA**, bukan bukti generalisasi dunia nyata.

LMMD tetap menerima label source dan gambar target tanpa memakai label target.
Label `target/val` dan `target/test` hanya dipakai untuk menghitung metrik.
Checkpoint dipilih berdasarkan source validation macro-F1.

## Domain

| Domain | Gangguan yang disimulasikan |
|---|---|
| `illumination` | gamma, brightness, color cast, dan bayangan berarah |
| `sensor` | downsampling, blur, noise, dan kompresi JPEG |
| `background` | penggantian latar setelah segmentasi biji |
| `combined` | background, illumination, dan sensor secara berurutan |

Setiap parameter dibuat deterministik dari seed, nama domain, split, kelas, dan
nama berkas. Resep persis setiap gambar disimpan dalam `manifest.jsonl`.
`metadata.json` menyimpan jumlah sampel, kegagalan segmentasi, dan batas klaim.

## Pencegahan leakage

Generator tidak membagi ulang data. Versi sintetis suatu identitas selalu
berada pada split yang sama dengan gambar asal:

```text
source/train/x.jpg -> target/train/x__combined.jpg
source/val/y.jpg   -> target/val/y__combined.jpg
source/test/z.jpg  -> target/test/z__combined.jpg
```

Dengan demikian gambar train tidak dibuat menjadi target test. Source dan
target memang merupakan pasangan dari identitas yang sama; karena itu benchmark
ini lebih mudah daripada domain nyata dan harus dilaporkan sebagai keterbatasan.

## Tahap eksperimen

### Screening hemat komputasi

Gunakan domain terberat dan satu seed terlebih dahulu:

```bash
python -u -m bilinear_lmmd.run_synthetic_benchmark \
  --source-root data/coffee_clean/folds/fold_1/source \
  --data-root /kaggle/working/coffee-synthetic \
  --output-root /kaggle/working/synthetic-results \
  --domains combined \
  --models M0 M1 M2 M3 M5 \
  --seeds 123 \
  --source-checkpoints \
    M0:123=/kaggle/working/finegrained-results/outputs/M0_seed123/best.pt \
    M1:123=/kaggle/working/finegrained-results/outputs/M1_seed123/best.pt
```

Argumen `--source-checkpoints` bersifat opsional. Jika checkpoint lama valid,
runner memeriksa seed, backbone, head, classifier, dan resolusinya lalu langsung
mengevaluasinya pada target sintetis. Jika argumen tidak diberikan, M0/M1 akan
dilatih seperti biasa.

Perbandingan yang diuji:

- `M2` vs `M0`: efek MMD pada fitur GAP;
- `M3` vs `M0`: efek LMMD pada fitur GAP;
- `M3` vs `M2`: alignment per kelas dibanding alignment global;
- `M5` vs `M1`: efek LMMD pada HBP.

### Konfirmasi

Hanya setelah screening menjanjikan, jalankan seluruh jenis shift dan tiga seed:

```bash
python -u -m bilinear_lmmd.run_synthetic_benchmark \
  --source-root data/coffee_clean/folds/fold_1/source \
  --data-root /kaggle/working/coffee-synthetic-all \
  --output-root /kaggle/working/synthetic-results-all \
  --domains illumination sensor background combined \
  --models M0 M1 M2 M3 M5 \
  --seeds 42 123 2026
```

Runner dapat dilanjutkan setelah interupsi. Dataset yang lengkap, training 50
epoch yang lengkap, dan report evaluasi yang sudah ada akan dilewati.

## Interpretasi

LMMD didukung sementara apabila peningkatan terhadap baseline pasangannya:

1. positif pada target macro-F1;
2. tidak hanya berasal dari satu seed;
3. tidak mengorbankan source macro-F1 secara besar;
4. muncul pada lebih dari satu jenis shift;
5. diperiksa juga pada hard-class dan worst-class F1.

Jika LMMD hanya menang pada `combined` atau satu seed, kesimpulannya adalah
sensitif terhadap kondisi, bukan unggul secara umum. Validasi target nyata tetap
diperlukan sebelum mengklaim ketahanan lapangan.

## Rescue control HBP-LMMD

Jika M5 dengan bobot LMMD 1,0 menaikkan target tetapi menurunkan source secara
besar, jalankan `configs/M5w01_mobilenetv3_hbp_lmmd_w01.yaml`. Konfigurasi ini
hanya mengubah `adaptation.weight` dari 1,0 menjadi 0,1; backbone, HBP, warmup,
confidence threshold, optimizer, dan epoch tetap sama.

Kriteria screening ditetapkan sebelum training:

- source macro-F1 tidak turun lebih dari 5 poin terhadap M1;
- target macro-F1 lebih tinggi daripada M1;
- worst-class F1 lebih besar dari nol;
- kelas yang sebelumnya runtuh tidak tetap bernilai nol.

Jika kontrol ini gagal, kombinasi langsung HBP-LMMD tidak diteruskan ke
multiseed. Hasilnya dilaporkan sebagai interaksi negatif, bukan disembunyikan
dengan pemilihan seed atau hyperparameter setelah melihat test set.
