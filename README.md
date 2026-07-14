# MobileNetV3–HBP–LMMD

Kerangka eksperimen PyTorch untuk klasifikasi fine-grained dengan **closed-set
unsupervised domain adaptation (UDA)**. MobileNetV3 adalah anchor utama,
MobileNetV4 dan backbone lain dipakai sebagai kontrol, HBP menguji kontribusi
interaksi fitur antarlapis, dan LMMD menyelaraskan distribusi source–target per
kelas.

Dataset source yang digunakan adalah [Coffee Green Bean with 17 Defects
(original)](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original)
dan referensi baseline-nya adalah Arwatchananukul et al. (2024), DOI
[10.1016/j.atech.2024.100680](https://doi.org/10.1016/j.atech.2024.100680).
Catatan protokol rinci tersedia di [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).

## Desain eksperimen

Tahap A menguji backbone dengan protokol yang sama:

| Kode | Konfigurasi |
|---|---|
| B0 | MobileNetV3-Large + GAP |
| B1 | MobileNetV3-Small + GAP |
| B2 | MobileNetV4-Conv-Small + GAP |
| B3 | Swin-T + GAP |
| B4 | EfficientNet-B0 + GAP |

Tahap B menguji kontribusi metode:

| Kode | Konfigurasi | Pertanyaan yang diuji |
|---|---|---|
| M0 | MobileNetV3 + GAP, source-only | baseline |
| M0b | MobileNetV3 + factorized bilinear, source-only | kontrol orde kedua satu lapis |
| M1 | MobileNetV3 + HBP, source-only | kontribusi HBP |
| M2 | MobileNetV3 + GAP + MMD | alignment global |
| M3 | MobileNetV3 + GAP + LMMD | alignment class-wise |
| M4 | MobileNetV3 + HBP + DANN | baseline adversarial |
| M5 | MobileNetV3 + HBP + LMMD | model usulan |

Model final tidak dikunci sebelum B0–B4 dibandingkan berdasarkan target
macro-F1, parameter, ukuran FP32, latency, dan memori pada perangkat yang sama.

## Struktur dataset

Nama folder kelas harus identik di keempat split. Label pada `target/train`
hanya diperlukan oleh `ImageFolder` untuk menentukan struktur kelas dan **tidak
pernah dipakai oleh loss training**.

```text
data/coffee/
├── source/
│   ├── train/
│   │   ├── class_01/*.jpg
│   │   └── ...
│   └── val/
│       ├── class_01/*.jpg
│       └── ...
└── target/
    ├── train/
    │   ├── class_01/*.jpg
    │   └── ...
    └── val/
        ├── class_01/*.jpg
        └── ...
```

Untuk evaluasi UDA yang sah, label `target/val` hanya boleh dipakai saat
validasi/evaluasi, bukan untuk pemilihan pseudo-label atau update bobot model.
Jika target benar-benar belum tersortir, buat symlink/copy terstruktur untuk
loader atau ganti loader dengan pembaca manifest; jangan memasukkan label target
ke loss.

### Menyiapkan dataset Kaggle

Dataset publik berisi 979 file di 17 folder kelas. Perintah berikut mengunduh,
memverifikasi jumlah setiap kelas terhadap Table 1 paper, lalu membuat split
stratified 70/20/10 dengan seed 42:

```powershell
python -m bilinear_lmmd.prepare_coffee17 --output data/coffee
```

Berbeda dari paper, rotasi dilakukan **sesudah split dan hanya secara online
pada train**. Ini mencegah satu biji asli masuk ke train sementara versi
rotasinya masuk ke validation/test. Sudut default tetap mengikuti paper:
0°, 45°, 90°, 135°, 180°, 225°, dan 270°.

Setelah proses ini, baseline `source_only` B0-B4 dan M0-M1 dapat langsung
dijalankan. M2-M5 tetap membutuhkan `data/coffee/target/train` dan
`data/coffee/target/val` dari domain pengambilan gambar lain.

## Instalasi dan training

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m bilinear_lmmd.prepare_coffee17 --output data/coffee
python -m bilinear_lmmd.train --config configs/B0_mobilenetv3_large_gap.yaml
```

Jika target domain belum tersedia, mulai dari baseline source-only:

```powershell
python -m bilinear_lmmd.train --config configs/B0_mobilenetv3_large_gap.yaml
```

Sesudah target domain tersedia dan disusun dengan 17 kelas yang sama, model
usulan dapat dijalankan dengan:

```powershell
python -m bilinear_lmmd.train --config configs/M5_mobilenetv3_hbp_lmmd.yaml
```

Ubah `data.root`, `model.num_classes`, batch size, dan hyperparameter lain pada
YAML. Semua nilai yang tidak ditulis di YAML mengambil default dari
`src/bilinear_lmmd/config.py`.

Contoh override dataset untuk 17 kelas:

```yaml
data:
  root: D:/dataset/coffee
  batch_size: 32
  workers: 4
model:
  num_classes: 17
```

## Mengukur efisiensi

Jalankan pada perangkat dan kondisi yang sama untuk setiap B0–B4:

```powershell
python -m bilinear_lmmd.benchmark --config configs/B0_mobilenetv3_large_gap.yaml
python -m bilinear_lmmd.benchmark --config configs/B2_mobilenetv4_conv_small_gap.yaml
```

Output berisi jumlah parameter, estimasi ukuran bobot FP32, serta latency batch
1. Pengukuran memori GPU sebaiknya ditambahkan saat eksperimen dijalankan pada
CUDA karena lingkungan CPU tidak memberikan peak CUDA memory.

## Urutan eksperimen yang disarankan

1. Jalankan B0–B4 minimal tiga seed dengan split, augmentasi, resolusi,
   optimizer, dan scheduler yang sama.
2. Pilih lightweight backbone memakai target macro-F1 **bersama** latency dan
   ukuran model. MobileNetV3 tetap anchor; MobileNetV4 boleh menang secara
   empiris.
3. Jalankan M0–M5 pada backbone terpilih. Jika backbone berubah, ganti hanya
   `model.backbone` dan cek `out_indices`; kontribusi HBP–LMMD tetap sama.
4. Uji `projection_dim` HBP, misalnya 256, 512, 1024. Laporkan trade-off, bukan
   hanya skor terbaik.
5. Setelah model dan ablation stabil, tambahkan Finer-CAM berbasis LayerCAM
   sebagai modul analisis kelas-pembanding. XAI tidak boleh memengaruhi training.

## Membuktikan kontribusi HBP

Analisis angka paper dan kriteria keputusan tersedia di
[docs/HBP_HYPOTHESIS.md](docs/HBP_HYPOTHESIS.md). Eksperimen minimum harus
membandingkan GAP, bilinear satu lapis, dan HBP pada split serta seed identik.
Notebook siap-Colab tersedia di
[notebooks/hbp_ablation_colab.ipynb](notebooks/hbp_ablation_colab.ipynb).

Contoh satu seed:

```powershell
python -m bilinear_lmmd.train --config configs/M0_mobilenetv3_gap_source.yaml --seed 42 --output-dir outputs/M0_seed42
python -m bilinear_lmmd.train --config configs/M0b_mobilenetv3_bilinear_source.yaml --seed 42 --output-dir outputs/M0b_seed42
python -m bilinear_lmmd.train --config configs/M1_mobilenetv3_hbp_source.yaml --seed 42 --output-dir outputs/M1_seed42

python -m bilinear_lmmd.evaluate_checkpoint --checkpoint outputs/M0_seed42/best.pt --domain source --split test --output-dir reports/M0_seed42
python -m bilinear_lmmd.evaluate_checkpoint --checkpoint outputs/M1_seed42/best.pt --domain source --split test --output-dir reports/M1_seed42

python -m bilinear_lmmd.compare_reports --baseline reports/M0_seed42/metrics.json --candidate reports/M1_seed42/metrics.json --output reports/M0_vs_M1_seed42.json
```

Ulangi minimal untuk seed 42, 123, dan 2026. Kesimpulan tidak boleh diambil dari
satu run terbaik.

Setelah ketiga seed dievaluasi, agregasikan secara berpasangan:

```powershell
python -m bilinear_lmmd.aggregate_ablation `
  --baseline reports/M0_seed42/metrics.json reports/M0_seed123/metrics.json reports/M0_seed2026/metrics.json `
  --candidate reports/M1_seed42/metrics.json reports/M1_seed123/metrics.json reports/M1_seed2026/metrics.json `
  --output reports/M0_vs_M1_aggregate.json
```

## Konfirmasi clean grouped 5-fold

Screening holdout memakai test set kecil. Konfirmasi utama membentuk lima fold
dari 979 citra asli. Setiap identitas menjadi test tepat satu kali, sedangkan
augmentasi tetap hanya diterapkan oleh loader train.

Siapkan fold:

```powershell
python -m bilinear_lmmd.prepare_grouped_folds `
  --source-root data/coffee/source `
  --output-root data/coffee_5fold
```

Jalankan GAP dan HBP dengan satu perintah yang aman dijalankan ulang setelah
interupsi:

```powershell
python -u -m bilinear_lmmd.run_grouped_cv `
  --data-root data/coffee_5fold `
  --output-root outputs/grouped5fold `
  --models M0 M1 `
  --seed 42
```

Fold yang sudah menyelesaikan 50 epoch dan evaluasi akan dilewati otomatis.
Setelah lima fold selesai, prediksi test digabung menjadi 979 out-of-fold
predictions di `outputs/grouped5fold/oof/`.

## Catatan implementasi HBP dan LMMD

HBP memproyeksikan tiga feature map ke dimensi yang sama, menyamakan ukuran
spasial, lalu membentuk interaksi multiplikatif untuk pasangan 1–2, 1–3, dan
2–3. Setiap pasangan melalui signed square-root dan normalisasi L2 sebelum
digabungkan.

LMMD memakai one-hot label source dan probabilitas softmax target yang dilepas
dari graph sebagai bobot pseudo-label. Kernel RBF kemudian menghitung discrepancy
per kelas yang aktif pada kedua batch. `warmup_epochs` mencegah pseudo-label acak
awal langsung mendominasi training.
