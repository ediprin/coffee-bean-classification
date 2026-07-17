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
Hasil test M0/M1 tiga seed yang telah dikunci, keputusan model, batas klaim,
dan benchmark efisiensi dicatat di
[docs/FINAL_HBP_RESULTS.md](docs/FINAL_HBP_RESULTS.md).

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
| M1s | MobileNetV3 + spatially preserved HBP 14x14 + CE | preservasi detail spasial tanpa tambahan parameter |
| E1 | MobileNetV3 + HBP global/local MoE + CE | komplementaritas konteks global dan detail lokal |
| M1c | MobileNetV3 + HBP + nonlinear projection | kontrol kapasitas fusion |
| M1f | MobileNetV3 + GAP-HBP feature fusion | komplementaritas orde pertama-kedua |
| M1rc | HBP utuh + auxiliary HBP kecil | kontrol kapasitas residual fusion |
| M1r | HBP utuh + residual GAP kecil | fusion tanpa kompresi HBP |
| M2 | MobileNetV3 + GAP + MMD | alignment global |
| M3 | MobileNetV3 + GAP + LMMD | alignment class-wise |
| M4 | MobileNetV3 + HBP + DANN | baseline adversarial |
| M5 | MobileNetV3 + HBP + LMMD | model usulan |
| M5w01 | M5 dengan bobot LMMD 0,1 | rescue control untuk source degradation |

Eksperimen lintas-dataset USK-Coffee memakai kode U0-U3; lihat bagian
"Benchmark USK-Coffee" dan `docs/USK_COFFEE_PROTOCOL.md`.

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

Jika dataset sudah ditambahkan sebagai Kaggle Input dan telah diekstrak,
gunakan folder mount secara langsung tanpa download ulang:

```powershell
python -u -m bilinear_lmmd.prepare_coffee17 `
  --raw-root /kaggle/input `
  --output data/coffee
```

Hanya folder 17 kelas Coffee17 yang dikumpulkan; dataset lain di bawah
`/kaggle/input` diabaikan. Jumlah setiap kelas tetap harus persis sesuai paper.

Berbeda dari paper, rotasi dilakukan **sesudah split dan hanya secara online
pada train**. Ini mencegah satu biji asli masuk ke train sementara versi
rotasinya masuk ke validation/test. Sudut default tetap mengikuti paper:
0°, 45°, 90°, 135°, 180°, 225°, dan 270°.

Setelah proses ini, baseline `source_only` B0-B4 dan M0-M1 dapat langsung
dijalankan. M2-M5 tetap membutuhkan `data/coffee/target/train` dan
`data/coffee/target/val` dari domain pengambilan gambar lain.

### Benchmark USK-Coffee

USK-Coffee diperlakukan sebagai task empat kelas terpisah, bukan digabungkan
dengan label Coffee17. Runner menemukan folder kelas secara rekursif,
mempertahankan split arsip jika ada, menghapus exact duplicate, dan mengaudit
pasangan sisi biji berdasarkan filename.

Screening awal membandingkan MobileNetV2 paper, MobileNetV3-GAP, dan
MobileNetV3-HBP pada validation seed 42:

```powershell
python -u -m bilinear_lmmd.run_usk_screening `
  --raw-root /kaggle/input/usk-coffee `
  --data-root /kaggle/working/usk-coffee-prepared `
  --output-root /kaggle/working/usk-results `
  --stage quick `
  --seeds 42 `
  --evaluation-split val
```

Perintah yang sama dapat dijalankan ulang; dataset, training lengkap, dan
report lengkap akan dilewati. Jika audit mendeteksi pasangan depan-belakang
lintas split, runner berhenti sebelum training. Batas perbandingan terhadap
test accuracy paper 81,31% dan protokol konfirmasi tersedia di
[docs/USK_COFFEE_PROTOCOL.md](docs/USK_COFFEE_PROTOCOL.md).

### Screening HBP pada dataset coffee roast

Dataset publik tambahan dari Jiao et al. memiliki empat kelas roast-level,
bukan kelas cacat. Uji minimal R0 (MobileNetV3-GAP) versus R1
(MobileNetV3-HBP) dijalankan dengan:

```powershell
python -u -m bilinear_lmmd.run_roast_hbp_screening `
  --raw-root /kaggle/input/coffee-bean-dataset-resized-224-x-224 `
  --data-root /kaggle/working/coffee-roast-prepared `
  --output-root /kaggle/working/coffee-roast-hbp-results `
  --seeds 42 `
  --evaluation-split val
```

Runner mempertahankan test bawaan, membuat validation hanya dari train bila
perlu, serta membuang salinan exact duplicate lintas split dengan prioritas
mempertahankan test. Protokol dan
batas klaim tersedia di
[docs/ROAST_HBP_PROTOCOL.md](docs/ROAST_HBP_PROTOCOL.md).

### Screening hierarchical HBP pada Coffee17

H1 mempertahankan MobileNetV3-HBP dan classifier fine 17 kelas milik M1, lalu
menambahkan auxiliary parent loss untuk pasangan Black, Sour, dan Insect
Damage. Screening terkontrol dijalankan pada validation:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee_clean_for_synthetic/folds/fold_1 `
  --output-root outputs/hierarchical-hbp `
  --stage hierarchy `
  --seeds 42 `
  --evaluation-split val
```

Pemetaan kelas, bobot loss yang dikunci, dan kriteria keputusan tersedia di
[docs/HIERARCHICAL_HBP_PROTOCOL.md](docs/HIERARCHICAL_HBP_PROTOCOL.md).

### Screening SPPF-Attention sebelum HBP

Adaptasi terkontrol dari Hong et al. (2026) memperbaiki feature map terdalam
MobileNetV3 dengan pooling multi-skala serta channel-spatial attention sebelum
HBP. Jalankan M1 versus S1 pada validation:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee17_hierarchy_clean/folds/fold_1 `
  --output-root outputs/sppf-attention-hbp `
  --stage sppf `
  --seeds 42 `
  --evaluation-split val
```

Formulasi modul, perbedaan terhadap detector Hong, dan kriteria penghentian
tersedia di
[docs/SPPF_ATTENTION_HBP_PROTOCOL.md](docs/SPPF_ATTENTION_HBP_PROTOCOL.md).

Untuk membedakan efek SPPF dari efek HBP, jalankan ablasi faktorial
M0/M1/S0/S1. S0 menempatkan SPPF-Attention sebelum GAP tanpa HBP:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee17_hierarchy_clean/folds/fold_1 `
  --output-root outputs/sppf-attention-hbp `
  --stage sppf_factorial `
  --seeds 42 123 2026 `
  --evaluation-split val
```

Jika S1 lolos screening terhadap M1, isolasi efek kapasitas dengan:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee17_hierarchy_clean/folds/fold_1 `
  --output-root outputs/sppf-attention-hbp `
  --stage sppf_control `
  --seeds 42 `
  --evaluation-split val
```

Runner memakai ulang M1/S1 yang sudah lengkap dan hanya melatih C1, yaitu
pointwise residual control dengan parameter yang cocok terhadap S1.

### Benchmark domain sintetis terkontrol

Jika domain target nyata belum tersedia, pipeline dapat diuji dengan empat
shift sintetis: illumination, sensor-quality, background, dan gabungan ketiganya.
Generator mempertahankan split asli, membuat transformasi deterministik, dan
menulis resep per gambar agar eksperimen dapat diaudit.

Screening hemat komputasi memakai domain `combined` dan satu seed:

```powershell
python -u -m bilinear_lmmd.run_synthetic_benchmark `
  --source-root data/coffee_clean/folds/fold_1/source `
  --data-root data/coffee_synthetic `
  --output-root outputs/synthetic_screen `
  --domains combined `
  --models M0 M1 M2 M3 M5 `
  --seeds 123 `
  --source-checkpoints `
    M0:123=/kaggle/working/finegrained-results/outputs/M0_seed123/best.pt `
    M1:123=/kaggle/working/finegrained-results/outputs/M1_seed123/best.pt
```

Perintah aman dilanjutkan setelah interupsi dan menampilkan ringkasan GAP/HBP,
MMD, serta LMMD. Hasil berada di `reports/summary.json` dan `summary.csv`.
`--source-checkpoints` bersifat opsional dan mencegah training ulang baseline
source-only yang seed serta arsitekturnya sudah cocok.
Eksperimen ini hanya mendukung klaim **controlled synthetic robustness/UDA
sanity-check**, bukan ketahanan dunia nyata. Protokol lengkap dan perintah
konfirmasi empat domain x tiga seed tersedia di
[docs/SYNTHETIC_DOMAIN_PROTOCOL.md](docs/SYNTHETIC_DOMAIN_PROTOCOL.md).

Setelah rescue control M5w01 lulus pada seed screening 123, konfirmasi tanpa
tuning tambahan pada held-out seeds 42 dan 2026 dijalankan dengan:

```powershell
python -u -m bilinear_lmmd.run_lmmd_rescue_confirmation `
  --data-root data/coffee_synthetic_components/illumination `
  --output-root outputs/lmmd_rescue_confirmation `
  --seeds 42 2026
```

Jika konfirmasi illumination lulus, uji konfigurasi yang sudah dibekukan pada
sensor dan background dengan reuse checkpoint M1 yang diverifikasi melalui
fingerprint source:

```powershell
python -u -m bilinear_lmmd.run_lmmd_cross_shift_confirmation `
  --data-root data/coffee_synthetic_components `
  --baseline-output-root outputs/lmmd_rescue_confirmation `
  --output-root outputs/lmmd_cross_shift_confirmation `
  --domains sensor background `
  --seeds 42 2026
```

### Diagnosis XAI setelah konfirmasi

Setelah semua checkpoint dan `predictions.csv` tersedia, bandingkan M1 dengan
M5w01 memakai multilayer LayerCAM dan Finer-CAM. Runner ini tidak melakukan
training baru, dapat dilanjutkan setelah interupsi, serta memeriksa target gain
dan source forgetting pada sampel `rescued`, `negative_transfer`,
`both_correct`, dan `both_wrong`.

```bash
python -u -m bilinear_lmmd.run_xai_analysis \
  --data-root /kaggle/working/coffee-synthetic-components \
  --illumination-root /kaggle/working/lmmd-rescue-confirmation \
  --cross-shift-root /kaggle/working/lmmd-cross-shift-confirmation \
  --output-root /kaggle/working/xai-results \
  --domains illumination sensor background \
  --evaluation-domains target source \
  --seeds 42 2026
```

Output mencakup panel heatmap, foreground-attention terhadap mask biji,
background leakage, dan relative confidence drop setelah penghapusan 5% piksel
terkuat. Protokol dan batas interpretasi tersedia di
[docs/XAI_PROTOCOL.md](docs/XAI_PROTOCOL.md).

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

Setelah report test M0/M1 tiga seed terkunci, buat paket pelaporan final tanpa
melatih ulang atau mengubah checkpoint:

```powershell
python -u -m bilinear_lmmd.run_final_hbp_report `
  --report-root outputs/hierarchical-hbp-results/reports `
  --output-dir outputs/hierarchical-hbp-results/final_report `
  --seeds 42 123 2026
```

Runner menulis `final_summary.json`, `per_seed.csv`, `per_class.csv`, dan
`FINAL_HBP_REPORT.md`, serta mengukur parameter, ukuran FP32, dan latency batch-1
M0/M1 pada perangkat yang sama. Benchmark memakai bobot acak karena efisiensi
arsitektur tidak bergantung pada nilai checkpoint dan tidak mengakses test data.

Visualisasikan bukti kelas model final M0/M1 dengan raw LayerCAM heatmap,
overlay, dan Finer-LayerCAM. Sampel dipilih deterministik berdasarkan outcome
prediksi, bukan dipilih manual setelah heatmap terlihat:

```powershell
python -u -m bilinear_lmmd.run_final_hbp_xai `
  --data-root data/coffee17_hierarchy_clean/folds/fold_1 `
  --experiment-root outputs/hierarchical-hbp-results `
  --output-root outputs/final-hbp-xai `
  --seeds 42 `
  --samples-per-outcome 1
```

Output utama `gallery_seed42.png` menampilkan input, heatmap mentah, dan overlay
M0/M1. XAI adalah diagnosis post-hoc pada sampel terpilih dan bukan bukti kausal
atau estimasi seluruh populasi test.

### Screening stabilisasi HBP dengan EMA

M1e mempertahankan arsitektur, loss, optimizer, dan inferensi M1. EMA decay
`0.995` dimulai setelah lima epoch penuh dan hanya mengubah bobot checkpoint
yang dievaluasi. Gunakan fold baru; test final fold 1 tidak boleh dipakai lagi:

```powershell
python -u -m bilinear_lmmd.run_ema_screening `
  --data-root data/coffee17_hierarchy_clean/folds/fold_2 `
  --output-root outputs/hbp-ema-fold2 `
  --seeds 42 123 2026 `
  --evaluation-split val
```

Satu trajectory per seed menghasilkan `best_raw.pt` (M1) dan `best.pt` (M1e),
sehingga waktu training hampir separuh dan perbandingan bobot benar-benar
berpasangan.

Kriteria keputusan dan batas test leakage tersedia di
[docs/EMA_HBP_PROTOCOL.md](docs/EMA_HBP_PROTOCOL.md).

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

### Screening feature fusion tiga seed

Setelah M1 tersedia untuk seed 42, 123, dan 2026, jalankan kontrol kapasitas
M1c dan feature fusion M1f dengan satu perintah:

```powershell
python -u -m bilinear_lmmd.run_fusion_screening `
  --data-root data/coffee `
  --output-root outputs/holdout
```

Runner aman dijalankan ulang: training dan evaluasi yang sudah lengkap akan
dilewati. M1c dan M1f memiliki jumlah parameter yang berbeda kurang dari 0,1%,
sehingga perbandingan M1c-vs-M1f mengisolasi manfaat fitur GAP dari sekadar
tambahan kapasitas. Setelah enam run selesai, agregasi M1-vs-M1c,
M1c-vs-M1f, dan M1-vs-M1f dicetak otomatis.

Untuk stress test residual fusion pada seed 123 saja:

```powershell
python -u -m bilinear_lmmd.run_fusion_screening `
  --data-root data/coffee `
  --output-root outputs/holdout `
  --models M1rc M1r `
  --seeds 123
```

M1r mempertahankan seluruh embedding HBP 1536-D dan hanya menambahkan residual
GAP 128-D. M1rc mempertahankan HBP yang sama dan menambahkan auxiliary HBP 80-D;
jumlah parameter M1rc dan M1r berbeda kurang dari 0,1%.

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

### Ensemble probabilitas GAP-HBP tanpa training ulang

Setelah grouped 5-fold M0 dan M1 lengkap, evaluasi ensemble dengan checkpoint
yang sama. Untuk setiap fold, bobot HBP (`alpha`) dipilih hanya dari validation
set, kemudian diterapkan ke outer test fold. Label OOF tidak digunakan untuk
memilih bobot.

```powershell
python -u -m bilinear_lmmd.run_oof_ensemble `
  --data-root data/coffee_5fold `
  --output-root outputs/grouped5fold `
  --seed 42
```

Definisi ensemble adalah
`p = (1 - alpha) * p_GAP + alpha * p_HBP`. Perintah ini hanya melakukan
inference; model tidak dilatih ulang. Progress ditampilkan untuk validation dan
test setiap fold. Ringkasan akhir tersimpan di
`outputs/grouped5fold/oof/M0_M1_ensemble_seed42/metrics.json`, sedangkan kurva
alpha dan metrik lengkap tersimpan di `comparison.json`.

## Ablasi ciri warna, bentuk, dan tekstur (CPU)

Runner berikut menguji seluruh tujuh kombinasi non-kosong: warna (`C`), bentuk
(`S`), tekstur (`T`), `CS`, `CT`, `ST`, dan `CST`. Mask biji dibuat otomatis,
lalu warna diringkas di ruang LAB/HSV, bentuk memakai properti region dan Hu
moments, sedangkan tekstur memakai masked GLCM dan LBP. Setiap kombinasi memakai
RBF-SVM; `C` dan `gamma` dipilih hanya dari validation fold.

```powershell
python -u -m bilinear_lmmd.run_attribute_ablation `
  --data-root data/coffee_5fold `
  --output-root outputs/attribute_ablation
```

Ekstraksi fitur hanya dilakukan sekali dan disimpan dalam cache. Kombinasi yang
sudah selesai juga dilewati saat runner dijalankan ulang. Periksa
`mask_audit.png` sebelum menafsirkan metrik. Hasil utama berada di
`summary.json`; setiap kombinasi juga menyimpan `metrics.json` dan 979 prediksi
OOF di `predictions.csv`.

Jika prediksi OOF HBP tersedia, tambahkan
`--hbp-predictions path/to/HBP_predictions.csv`. Runner akan menghitung berapa
sampel yang hanya benar oleh model atribut atau hanya benar oleh HBP. Audit ini
menentukan apakah eksperimen hybrid layak dilakukan; ia bukan ensemble yang
memakai label test.

### Screening disagreement gate HBP-CST

Jika audit menemukan prediksi CST yang benar ketika HBP salah, screening gate
berbasis pasangan label dapat dijalankan tanpa training ulang backbone:

```powershell
python -u -m bilinear_lmmd.run_disagreement_gate `
  --attribute-predictions outputs/attribute_ablation/CST/predictions.csv `
  --hbp-predictions outputs/grouped5fold/oof/M1_seed42/predictions.csv `
  --output-dir outputs/attribute_ablation/HBP_CST_gate
```

Untuk setiap outer fold, gate hanya belajar dari empat fold lainnya. Nilai
`min_support` dan margin keuntungan CST dipilih dengan inner leave-one-fold
validation. Pasangan yang jarang, seri, atau tidak dikenal selalu kembali ke
HBP; kandidat HBP murni juga tersedia agar validation tidak dipaksa memakai
CST. Ini tetap screening meta-model atas hard prediction OOF. Klaim final
memerlukan kalibrasi gate dari validation prediction model dasar atau test set
independen.

## Screening LGF-CBAM tanpa HBP

Eksperimen attention dari Techie-Menson et al. (2026) diuji sebagai model yang
berdiri sendiri: feature map terdalam MobileNetV3 diproses oleh channel dan
spatial attention, lalu GAP dan classifier. HBP tidak digunakan.

| Kode | Model |
|---|---|
| M0 | MobileNetV3 + GAP |
| M0a | MobileNetV3 + fixed 50:50 channel/spatial fusion + GAP |
| M0lgf | MobileNetV3 + learnable gated channel/spatial fusion + GAP |

Jalankan screening seed 123:

```powershell
python -u -m bilinear_lmmd.run_attention_screening `
  --data-root data/coffee `
  --output-root outputs/attention_screen `
  --seeds 123
```

Runner aman dilanjutkan setelah interupsi. Perbandingan primer adalah M0lgf vs
M0a, karena keduanya memiliki jalur attention yang sama. Gate LGF diinisialisasi
50:50 sehingga kedua model memulai dari fungsi fusion yang identik. Detail
operasional dan keterbatasan paper dicatat di
`docs/LGF_CBAM_HYPOTHESIS.md`.

## Membersihkan exact duplicate sebelum grouped CV

Dataset publik berisi exact duplicate satu kelas serta satu hash gambar yang
memiliki dua label berbeda. Buat salinan bersih tanpa mengubah data asli:

```powershell
python -u -m bilinear_lmmd.prepare_clean_grouped_folds `
  --source-root data/coffee/source `
  --output-root data/coffee_clean
```

Kebijakan konservatif runner:

- satu file canonical dipertahankan untuk exact duplicate dengan label sama;
- semua file byte-identik dengan label berbeda masuk karantina;
- near-duplicate tidak dihapus otomatis;
- 5-fold baru dibuat dari 965 hash unik.

Audit tersimpan di `data/coffee_clean/audit.json`, konflik label berada di
`quarantine_label_conflicts/`, dan fold siap training berada di `folds/`.
Konfirmasi ulang GAP dan HBP dengan:

```powershell
python -u -m bilinear_lmmd.run_grouped_cv `
  --data-root data/coffee_clean/folds `
  --output-root outputs/grouped5fold_clean `
  --models M0 M1 `
  --seed 42 `
  --expected-count 965
```

## Ablation object-centric crop

Preprocessing ala studi Jiao et al. diuji tanpa memasukkan mask ke CNN. Sistem
mencari largest component biji pada background terang, mengambil crop RGB
dengan margin 10%, lalu menjalankan pipeline training/evaluasi biasa.

Factorial ablation membandingkan M0/M1 pada gambar asli dengan O0/O1 pada crop:

```bash
python -u -m bilinear_lmmd.run_finegrained_screening \
  --data-root DATA_ROOT_SATU_FOLD \
  --output-root outputs/object_crop \
  --stage object_crop \
  --seeds 123 \
  --evaluation-split val
```

Konfigurasi preprocessing tersimpan di checkpoint sehingga inference dan
evaluasi tidak membutuhkan mask manual. Definisi operasional, kontrol
factorial, dan kriteria keputusan ada di
[`docs/OBJECT_CROP_PROTOCOL.md`](docs/OBJECT_CROP_PROTOCOL.md).

## Reproduksi paper Arwatchananukul et al.

P0 dan P1 membandingkan GAP dengan HBP+CE pada protokol paper yang membuat enam
rotasi sebelum split 70/20/10. Runner mempertahankan kelemahan tersebut hanya
untuk keterbandingan dan mencetak audit identity leakage secara eksplisit:

```bash
python -u -m bilinear_lmmd.run_paper_reproduction \
  --raw-root RAW_COFFEE17 \
  --data-root data/coffee17_paper_protocol \
  --output-root outputs/paper_reproduction \
  --seeds 42
```

Hasil P0/P1 bukan pengganti hasil clean M0/M1. Asumsi yang tidak dilaporkan
paper dan aturan interpretasi dicatat di
[`docs/PAPER_REPRODUCTION_PROTOCOL.md`](docs/PAPER_REPRODUCTION_PROTOCOL.md).

Hasil test terkunci tiga seed menunjukkan P1 mengungguli P0 pada seluruh seed:
Accuracy +6,41 ± 2,26 poin, Macro-F1 +6,61 ± 2,33 poin, Hard-F1 +7,58 ± 2,59
poin, dan Worst-class F1 +15,98 ± 6,13 poin. Karena variant rotasi dibuat
sebelum split, angka ini hanya bukti keterbandingan paper-style. Hasil clean
grouped M0/M1 tetap bukti utama. Ringkasan lengkap tersedia di
[`docs/PAPER_REPRODUCTION_PROTOCOL.md`](docs/PAPER_REPRODUCTION_PROTOCOL.md)
dan [`docs/PAPER_REPRODUCTION_RESULTS.json`](docs/PAPER_REPRODUCTION_RESULTS.json).

## Benchmark Roboflow CBD-Multiclassify

Dataset publik `asdasd-zsar1/cbd-multiclassify` dipakai sebagai benchmark
terpisah delapan kelas untuk menguji generalitas efek HBP. Runner mengeluarkan
`Unlabeled` serta satu stray label `Insect`, melakukan exact deduplication, dan
membuat split baru 60/20/20
secara identity-grouped berdasarkan nama asli sebelum suffix `.rf.<hash>`:

```bash
python -u -m bilinear_lmmd.run_cbd_multiclassify_screening \
  --raw-root RAW_CBD_MULTICLASSIFY \
  --data-root data/cbd_multiclassify_prepared \
  --output-root outputs/cbd_hbp \
  --seeds 42 \
  --evaluation-split val
```

CBD0/CBD1 membandingkan GAP dan HBP dengan CE. CBD2/CBD3 mengulang faktorial
head yang sama memakai Balanced Softmax untuk menguji apakah imbalance menutupi
efek HBP. Dataset ini tidak dicampur dengan Coffee-17 karena labelnya lebih
kasar. Protokol dan batas klaim lengkap ada di
[`docs/CBD_MULTICLASSIFY_PROTOCOL.md`](docs/CBD_MULTICLASSIFY_PROTOCOL.md).

Konfirmasi tiga seed untuk logistic stacking GAP-HBP dapat dilanjutkan tanpa
mengulang run yang sudah lengkap:

```bash
python -u -m bilinear_lmmd.run_cbd_stacking_confirmation \
  --data-root data/cbd_multiclassify_prepared \
  --output-root outputs/cbd_hbp \
  --seeds 42 123 2026
```

Setelah stacking tiga seed berstatus PASS, distilasi teacher GAP-HBP ke satu
student GAP dapat disaring pada seed 42. CBD4 adalah kontrol KD dari GAP yang
terkalibrasi, sedangkan CBD5 menerima teacher stacking:

```bash
python -u -m bilinear_lmmd.run_cbd_kd_confirmation \
  --data-root data/cbd_multiclassify_prepared \
  --output-root outputs/cbd_hbp \
  --seeds 42
```

Jika hasil screening layak, ubah seed menjadi `42 123 2026`; artefak lengkap
akan dilewati. Student hasil KD tetap satu MobileNetV3-GAP saat inference.
Rancangan, kontrol kalibrasi, dan kriteria keputusan tercatat di
[`docs/CBD_MULTICLASSIFY_PROTOCOL.md`](docs/CBD_MULTICLASSIFY_PROTOCOL.md).

## Screening preservasi spasial HBP

M1 memakai endpoint MobileNetV3 berukuran 56 x 56, 14 x 14, dan 7 x 7 pada
input 224, lalu menyelaraskan semuanya ke grid terdalam 7 x 7. M1s adalah
ablasi terkendali yang menyelaraskan ketiganya ke grid tengah 14 x 14 sebelum
interaksi bilinear. M1s mempertahankan embedding 1536-D dan jumlah parameter
M1; biaya interaksi spasialnya lebih besar.

Jalankan screening M1 vs M1s pada fold dan seed yang sama:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee_clean/folds/fold_1 `
  --output-root outputs/sp_hbp_screen `
  --stage spatial `
  --seeds 42 `
  --evaluation-split val
```

Runner dapat dilanjutkan dengan perintah yang sama setelah interupsi. Batas
klaim, kontrol eksperimen, dan kriteria keputusan tercatat di
`docs/SP_HBP_PROTOCOL.md`. SP-HBP 14 x 14 adalah modifikasi penelitian ini;
interaksi HBP lintas-layer mengacu pada Yu et al. (ECCV 2018).

Jika kriteria validation lolos, jalankan seed konfirmasi dan evaluasi test:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee_clean/folds/fold_1 `
  --output-root outputs/sp_hbp_screen `
  --stage spatial `
  --seeds 42 123 2026 `
  --evaluation-split test
```

## Screening global-local HBP mixture-of-experts

E1 mempertahankan HBP M1 sebagai expert global dan menambahkan expert lokal
ringan dari feature map tengah 14 x 14. Gate per sampel menggabungkan logit
keduanya dan mulai dari bobot 0,8/0,2 untuk global/lokal. Implementasi ini
terinspirasi MGE-CNN (Zhang et al., ICCV 2019), tetapi merupakan adaptasi
ringan dengan satu shared backbone dan tanpa Grad-CAM crop; bukan reproduksi
arsitektur paper tersebut.

Screening validation M1 vs E1:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee_clean/folds/fold_1 `
  --output-root outputs/hbp_moe_screen `
  --stage moe `
  --seeds 42 `
  --evaluation-split val
```

`metrics.json` E1 memuat mean gate, fraksi pemilihan expert, dan entropy gate;
`history.json` memuat loss tiap expert. Protokol dan batas klaim lengkap ada di
`docs/HBP_MOE_PROTOCOL.md`.

## Screening fine-grained: resolusi, HBP, dan ArcFace

Eksperimen utama memakai klasifikasi langsung 17 kelas. ArcFace hanya memberi
angular margin saat training; evaluasi tetap menghasilkan satu label dari 17
kelas, bukan klasifikasi pasangan ala Siamese.

Mulai dari satu clean fold dan seed 123 untuk memilih resolusi:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee_clean/folds/fold_1 `
  --output-root outputs/finegrained_screen `
  --stage resolution `
  --seeds 123
```

Tahap ini membandingkan M1 (HBP 224 + CE) dengan F1 (HBP 320 + CE). Pada clean
fold 1 seed 123, resolusi 320 menurunkan Macro-F1 sebesar 3,49 poin,
Hard-F1 4,40 poin, dan Worst-F1 6,06 poin. Karena itu eksperimen utama diteruskan
pada 224 dengan matrix GAP/HBP x CE/ArcFace:

```powershell
python -u -m bilinear_lmmd.run_finegrained_screening `
  --data-root data/coffee_clean/folds/fold_1 `
  --output-root outputs/finegrained_screen `
  --stage arcface224 `
  --seeds 123
```

Kode A2 adalah GAP 224 + ArcFace dan A3 adalah HBP 224 + ArcFace. M0 dan M1
akan dilewati otomatis jika hasil lengkapnya sudah berada pada output root yang
sama.

Runner menampilkan progres epoch langsung, melewati hasil yang sudah lengkap,
dan meneruskan training dari `last.pt` setelah interupsi untuk checkpoint baru.
Setelah maksimal dua kandidat dipilih, jalankan grouped OOF bersih, misalnya:

```powershell
python -u -m bilinear_lmmd.run_grouped_cv `
  --data-root data/coffee_clean/folds `
  --output-root outputs/finegrained_grouped5fold `
  --models M1 A3 `
  --seed 123 `
  --expected-count 965
```

Ganti `M1 A3` dengan dua kandidat yang benar-benar menang screening. Rancangan
hipotesis dan aturan keputusan dicatat di
`docs/FINEGRAINED_HBP_ARCFACE.md`.

## Catatan implementasi HBP dan LMMD

HBP memproyeksikan tiga feature map ke dimensi yang sama, menyamakan ukuran
spasial, lalu membentuk interaksi multiplikatif untuk pasangan 1–2, 1–3, dan
2–3. Setiap pasangan melalui signed square-root dan normalisasi L2 sebelum
digabungkan.

LMMD memakai one-hot label source dan probabilitas softmax target yang dilepas
dari graph sebagai bobot pseudo-label. Kernel RBF kemudian menghitung discrepancy
per kelas yang aktif pada kedua batch. `warmup_epochs` mencegah pseudo-label acak
awal langsung mendominasi training.
### Decoupled dual-branch GAP-HBP

Eksperimen D1/D2 membagi MobileNetV3 setelah block 4, lalu memberi GAP dan HBP
blok akhir serta classifier masing-masing. Jalankan screening hanya pada
validation terlebih dahulu:

```bash
python -u -m bilinear_lmmd.run_decoupled_screening \
  --data-root data/coffee17_hierarchy_clean/folds/fold_1 \
  --output-root outputs/decoupled-gap-hbp \
  --seeds 123 \
  --evaluation-split val
```

Runner juga menyimpan checkpoint terbaik setiap expert, audit
komplementaritas, bobot gate, dan cosine gradient. Protokol lengkap:
[`docs/DECOUPLED_DUAL_BRANCH_PROTOCOL.md`](docs/DECOUPLED_DUAL_BRANCH_PROTOCOL.md).

### Controlled fine-vs-coarse granularity

Uji penyebab granularitas memakai gambar/split Coffee-17 yang identik dan
membandingkan gain GAP, factorized bilinear, serta HBP pada label fine-17 dan
coarse-9:

```bash
python -u -m bilinear_lmmd.run_granularity_experiment \
  --fine-root data/coffee_clean/folds/fold_1 \
  --coarse-root outputs/coffee17_coarse9_fold1 \
  --output-root outputs/granularity \
  --seeds 123 \
  --evaluation-split val
```

Lihat [`docs/GRANULARITY_PROTOCOL.md`](docs/GRANULARITY_PROTOCOL.md).

Hasil screening seed 123 dan status konfirmasi multi-seed dicatat pada protokol
tersebut. Kandidat backbone SHViT dibatasi sebagai ablation validation setelah
konfirmasi granularity; lihat
[`docs/SHVIT_BACKBONE_SCREENING.md`](docs/SHVIT_BACKBONE_SCREENING.md).

Setelah test terkunci tersedia, confidence interval berpasangan dapat dihitung
tanpa training ulang dari `predictions.csv`:

```bash
python -u -m bilinear_lmmd.run_granularity_bootstrap \
  --report-root outputs/granularity/reports \
  --seeds 42 123 2026 \
  --iterations 10000 \
  --output outputs/granularity/reports/granularity_bootstrap.json
```

Hasil final 10.000 bootstrap test: Fine-17 HBP gain `+3,03%` dengan CI 95%
`[+0,57%; +5,69%]`; Coarse-9 gain `+0,42%` dengan CI
`[-2,35%; +3,22%]`; granularity difference-in-differences `+2,60%` dengan CI
`[-0,88%; +6,25%]`. Interpretasi dan batas klaim tersedia di
[`docs/GRANULARITY_PROTOCOL.md`](docs/GRANULARITY_PROTOCOL.md).
