# XAI Protocol: M1 vs M5w01

## Tujuan

Analisis ini menjawab pertanyaan setelah eksperimen domain adaptation:

1. ketika LMMD menyelamatkan prediksi target, apakah perhatian model menjadi
   lebih terpusat pada biji;
2. ketika LMMD merusak prediksi yang sebelumnya benar, apakah perhatian bocor
   ke latar atau kehilangan bukti kelas yang relevan;
3. apakah pola tersebut berbeda antara `illumination`, `sensor`, dan
   `background`;
4. apakah source forgetting pada sensor/background terlihat pada penjelasan
   model, bukan hanya pada skor agregat.

XAI bersifat **post-hoc** dan tidak mengubah training atau memilih checkpoint.
Hasilnya adalah diagnosis pada benchmark sintetis terkontrol, bukan bukti
kausal atau validasi dunia nyata.

## Metode

Runner menghasilkan dua penjelasan untuk M1 dan M5w01:

- **LayerCAM**: baseline penjelasan kelas pada tiga feature map backbone yang
  dipakai HBP;
- **Finer-LayerCAM**: menjelaskan selisih logit
  `y_actual - gamma * y_reference`, bukan mengurangkan dua heatmap yang sudah
  jadi.

Tiga kelas dengan logit tertinggi selain kelas aktual dipakai sebagai referensi.
Setiap peta pasangan dinormalisasi dan dirata-ratakan. Nilai default mengikuti
desain final Finer-CAM: `gamma=0.6` dan tiga reference class.

Kelas aktual sengaja menjadi explanation target karena tujuan analisis error
adalah melihat bukti yang mendukung atau gagal mendukung label benar. Label test
hanya dipakai setelah training selesai dan tidak masuk ke loss atau pemilihan
checkpoint.

## Pemilihan sampel

Untuk setiap domain, source/target, dan seed, prediksi M1 dan M5w01 dipasangkan
menjadi empat kategori:

| Kategori | M1 | M5w01 | Arti |
|---|---|---|---|
| `rescued` | salah | benar | prediksi diselamatkan LMMD |
| `negative_transfer` | benar | salah | prediksi dirusak LMMD |
| `both_correct` | benar | benar | stabil benar |
| `both_wrong` | salah | salah | kegagalan bersama |

Default mengambil dua sampel per kategori melalui ranking SHA-256 deterministik.
Ini menjaga waktu eksekusi dan mencegah pemilihan gambar manual setelah melihat
heatmap. Karena merupakan subsampel diagnostik, rata-rata XAI tidak boleh
dilaporkan sebagai estimasi seluruh populasi test.

## Metrik

Mask foreground selalu dibuat dari pasangan gambar source asli. Transformasi
sintetis mempertahankan geometri, sehingga mask yang sama dapat disejajarkan
dengan target setelah resize dan center crop.

| Metrik | Interpretasi |
|---|---|
| `foreground_mass` | fraksi total energi CAM di dalam biji; lebih tinggi lebih baik |
| `background_leakage` | `1 - foreground_mass`; lebih rendah lebih baik |
| `foreground_lift` | foreground mass dibagi luas mask; >1 berarti konsentrasi di biji |
| `top20_iou` | IoU 20% piksel CAM terkuat dengan mask biji |
| `target_confidence_drop` | penurunan probabilitas kelas aktual setelah 5% piksel terkuat dimask |
| `relative_confidence_drop` | drop kelas aktual dikurangi drop reference class |

Relative confidence drop mengikuti motivasi Finer-CAM: region diskriminatif
seharusnya lebih merusak confidence kelas target daripada kelas pembanding saat
dihapus. Metrik dapat negatif dan tidak boleh ditafsirkan sebagai akurasi.

## Menjalankan di Kaggle

Runner memakai artefak yang sudah dihasilkan oleh rescue confirmation dan
cross-shift confirmation. Tidak ada training baru.

```bash
cd /kaggle/working/bilinear-LMMD
git pull

python -u -m bilinear_lmmd.run_xai_analysis \
  --data-root /kaggle/working/coffee-synthetic-components \
  --illumination-root /kaggle/working/lmmd-rescue-confirmation \
  --cross-shift-root /kaggle/working/lmmd-cross-shift-confirmation \
  --output-root /kaggle/working/xai-results \
  --domains illumination sensor background \
  --evaluation-domains target source \
  --seeds 42 2026 \
  --samples-per-outcome 2
```

Runner aman dilanjutkan setelah notebook berhenti. Satu sampel dianggap lengkap
hanya jika JSON dan panel PNG sudah ada. Protokol disimpan di `protocol.json`;
runner menolak mencampur hasil jika gamma, seed, atau aturan sampling berubah.

Artefak utama:

```text
xai-results/
|-- protocol.json
|-- samples/<domain>/<source|target>/seed<seed>/<outcome>/*.png
|-- samples/<domain>/<source|target>/seed<seed>/<outcome>/*.json
`-- reports/
    |-- xai_samples.csv
    `-- xai_summary.json
```

Panel PNG berisi input, LayerCAM M1, Finer-LayerCAM M1, LayerCAM M5w01,
dan Finer-LayerCAM M5w01. Garis hijau adalah batas mask biji.

## Aturan interpretasi

Kesimpulan paling kuat membutuhkan kesesuaian tiga bukti:

1. kategori prediksi (`rescued` atau `negative_transfer`);
2. perubahan foreground mass/leakage;
3. relative confidence drop yang searah.

Heatmap yang terlihat menarik tanpa perubahan faithfulness tidak cukup untuk
menyimpulkan model memakai ciri yang benar. Demikian pula, target macro-F1 yang
naik tetapi source retention gagal tetap dilaporkan sebagai trade-off, bukan
robustness universal.

Referensi utama:

- Zhang et al. (2025), *Finer-CAM: Spotting the Difference Reveals Finer
  Details for Visual Explanation*, CVPR 2025.
- Jiang et al. (2021), *LayerCAM: Exploring Hierarchical Class Activation Maps
  for Localization*, IEEE Transactions on Image Processing.
