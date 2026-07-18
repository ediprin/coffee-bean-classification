# Protokol Benchmark Backbone Kontemporer

## Pertanyaan penelitian

Benchmark ini menguji dua faktor secara terpisah:

1. backbone mana yang memberi transfer terbaik pada Coffee-17; dan
2. apakah efek HBP dibanding GAP konsisten pada CNN, Transformer hierarkis,
   dan hybrid CNN-Transformer.

Perbandingan orde kedua memakai kontrol tambahan
`Projected Hierarchical GAP (PH-GAP)`. Kontrol ini memiliki proyeksi,
normalisasi, dimensi embedding, classifier, dan jumlah parameter yang identik
dengan HBP, tetapi tidak melakukan perkalian antarlayer. Karena itu:

- GAP vs PH-GAP mengukur efek feature hierarchy, proyeksi, dan kapasitas;
- PH-GAP vs HBP mengisolasi efek interaksi multiplikatif orde kedua; dan
- HBP vs HBP-linear mengukur efek modifikasi BatchNorm-ReLU terhadap
  formulasi proyeksi linear.

Benchmark bukan klaim pretraining arsitektur murni. Setiap model memakai
checkpoint publiknya sendiri, sehingga hasil harus disebut sebagai
**transfer-learning system comparison**.

## Backbone yang telah dikunci

| Keluarga | Backbone/checkpoint | Catatan pretraining |
|---|---|---|
| CNN mobile | `mobilenetv4_conv_medium.e500_r224_in1k` | supervised ImageNet-1K, 224 |
| CNN scaled | `tf_efficientnetv2_b0.in1k` | supervised ImageNet-1K, pretrained 192 |
| CNN modern | `convnextv2_atto.fcmae_ft_in1k` | FCMAE lalu fine-tune ImageNet-1K |
| Transformer hierarkis | `pvt_v2_b0.in1k` | ImageNet-1K, 224 |
| CNN-Transformer | `shvit_s1.in1k` | ImageNet-1K, 224 |

MobileNetV3-Large tetap dilaporkan sebagai legacy anchor dari eksperimen HBP
sebelumnya, tetapi bukan peserta baru dalam seleksi lintas keluarga.

## Matriks eksperimen

Setiap backbone yang kompatibel mempunyai `GAP + CE`, `PH-GAP + CE`,
`HBP + CE`, dan `HBP-linear + CE`. Faktor lain
harus sama: data split, input 224, augmentasi, optimizer, scheduler, epoch,
label smoothing, dan seed.

HBP memakai tiga feature map. Indeks dipilih agar mencakup kedalaman awal,
menengah, dan terdalam yang tersedia:

| Keluarga | GAP indices | HBP indices |
|---|---:|---:|
| MV4 | `[4]` | `[1, 3, 4]` |
| EV2 | `[4]` | `[1, 3, 4]` |
| CV2 | `[3]` | `[0, 2, 3]` |
| PV2 | `[3]` | `[0, 2, 3]` |
| SHV | `[2]` | `[0, 1, 2]` |

MV3, MV4, EV2, CV2, dan PV2 menghasilkan reduksi `[4, 16, 32]`, yaitu grid
`[56x56, 14x14, 7x7]` untuk input 224. SHViT menghasilkan reduksi
`[16, 32, 64]` atau grid `[14x14, 7x7, 4x4]`. SHViT karena itu hanya boleh
masuk leaderboard GAP utama. Head hierarkis SHViT bersifat eksploratif dan
tidak boleh dibandingkan langsung dengan HBP backbone lain sebelum tersedia
adapter yang mengekspos feature map beresolusi lebih tinggi.

Runner memeriksa geometri ini sebelum training. Audit tanpa training:

```bash
python -u -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/backbone-audit \
  --backbones MV3 MV4 EV2 CV2 PV2 SHV \
  --heads gap hierarchical_gap hbp hbp_linear \
  --audit-only
```

Laporan disimpan sebagai `backbone_compatibility.json`. Flag
`--allow-incompatible-hierarchy` hanya boleh digunakan untuk eksperimen
eksploratif yang dilabeli secara eksplisit.

## Screening

Gunakan validation dan satu seed dahulu:

```bash
python -u -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/coffee-bean-classification-backbones/results \
  --backbones MV4 EV2 CV2 PV2 \
  --heads gap hierarchical_gap hbp \
  --seeds 123 \
  --evaluation-split val
```

Runner menampilkan nomor run, output epoch dari proses training, efek HBP per
backbone, lalu menyimpan `backbone_leaderboard.json` dan
`backbone_leaderboard.csv`. Training dapat dilanjutkan dengan menjalankan
perintah yang sama karena runner memakai `--resume` dan melewati artefak yang
sudah lengkap.

## Checkpoint lintas runtime dengan Hugging Face

Google Drive tetap dapat digunakan, tetapi runner juga mendukung private model
repository Hugging Face. `last.pt` menyimpan model, optimizer, scheduler,
history, dan nomor epoch sehingga training benar-benar dilanjutkan, bukan
sekadar memuat bobot terbaik.

Simpan token **write** dengan nama `HF_TOKEN` di Colab/Kaggle Secrets. Jangan
menulis token langsung di notebook. Pada Colab:

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
HF_REPO = "NAMA_USER/coffee-backbone-checkpoints"
```

Kemudian jalankan output lokal yang cepat dan sinkronkan setiap lima epoch:

```bash
python -u -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/backbone-results \
  --seeds 123 \
  --evaluation-split val \
  --hf-repo NAMA_USER/coffee-backbone-checkpoints \
  --hf-sync-every 5
```

Repo private dibuat otomatis jika belum ada. Sebelum setiap run, runner
memulihkan checkpoint dan report yang belum tersedia secara lokal. Setiap lima
epoch, file berikut dikirim sebagai satu commit:

- `last.pt` untuk resume lengkap;
- `best.pt` untuk evaluasi;
- `history.json`;
- `resolved_config.json`; dan
- `artifact_manifest.json`, termasuk commit Git, seed, dan data root.

Jika runtime mati, maksimum empat epoch setelah sinkronisasi terakhir perlu
diulang. Gunakan `--hf-sync-every 1` untuk perlindungan setiap epoch dengan
konsekuensi upload lebih sering. Dataset tetap harus tersedia kembali dengan
split dan urutan kelas yang identik; checkpoint akan menolak resume jika kelas
berbeda.

Untuk menghemat waktu, audit dan GAP-only dapat dijalankan lebih dahulu:

```bash
python -u -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/coffee-bean-classification-backbones/results \
  --heads gap \
  --seeds 123 \
  --evaluation-split val
```

Namun keputusan final tentang kegunaan HBP harus memakai PH-GAP/HBP yang
capacity-matched pada backbone yang telah ditetapkan sebelum hasil test dilihat.

## Konfirmasi dan test

Setelah shortlist dikunci, jalankan tiga seed hanya untuk kandidat tersebut,
misalnya:

```bash
python -u -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/coffee-bean-classification-backbones/results \
  --backbones EV2 PV2 \
  --heads gap hierarchical_gap hbp hbp_linear \
  --seeds 42 123 2026 \
  --evaluation-split val
```

Test dibuka sekali setelah keputusan dikunci. Runner sengaja mewajibkan flag
tambahan:

```bash
python -u -m bilinear_lmmd.experiments.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/coffee-bean-classification-backbones/results \
  --backbones EV2 PV2 \
  --heads gap hierarchical_gap hbp hbp_linear \
  --seeds 42 123 2026 \
  --evaluation-split test \
  --allow-test
```

## Kriteria keputusan

Macro-F1 adalah metrik primer. Hard-F1, Worst-F1, stabilitas antar-seed,
parameter, latensi batch-1, throughput batch-32, dan peak VRAM adalah metrik
sekunder. Model dipilih dari Pareto frontier; jangan memakai satu skor gabungan
post-hoc yang bobotnya ditentukan setelah hasil terlihat.
