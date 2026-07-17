# Protokol Benchmark Backbone Kontemporer

## Pertanyaan penelitian

Benchmark ini menguji dua faktor secara terpisah:

1. backbone mana yang memberi transfer terbaik pada Coffee-17; dan
2. apakah efek HBP dibanding GAP konsisten pada CNN, Transformer hierarkis,
   dan hybrid CNN-Transformer.

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

Setiap backbone mempunyai pasangan `GAP + CE` dan `HBP + CE`. Faktor lain
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

Perbedaan stride SHViT harus dicatat saat menginterpretasikan hasil karena
feature map-nya berada pada reduksi 16, 32, dan 64.

## Screening

Gunakan validation dan satu seed dahulu:

```bash
python -u -m bilinear_lmmd.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/bilinear-LMMD-backbones/results \
  --seeds 123 \
  --evaluation-split val
```

Runner menampilkan nomor run, output epoch dari proses training, efek HBP per
backbone, lalu menyimpan `backbone_leaderboard.json` dan
`backbone_leaderboard.csv`. Training dapat dilanjutkan dengan menjalankan
perintah yang sama karena runner memakai `--resume` dan melewati artefak yang
sudah lengkap.

Untuk menghemat waktu, GAP-only dapat dijalankan lebih dahulu:

```bash
python -u -m bilinear_lmmd.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/bilinear-LMMD-backbones/results \
  --heads gap \
  --seeds 123 \
  --evaluation-split val
```

Namun keputusan final tentang kegunaan HBP harus memakai pasangan GAP/HBP pada
backbone yang telah ditetapkan sebelum hasil test dilihat.

## Konfirmasi dan test

Setelah shortlist dikunci, jalankan tiga seed hanya untuk kandidat tersebut,
misalnya:

```bash
python -u -m bilinear_lmmd.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/bilinear-LMMD-backbones/results \
  --backbones PV2 SHV \
  --seeds 42 123 2026 \
  --evaluation-split val
```

Test dibuka sekali setelah keputusan dikunci. Runner sengaja mewajibkan flag
tambahan:

```bash
python -u -m bilinear_lmmd.run_backbone_screening \
  --data-root /content/coffee17-clean-grouped/folds/fold_1 \
  --output-root /content/drive/MyDrive/bilinear-LMMD-backbones/results \
  --backbones PV2 SHV \
  --seeds 42 123 2026 \
  --evaluation-split test \
  --allow-test
```

## Kriteria keputusan

Macro-F1 adalah metrik primer. Hard-F1, Worst-F1, stabilitas antar-seed,
parameter, latensi batch-1, throughput batch-32, dan peak VRAM adalah metrik
sekunder. Model dipilih dari Pareto frontier; jangan memakai satu skor gabungan
post-hoc yang bobotnya ditentukan setelah hasil terlihat.
