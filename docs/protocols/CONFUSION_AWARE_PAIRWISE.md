# Confusion-aware pairwise protocol

## Pertanyaan penelitian

Apakah pembelajaran representasi berbasis pasangan meningkatkan klasifikasi
fine-grained Coffee17, dan apakah memprioritaskan pasangan kelas yang sedang
tertukar pada **training** memberi manfaat di luar supervised contrastive
learning biasa?

## Pijakan literatur

- Zhuang et al., *Learning Attentive Pairwise Interaction for Fine-Grained
  Classification*, AAAI 2020, DOI 10.1609/aaai.v34i07.7016.
- Dubey et al., *Pairwise Confusion for Fine-Grained Visual Classification*,
  ECCV 2018.
- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020.
- Suh et al., *Stochastic Class-Based Hard Example Mining for Deep Metric
  Learning*, CVPR 2019.
- Anderson et al., *Elusive Images: Beyond Coarse Analysis for Fine-Grained
  Recognition*, WACV 2024.
- Hu et al., *Siamese networks for few-shot coffee bean defect detection*, LWT
  2025, DOI 10.1016/j.lwt.2025.118631.

Implementasi ini bukan reproduksi identik API-Net, Pairwise Confusion, atau
Siamese Hu et al. Metode mengambil prinsip yang terkontrol: dua augmentasi
independen dari kelas yang sama, supervised contrastive objective, dan bobot
negative berdasarkan confusion antar kelas yang dihitung dari train.

## Model terkontrol

| Kode | Model | Objective training |
|---|---|---|
| BE2G | EfficientNetV2-B0 + GAP | CE |
| BE2H | EfficientNetV2-B0 + HBP | CE, baseline terkuat sebelumnya |
| CP1 | EfficientNetV2-B0 + GAP | CE + vanilla SupCon |
| CP2 | EfficientNetV2-B0 + GAP | CE + confusion-aware SupCon |

CP1 dan CP2 identik pada backbone, GAP, classifier, augmentasi, projection head,
optimizer, schedule, dan jumlah epoch. Satu-satunya perbedaan adalah CP2 memberi
bobot lebih besar pada negative dari pasangan kelas yang sedang membingungkan.

Projection head hanya digunakan ketika training. `best.pt` hanya menyimpan
backbone GAP dan classifier biasa. Karena itu parameter dan jalur inference CP1
dan CP2 sama dengan BE2G.

## Objective

Untuk embedding ternormalisasi `z`, CP1 memakai supervised contrastive loss.
CP2 mengubah bobot negative:

```text
w(a,b) = 1 + alpha * C[a,b]
loss    = CE + lambda * weighted-SupCon
```

`C` adalah confusion matrix simetris yang dibangun dari softmax prediction
train, bukan validation/test. Diagonal dibuat nol dan matriks dinormalisasi ke
`[0,1]`. CP2 memulai dengan warm-up SupCon biasa; setelah warm-up, `C` diperbarui
dengan EMA setiap epoch. Validation hanya untuk pemilihan checkpoint.

## Audit sebelum training

`run_confusion_pair_audit` membaca `predictions.csv` lintas model/seed dan
menyimpan:

- `pairwise_confusion.csv`;
- `sample_consensus.csv`;
- `confusion_audit.json`.

Audit validation bersifat diagnostik dan **dilarang** menjadi sampler training.
Pasangan disebut stabil hanya bila salah minimal dua kali serta muncul pada
minimal dua model dan dua seed. Sampel yang salah pada seluruh run ditandai
untuk audit label/visual, bukan otomatis dibuang.

## Fail-fast

Screening pertama hanya seed 123 pada validation. Test dikunci. Perbandingan
wajib:

1. CP1 vs BE2G: efek vanilla SupCon;
2. CP2 vs CP1: efek confusion-aware weighting;
3. CP2 vs BE2H: apakah usulan melampaui baseline HBP terkuat.

Setiap perbandingan PASS hanya jika Macro-F1 dan Hard-F1 naik serta Worst-F1
tidak turun lebih dari satu poin. CP2 final PASS hanya bila CP2 vs CP1 dan CP2 vs
BE2H sama-sama PASS. Jika gagal, metode dihentikan tanpa membuka test atau
menyetel bobot berdasarkan validation.

## Menjalankan

```bash
python -u -m bilinear_lmmd.experiments.run_pairwise_contrastive_screening \
  --data-root data/coffee17-clean/folds/fold_1 \
  --baseline-root outputs/backbone-results \
  --output-root outputs/confusion-aware-pairwise \
  --seeds 123 \
  --models CP1 CP2 \
  --evaluation-split val
```

Audit prediction terpisah:

```bash
python -u -m bilinear_lmmd.experiments.run_confusion_pair_audit \
  --prediction BE2G:42=.../BE2G_seed42/predictions.csv \
  --prediction BE2H:42=.../BE2H_seed42/predictions.csv \
  --prediction BE2G:123=.../BE2G_seed123/predictions.csv \
  --prediction BE2H:123=.../BE2H_seed123/predictions.csv \
  --output-dir outputs/confusion-audit
```

## Batas klaim

- CP1/CP2 adalah adaptation study, bukan reproduksi API-Net atau Hu et al.
- Confusion dinamis tidak membuktikan label benar; unanimous-wrong memerlukan
  pemeriksaan manusia.
- Hasil satu seed hanya screening.
- Worst-F1 satu kelas tetap dilaporkan, tetapi interpretasi harus didampingi
  per-class support karena validation Coffee17 sangat kecil.
