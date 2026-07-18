# Protokol Decoupled Dual-Branch GAP-HBP

## Pertanyaan penelitian

Apakah GAP dan HBP menjadi lebih komplementer ketika hanya early/mid encoder
yang dibagi, sedangkan blok akhir dan classifier dilatih secara independen?

Eksperimen ini dibuat karena ensemble dua model terpisah menunjukkan
komplementaritas, sementara dua pooling pada feature map yang sama dapat saling
membentuk backbone melalui gradient yang berbeda.

## Model

| Kode | Shared encoder | Late branch | Fusion | Pemakaian |
|---|---|---|---|---|
| M0 | MobileNetV3 | satu GAP | tidak ada | baseline GAP |
| M1 | MobileNetV3 | satu HBP | tidak ada | baseline HBP |
| D1 | stem + block 0--4 | GAP dan HBP pada copy block 5--6 | rata-rata 0,5/0,5 | kontrol dual-branch |
| D2 | sama dengan D1 | sama dengan D1 | gate kecil berbasis logits | kandidat adaptive fusion |
| D3 | model D2 | hanya GAP saat inference | HBP dilepas | diagnostic detachable |

HBP D1/D2 tetap memakai fitur MobileNetV3 yang sama dengan M1:
`out_indices=[1,3,4]`. Jadi definisi HBP tidak diam-diam berubah.

Fusion memakai logits expert yang di-`detach`. Konsekuensinya:

- `CE_gap` melatih shared encoder dan late GAP branch;
- `CE_hbp` melatih shared encoder dan late HBP branch;
- `CE_fused` hanya melatih gate D2;
- D1 tidak memiliki parameter fusion yang perlu dilatih.

Objective:

```text
L = CE_fused + 0.5 CE_gap + 0.5 CE_hbp
```

Tidak ada diversity loss pada eksperimen awal. Gradient cosine antara
`CE_gap` dan `CE_hbp` pada shared encoder dicatat sekali per epoch; PCGrad hanya
layak diuji bila konflik negatif memang berulang.

## Checkpoint yang disimpan

- `best.pt`: epoch Macro-F1 validation fusion terbaik.
- `best_gap.pt`: epoch Macro-F1 validation GAP expert terbaik.
- `best_hbp.pt`: epoch Macro-F1 validation HBP expert terbaik.
- `last.pt`: state resume lengkap.

Expert `D1_gap`/`D1_hbp` dan `D2_gap`/`D2_hbp` dievaluasi dari `best.pt` untuk
mengukur komplementaritas pada model fusion yang sama. D3 memakai
`best_gap.pt`, karena checkpoint-nya harus dipilih berdasarkan main branch GAP.

## Tahapan dan aturan keputusan

### Screening

- split: validation;
- seed: 123;
- kandidat: D1 dan D2;
- kontrol: M0 dan M1 pada data dan konfigurasi training yang sama.

Kandidat diteruskan ke konfirmasi bila:

1. Macro-F1 fusion minimal tidak lebih buruk dari baseline terbaik;
2. Hard-F1 atau Worst-F1 membaik;
3. terdapat sampel `gap_only` dan `hbp_only` (expert tidak identik);
4. gate D2 tidak runtuh permanen ke satu expert.

Ambang ini sengaja menjadi filter murah, bukan bukti final.

### Konfirmasi

- kunci kandidat dari validation;
- jalankan seed 42, 123, dan 2026;
- baru kemudian evaluasi `test` satu kali;
- laporkan mean, standard deviation, dan delta berpasangan.

Jika kandidat menang, capacity-matched dual-branch control tetap diperlukan
sebelum peningkatan diatribusikan kepada kombinasi GAP-HBP. D1/D2 memiliki
lebih banyak parameter karena menduplikasi blok akhir.

## Audit wajib

- Macro-F1, Hard-F1, Worst-F1;
- GAP-only, HBP-only, both-correct, both-wrong, oracle accuracy;
- prediction agreement;
- mean/entropy/selection fraction gate;
- gradient cosine per epoch;
- parameter, ukuran model, latency, dan biaya training;
- hasil detachable D3.

## Perintah Kaggle

Screening satu seed pada validation:

```python
%cd /kaggle/working/bilinear-LMMD

import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/bilinear-LMMD/src" + os.pathsep + env.get("PYTHONPATH", "")

subprocess.run([
    sys.executable, "-u", "-m", "bilinear_lmmd.experiments.run_decoupled_screening",
    "--data-root", "/kaggle/working/bilinear-LMMD/data/coffee17_hierarchy_clean/folds/fold_1",
    "--output-root", "/kaggle/working/decoupled-gap-hbp-results",
    "--seeds", "123",
    "--evaluation-split", "val",
], check=True, env=env)
```

Jika dataset berada pada Kaggle Input, ganti `--data-root` dengan folder yang
langsung berisi `source/train`, `source/val`, dan `source/test`.

Jangan mengganti `val` menjadi `test` saat masih memilih D1 atau D2.

Untuk dataset CBD 8 kelas yang telah dipreparasi, gunakan preset tersendiri:

```python
subprocess.run([
    sys.executable, "-u", "-m", "bilinear_lmmd.experiments.run_decoupled_screening",
    "--preset", "cbd",
    "--data-root", "/kaggle/working/cbd-prepared-kd-v2",
    "--output-root", "/kaggle/working/cbd-decoupled-results",
    "--seeds", "42",
    "--evaluation-split", "val",
], check=True, env=env)
```

Preset CBD memakai kode CBD0, CBD1, CBDC1, CBDD1, dan CBDD2, `num_classes=8`, 25
epoch, serta hard group tujuh kelas cacat. CBD bukan external test Coffee-17.

Setelah screening seed 42 mengunci CBD1 dan CBDD2, confirmation validation
tidak perlu melatih ulang kontrol yang gugur:

```python
subprocess.run([
    sys.executable, "-u", "-m", "bilinear_lmmd.experiments.run_decoupled_screening",
    "--preset", "cbd",
    "--models", "CBD1", "CBDD2",
    "--data-root", str(PREPARED_ROOT),
    "--output-root", "/kaggle/working/cbd-decoupled-results",
    "--seeds", "123", "2026",
    "--evaluation-split", "val",
], check=True, env=env)
```

Setelah CBDD2 menang pada test yang telah dikunci, capacity audit diperlakukan
sebagai analisis post-hoc. CBDC1 memakai ordinary HBP dengan residual pointwise
block berdimensi 1137: 5.735.706 parameter versus 5.736.234 pada CBDD2
(selisih 528; kurang dari 0,01%). Dimensi ini dikunci berdasarkan jumlah
parameter, bukan performa.

```python
subprocess.run([
    sys.executable, "-u", "-m", "bilinear_lmmd.experiments.run_decoupled_screening",
    "--preset", "cbd",
    "--models", "CBDC1",
    "--data-root", str(DATA_ROOT),
    "--output-root", str(OUTPUT_ROOT),
    "--seeds", "42", "123", "2026",
    "--evaluation-split", "val",
], check=True, env=env)
```

Benchmark arsitektur dijalankan pada perangkat dan session yang sama:

```python
subprocess.run([
    sys.executable, "-u", "-m", "bilinear_lmmd.experiments.run_decoupled_efficiency",
    "--output", str(OUTPUT_ROOT / "efficiency.json"),
    "--models", "CBD1", "CBDC1", "CBDD2",
    "--batch-sizes", "1", "32",
    "--warmup", "20",
    "--iterations", "100",
], check=True, env=env)
```

## Batas klaim

- D1/D2 adalah sintesis eksperimental, bukan implementasi identik ONE atau
  DSoP.
- D3 paling dekat dengan pertanyaan DSoP, tetapi tetap adaptasi sendiri.
- Hasil Coffee-17 tidak otomatis berlaku pada USK, CBD, atau citra conveyor.
- Peningkatan satu seed validation tidak cukup untuk klaim superior.

Dasar literatur dan tautan primer terdapat pada
[`DECOUPLED_DUAL_BRANCH_REFERENCES.md`](DECOUPLED_DUAL_BRANCH_REFERENCES.md).
