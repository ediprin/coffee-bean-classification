# Controlled Fine-vs-Coarse Granularity Protocol

## Pertanyaan

Apakah keuntungan representasi orde kedua terhadap GAP lebih besar ketika
label benar-benar fine-grained?

CBD tetap digunakan sebagai independent natural-coarse benchmark, tetapi tidak
dapat mengisolasi granularitas karena berbeda dari Coffee-17 dalam ukuran,
imbalance, sumber gambar, dan taxonomy. Kontrol internal ini memakai gambar dan
split Coffee-17 yang sama; hanya target label yang diubah.

## Taxonomy coarse operasional

| Parent | Fine classes |
|---|---|
| Black | Full Black, Partial Black |
| Sour | Full Sour, Partial Sour |
| Insect Damage | Severe Insect Damage, Slight Insect Damage |
| Physical Form | Broken, Cut, Shell |
| Covering Residue | Husk, Parchment |
| Developmental | Immature, Withered |
| Processing Density | Dry Cherry, Floater |
| Fade | Fade |
| Fungus Damage | Fungus Damage |

Grouping ini dibuat untuk eksperimen visual/mechanistic. Ini bukan taxonomy
resmi SNI atau CBD dan tidak boleh disebut demikian. Dua singleton dipertahankan
karena memaksakan Fade/Fungus ke mekanisme lain akan menambah label noise.

Preparasi memakai hard link bila tersedia dan tidak mengubah split. Audit
merekam setiap fine-to-coarse assignment dan jumlah gambar per split.

## Faktorial model

| Task | Kode | Head | Statistik |
|---|---|---|---|
| Fine-17 | GF0 | GAP | orde pertama |
| Fine-17 | GF0b | factorized bilinear rank 160 | orde kedua satu layer |
| Fine-17 | GF1 | HBP 3 x rank 512 | orde kedua lintas layer |
| Coarse-9 | GC0 | GAP | orde pertama |
| Coarse-9 | GC0b | factorized bilinear rank 160 | orde kedua satu layer |
| Coarse-9 | GC1 | HBP 3 x rank 512 | orde kedua lintas layer |

GF0b/GF1 dan GC0b/GC1 mempunyai embedding 1.536 dimensi dan kapasitas yang
berdekatan. Konfigurasi data, augmentasi, optimizer, epoch, backbone, dan seed
identik antar-task.

## Estimand

Macro-F1 fine dan coarse tidak dibandingkan secara absolut. Yang dibandingkan
adalah gain terhadap GAP:

```text
fine_gain   = MacroF1(GF1) - MacroF1(GF0)
coarse_gain = MacroF1(GC1) - MacroF1(GC0)
effect      = fine_gain - coarse_gain
```

Hipotesis didukung bila `effect > 0` secara konsisten lintas seed. Analisis
serupa dilakukan untuk factorized bilinear. Hasil satu seed validation adalah
screening, bukan bukti final.

## Tahapan

1. Validation seed 123 untuk keenam model.
2. Jika tanda effect sesuai hipotesis, konfirmasi validation seed 42 dan 2026.
3. Kunci interpretasi sebelum membuka test.
4. Untuk bukti lebih kuat, ulangi sebagai grouped OOF, bukan hanya holdout.
5. CBD dilaporkan setelah kontrol internal sebagai external pattern check.

## Perintah Kaggle screening

```python
%cd /kaggle/working/bilinear-LMMD

import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/bilinear-LMMD/src" + os.pathsep + env.get("PYTHONPATH", "")

subprocess.run([
    sys.executable, "-u", "-m", "bilinear_lmmd.run_granularity_experiment",
    "--fine-root", str(FINE_ROOT),
    "--coarse-root", "/kaggle/working/coffee17-coarse9-fold1",
    "--output-root", "/kaggle/working/granularity-results",
    "--seeds", "123",
    "--evaluation-split", "val",
], check=True, env=env)
```

Jangan memakai test untuk memilih taxonomy, rank, atau model.
