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
| Fine-17 | GF0b | factorized bilinear, projection dim 160 | orde kedua satu layer |
| Fine-17 | GF1 | HBP 3 x projection dim 512 | orde kedua lintas layer |
| Coarse-9 | GC0 | GAP | orde pertama |
| Coarse-9 | GC0b | factorized bilinear, projection dim 160 | orde kedua satu layer |
| Coarse-9 | GC1 | HBP 3 x projection dim 512 | orde kedua lintas layer |

GF0b/GF1 dan GC0b/GC1 mempunyai embedding 1.536 dimensi dan kapasitas yang
berdekatan. `projection_dim` menyatakan dimensi embedding faktorisasi dan tidak
boleh langsung disebut sebagai rank aljabar efektif. Konfigurasi data,
augmentasi, optimizer, epoch, backbone, dan seed identik antar-task.

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

## Hasil screening validation seed 123

Eksperimen satu seed selesai sebelum sesi komputasi direset. Hasil ini hanya
dipakai untuk menentukan apakah konfirmasi multi-seed layak dijalankan.

| Task/model | Accuracy | Macro-F1 | Worst-F1 |
|---|---:|---:|---:|
| Fine-17 GAP (GF0) | 86,60% | 87,50% | 66,67% |
| Fine-17 factorized bilinear (GF0b) | 88,66% | 89,09% | 66,67% |
| Fine-17 HBP (GF1) | 89,69% | 90,45% | 72,73% |
| Coarse-9 GAP (GC0) | 94,85% | 95,24% | 92,31% |
| Coarse-9 factorized bilinear (GC0b) | 94,85% | 94,78% | 87,50% |
| Coarse-9 HBP (GC1) | 91,75% | 93,00% | 87,18% |

Macro-F1 HBP terhadap GAP meningkat `+2,96` poin pada Fine-17 dan menurun
`-2,24` poin pada Coarse-9. Difference-in-differences awal adalah `+5,20`
poin. Factorized bilinear satu layer juga menunjukkan pola yang searah:
`+1,59` poin pada Fine-17 dan `-0,46` poin pada Coarse-9.

Hasil ini mendukung hipotesis screening bahwa interaksi bilinear lebih berguna
ketika label benar-benar fine-grained. Hasil belum menjadi bukti final karena
baru seed 123 pada validation. Konfirmasi seed 42 dan 2026 harus dijalankan
ulang; test tetap tertutup sampai interpretasi validation dikunci.

## Konfirmasi tiga seed dan test terkunci

Validation tiga seed menghasilkan gain Macro-F1 HBP `+0,62 ± 1,77` poin pada
Fine-17 dan `-1,78 ± 0,86` poin pada Coarse-9. Difference-in-differences adalah
`+2,39 ± 1,84` poin dan positif pada 3/3 seed. Berdasarkan hasil ini,
interpretasi dikunci sebelum test dibuka.

Pada test, HBP meningkatkan Macro-F1 Fine-17 sebesar `+3,03 ± 1,65` poin dan
Coarse-9 sebesar `+0,42 ± 1,74` poin. Difference-in-differences test adalah
`+2,60 ± 3,33` poin dan positif pada 2/3 seed. Worst-class F1 Fine-17 meningkat
`+6,65` poin, sedangkan pada Coarse-9 turun `-2,82` poin.

Temuan mendukung arah hipotesis bahwa HBP lebih bermanfaat pada label
fine-grained, tetapi tidak membuktikan efek universal atau signifikansi
statistik. Uji t eksploratif berbasis tiga seed tidak melewati ambang 5% dan
tidak dijadikan bukti utama.

## Paired stratified bootstrap tanpa training ulang

Runner `run_granularity_bootstrap` membaca `predictions.csv` GF0/GF1/GC0/GC1,
memasangkan identitas gambar Fine dan Coarse, lalu melakukan resampling dengan
replacement di dalam setiap kelas Fine-17. Indeks bootstrap yang sama dipakai
untuk seluruh model dan seed sehingga estimand tetap berpasangan.

Runner melaporkan dua interval:

1. fixed-trained-seed bootstrap: ketidakpastian sampel test dengan tiga model
   seed yang telah dilatih dianggap tetap;
2. hierarchical bootstrap: selain sampel, tiga seed juga di-resample. Karena
   hanya ada tiga seed, interval ini tetap eksploratif.

```bash
python -u -m bilinear_lmmd.run_granularity_bootstrap \
  --report-root outputs/granularity/reports \
  --seeds 42 123 2026 \
  --iterations 10000 \
  --output outputs/granularity/reports/granularity_bootstrap.json
```

Bootstrap tidak menggantikan grouped OOF atau penambahan seed. Confidence
interval hanya boleh ditafsirkan sesuai sumber ketidakpastian yang di-resample.

## Hasil paired stratified bootstrap final

Bootstrap dijalankan pada 199 sampel test per seed, tiga seed
`[42, 123, 2026]`, 10.000 iterasi, random seed `20260717`, dan confidence
level 95%. Seluruh model memakai identitas sampel dan indeks resampling yang
sama.

| Estimand | Point estimate | CI 95% fixed-seed | Proporsi bootstrap > 0 |
|---|---:|---:|---:|
| Fine-17 HBP gain | +3,03% | [+0,57%; +5,69%] | 0,992 |
| Coarse-9 HBP gain | +0,42% | [-2,35%; +3,22%] | 0,607 |
| Granularity effect | +2,60% | [-0,88%; +6,25%] | 0,931 |

Hierarchical seed-and-sample bootstrap untuk granularity effect menghasilkan
CI 95% `[-2,74%; +8,19%]` dan proporsi replikasi positif `0,839`.

Keputusan:

1. HBP meningkatkan Macro-F1 Fine-17 secara jelas conditional pada tiga model
   seed yang telah dilatih, karena CI fixed-seed seluruhnya berada di atas nol.
2. Efek HBP pada Coarse-9 tidak dapat dibedakan dari nol.
3. Difference-in-differences mendukung arah hipotesis bahwa HBP lebih berguna
   pada label fine-grained, tetapi CI 95% masih mencakup nol. Hipotesis
   granularity belum boleh disebut signifikan atau terbukti universal.
4. Interval hierarchical lebih lebar karena hanya tiga seed tersedia. Hasil
   tersebut menegaskan bahwa generalisasi terhadap inisialisasi training baru
   masih memiliki ketidakpastian besar.

`probability_positive` adalah proporsi replikasi bootstrap dengan estimand di
atas nol. Nilai ini bukan p-value dan bukan probabilitas posterior bahwa
hipotesis benar.

Klaim final yang diperbolehkan:

> Paired stratified bootstrap menunjukkan bahwa HBP meningkatkan Macro-F1
> Fine-17 sebesar 3,03 poin persentase, dengan CI 95% [0,57; 5,69]. Pada
> Coarse-9, peningkatan 0,42 poin tidak dapat dibedakan dari nol.
> Difference-in-differences sebesar 2,60 poin menunjukkan kecenderungan bahwa
> manfaat HBP meningkat seiring granularitas label, tetapi CI 95%
> [-0,88; 6,25] masih mencakup nol.

Artefak lengkap di lingkungan eksperimen disimpan sebagai
`results/reports/granularity_bootstrap.json`. Test telah terkunci dan tidak
boleh digunakan untuk tuning lanjutan.

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
