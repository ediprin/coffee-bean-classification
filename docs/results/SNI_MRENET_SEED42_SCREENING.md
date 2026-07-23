# Screening SNI-MRENet seed 42

**Status: VALIDATION SCREENING SELESAI -- ONTOLOGY FAIL, TEST TERKUNCI.**

Eksperimen memakai dataset oracle instance-crop SNI dengan 21 kelas: 21.273
crop train, 4.969 crop validation, dan 4.832 crop test. Checkpoint setiap model
dipilih hanya berdasarkan Macro-F1 validation. Test tidak dibuka.

## Model dan checkpoint terpilih

| Model | Deskripsi | Epoch terbaik | Accuracy | Balanced accuracy | Macro-F1 | Hard-F1 | Worst-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| SNIB0 | EfficientNetV2-B0, final-stage GAP | 17 | 92,43% | 83,06% | 82,91% | 83,38% | 39,18% |
| SNIB1 | fusi multiresolusi, flat 21-class CE | 43 | **93,52%** | **85,16%** | **84,64%** | 85,19% | **46,46%** |
| SNIB2 | ontology experts + projected hierarchical GAP | 40 | 93,02% | 84,43% | 84,32% | **85,67%** | 34,09% |
| SNIB3 | ontology experts + selective HBP | tidak dijalankan | -- | -- | -- | -- | -- |

Delta SNIB1 terhadap SNIB0:

| Metrik | Delta |
|---|---:|
| Accuracy | +1,09 poin |
| Balanced accuracy | +2,10 poin |
| Macro-F1 | +1,73 poin |
| Hard-F1 | +1,81 poin |
| Worst-F1 | +7,29 poin |

Putusan `SNIB0_vs_SNIB1`: **PASS**. Fusi multiresolusi meningkatkan Macro-F1
dan Hard-F1 serta mempertahankan Worst-F1 sesuai gate yang dibekukan.

Delta SNIB2 terhadap SNIB1:

| Metrik | Delta |
|---|---:|
| Accuracy | -0,50 poin |
| Balanced accuracy | -0,73 poin |
| Macro-F1 | -0,32 poin |
| Hard-F1 | +0,48 poin |
| Worst-F1 | -12,37 poin |

Putusan `SNIB1_vs_SNIB2`: **FAIL**. Sesuai protokol fail-fast, SNIB3 tidak
dijalankan dan test tetap terkunci.

## Audit per kelas SNIB1 versus SNIB2

Nilai berikut berasal dari checkpoint Macro-F1 terbaik masing-masing model.

| Kelas | Support validation | F1 SNIB1 | F1 SNIB2 | Delta |
|---|---:|---:|---:|---:|
| `biji_berkulit_tanduk` | 96 | 95,05% | 91,79% | -3,26 |
| `biji_berlubang_lebih_satu` | 100 | 96,15% | 97,54% | +1,38 |
| `biji_berlubang_satu` | 84 | 87,70% | 84,38% | -3,33 |
| `biji_bertutul_tutul` | 54 | 92,98% | 94,55% | +1,56 |
| `biji_coklat` | 191 | 83,04% | 81,33% | -1,71 |
| `biji_hitam` | 75 | 94,34% | 96,15% | +1,81 |
| `biji_hitam_pecah` | 86 | 90,53% | 95,56% | +5,03 |
| `biji_hitam_sebagian` | 68 | 69,03% | 73,50% | +4,48 |
| `biji_muda` | 58 | 46,46% | 34,09% | **-12,37** |
| `biji_normal` | 3.072 | 97,24% | 96,83% | -0,40 |
| `biji_pecah` | 47 | 73,56% | 76,19% | +2,63 |
| `kopi_gelondong` | 73 | 96,50% | 96,55% | +0,05 |
| `kulit_kopi_ukuran_besar` | 114 | 87,83% | 87,18% | -0,65 |
| `kulit_kopi_ukuran_kecil` | 82 | 92,59% | 86,27% | -6,32 |
| `kulit_kopi_ukuran_sedang` | 115 | 90,30% | 88,03% | -2,26 |
| `kulit_tanduk_ukuran_besar` | 125 | 95,31% | 92,25% | -3,06 |
| `kulit_tanduk_ukuran_kecil` | 41 | 61,54% | 67,50% | +5,96 |
| `kulit_tanduk_ukuran_sedang` | 41 | 62,34% | 68,42% | +6,08 |
| `tanah_batu_ranting_besar` | 218 | 98,62% | 97,27% | -1,34 |
| `tanah_batu_ranting_kecil` | 111 | 83,84% | 83,40% | -0,44 |
| `tanah_batu_ranting_sedang` | 118 | 82,51% | 81,86% | -0,65 |

Perubahan hard-group SNIB2 terhadap SNIB1:

| Hard-group | Delta F1 |
|---|---:|
| `black_variant` | +3,77 poin |
| `hole_count` | -0,97 poin |
| `skin_size` | -0,04 poin |
| `foreign_size` | -0,81 poin |

Kenaikan Hard-F1 SNIB2 terkonsentrasi pada variasi hitam, sedangkan kelas
terlemah `biji_muda` turun 12,37 poin. Epoch 33 SNIB2 pernah menghasilkan
Worst-F1 52,63%, tetapi Macro-F1 83,94% dan Hard-F1 83,99%; memilih epoch itu
setelah melihat hasil akan melanggar aturan seleksi checkpoint Macro-F1.

## Batas atribusi dan keputusan

Perbandingan SNIB1--SNIB2 bukan ablation ontologi murni. SNIB2 sekaligus
menambahkan router/conditional experts dan projected hierarchical GAP pada
expert kondisi biji. Karena itu hasil hanya membuktikan bahwa **paket SNIB2**
gagal; history saja tidak dapat mengisolasi kegagalan router dari representasi
projected-GAP.

SNIB2 mencapai checkpoint terbaik pada epoch 40 dan beberapa nilai Macro-F1
tertingginya berada pada epoch 38--45. Kegagalan bukan akibat training berhenti
terlalu awal. Kandidat terbaik screening seed 42 adalah SNIB1. SNIB1 belum
menjadi bukti multi-seed atau locked-test, dan fusi multiresolusi bukan novelty
algoritmik karena sudah merupakan keluarga teknik yang mapan.

Ukuran model tetap kecil: SNIB0 memiliki 5.614.437 parameter, SNIB1 5.676.965
(+1,11%), sedangkan SNIB2/SNIB3 masing-masing 5.733.545 (+2,12%). Latency
perangkat belum diukur, sehingga peningkatan SNIB1 belum boleh diklaim bebas
biaya inferensi.

**Keputusan final screening v1:** pertahankan SNIB1 sebagai kandidat baseline
multiresolusi; hentikan SNIB2; jangan jalankan SNIB3 atau membuka test. Usulan
SNI-MRENet v1 belum memperoleh dukungan empiris.
