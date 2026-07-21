# Protokol SNI Instance-Crop v1

## Tujuan

Menyusun dataset **klasifikasi per objek** yang konsisten dari dua dataset
publik beranotasi COCO:

1. `yolo-skripsi-2-lh14g-y61eh` (bounding box, 27 label);
2. `robusta_sni_dataset-hr9ci` (instance segmentation, 21 label).

Unit masukan classifier bukan foto penuh. Setiap anotasi objek menghasilkan
satu crop biji atau benda asing. Dengan demikian foto yang memuat banyak objek
tetap dapat dipakai untuk klasifikasi, tetapi evaluasi ini masih merupakan
**oracle-crop classification**, bukan evaluasi detector end-to-end.

## Ruang label bersama: 21 kelas

Kedua dataset dinormalisasi ke label SNI berikut:

- 12 kondisi biji: `biji_berkulit_tanduk`,
  `biji_berlubang_lebih_satu`, `biji_berlubang_satu`,
  `biji_bertutul_tutul`, `biji_coklat`, `biji_hitam`,
  `biji_hitam_pecah`, `biji_hitam_sebagian`, `biji_muda`, `biji_normal`,
  `biji_pecah`, dan `kopi_gelondong`;
- tiga ukuran kulit kopi;
- tiga ukuran kulit tanduk;
- tiga ukuran benda asing.

Dataset Adrian memisahkan `tanah`, `batu`, dan `ranting`, sedangkan dataset
Faruq menggabungkannya. Agar label memiliki arti yang sama di kedua sumber,
sembilan label Adrian berikut dipetakan berdasarkan ukuran:

```text
{tanah,batu,ranting} x {besar,sedang,kecil}
                         |
                         v
tanah_batu_ranting_{besar,sedang,kecil}
```

Penggabungan ini tidak mengubah perhitungan nilai cacat SNI karena ketiga jenis
benda asing memiliki bobot yang sama pada ukuran yang sama. Bobot SNI disimpan
sebagai metadata; bobot tersebut **bukan** bobot loss klasifikasi.

## Audit dan pembentukan split

Split bawaan tidak langsung dipakai karena audit anotasi menemukan dua masalah:

- dataset Adrian mempunyai train/validation yang hampir seluruhnya satu objek,
  tetapi test berisi foto padat sekitar 30 objek per gambar;
- nama sumber Roboflow yang sama muncul pada beberapa split dataset Faruq.

Pipeline membentuk ulang split dengan target 70/15/15 dan aturan berikut:

1. suffix ekspor Roboflow `.rf.<hash>` dihapus untuk memperoleh identitas foto
   sumber;
2. SHA-256 file gambar dihitung untuk menemukan salinan piksel identik;
3. gambar dengan identitas sumber yang sama atau hash identik disatukan dalam
   satu grup;
4. distribusi kelas dan asal dataset diseimbangkan dengan grouped multilabel
   allocation;
5. seluruh objek dari satu grup foto selalu masuk ke split yang sama;
6. setiap kelas wajib hadir pada train, validation, dan test;
7. crop identik dengan label yang sama dideduplikasi dengan prioritas
   mempertahankan test, lalu validation, lalu train;
8. crop identik dengan label berbeda menyebabkan pipeline berhenti karena
   konflik anotasi.

Rasio 70/15/15 adalah target. Rasio aktual dapat sedikit berbeda karena satu
foto padat tidak boleh dipecah antar-split. Nilai aktual dicatat di
`audit.json`.

## Pembentukan crop

- sumber lokasi: COCO bounding box;
- Faruq: polygon tetap diaudit, tetapi common input menggunakan bbox-nya;
- crop berbentuk persegi, berpusat pada bbox;
- orientasi EXIF diterapkan sebelum bbox COCO digunakan; ukuran hasil orientasi
  wajib sama dengan metadata COCO;
- margin konteks: 10% pada setiap sisi;
- area di luar gambar dipad dengan rata-rata RGB foto sumber;
- tidak ada resize atau augmentasi offline;
- resize dan augmentasi hanya dilakukan loader model, dan augmentasi hanya pada
  train;
- mask tidak diberikan kepada classifier.

Semua transformasi disimpan di `manifest.csv`, termasuk dataset asal, foto
sumber, group ID, split baru, label asli, label canonical, bbox, dan hash crop.

## Tahapan eksperimen

1. Jalankan preparasi dan periksa `audit.json` serta dua contact sheet.
2. Bekukan mapping, margin, dan split.
3. Training dan seleksi model hanya menggunakan train/validation.
4. Test tetap terkunci sampai kandidat final dipilih.
5. Laporkan hasil gabungan dan hasil per dataset asal agar model tidak menang
   hanya karena domain atau kelas normal dari satu dataset.
6. Setelah oracle-crop classification valid, ukur sistem end-to-end dengan
   detector yang menghasilkan crop pada foto baru.

Baseline minimum tahap berikutnya adalah EfficientNetV2-B0 + GAP. Kandidat
fine-grained dibandingkan dalam faktorial terkontrol dengan backbone dan split
yang sama; belum ada model yang dianggap unggul sebelum eksperimen ini.

## Perintah

```bash
python -u -m bilinear_lmmd.data.preparation.prepare_sni_instance_crops \
  --adrian-root /content/sni-raw/adrian_detection \
  --faruq-root /content/sni-raw/faruq_segmentation \
  --output-root /content/sni-instance-crops \
  --seed 42 \
  --margin 0.10
```

Output ImageFolder:

```text
sni-instance-crops/
  source/
    train/<21 kelas>/*.jpg
    val/<21 kelas>/*.jpg
    test/<21 kelas>/*.jpg
  manifest.csv
  audit.json
  contact_sheet_adrian_detection.jpg
  contact_sheet_faruq_segmentation.jpg
```
