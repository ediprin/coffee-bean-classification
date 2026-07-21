# Protokol adaptasi klasifikasi Hong: DSConv x SPPF-Attention

## Tujuan

Menguji apakah dua komponen representasi dari Hong et al. (2026) berguna pada
closed-set fine-grained classification Coffee17. Eksperimen ini **bukan**
reproduksi YOLOv10 dan tidak memakai HBP, detection head, bounding-box loss,
atau PConv.

Referensi:

- Hong et al. (2026), `docs/Hong et al. - 2026 - Automated detection of
  defective coffee beans based on improved YOLOv10 framework.pdf`;
- Nascimento, Fawcett, dan Prisacariu (ICCV 2019), *DSConv: Efficient
  Convolution Operator*: https://openaccess.thecvf.com/content_ICCV_2019/html/do_Nascimento_DSConv_Efficient_Convolution_Operator_ICCV_2019_paper.html;
- implementasi resmi DSConv: https://github.com/ActiveVisionLab/DSConv.

## Klarifikasi DSConv

DSConv Hong bukan singkatan sederhana untuk depthwise-separable convolution.
Operator yang dirujuk adalah **Distribution Shifting Convolution** yang
memfaktorkan kernel menjadi:

```text
Variable Quantized Kernel (VQK)
          +
Kernel Distribution Shift (KDS)
          +
Channel Distribution Shift (CDS)
```

Adaptasi ini memakai VQK 4-bit dan block size 128, mengikuti setting praktis
utama paper operator asli. Lima full spatial convolution pada stage awal dan
menengah EfficientNetV2-B0 diganti:

```text
blocks.0.0.conv
blocks.1.0.conv_exp
blocks.1.1.conv_exp
blocks.2.0.conv_exp
blocks.2.1.conv_exp
```

Depthwise convolution dan proyeksi 1x1 tidak diganti. Paper asli mencatat
compact/depthwise networks memiliki redundansi lebih rendah dan lebih rentan
terhadap kehilangan akurasi akibat quantization.

Implementasi PyTorch merekonstruksi kernel secara float dengan straight-through
estimator agar dapat di-fine-tune. Nilai penyimpanan kernel integer dapat
dihitung secara teoretis, tetapi **tidak boleh diklaim sebagai percepatan atau
ukuran checkpoint aktual** tanpa backend integer khusus.

## Adaptasi SPPF-Attention

Implementasi mengikuti Persamaan 8-12 Hong:

1. reduksi channel 1x1;
2. tiga max-pooling 5x5 berurutan, stride 1, padding 2;
3. concatenation empat skala dan proyeksi 1x1;
4. channel attention dari GAP-MLP;
5. spatial attention dari channel average/max dan conv 7x7;
6. residual ke feature map masukan;
7. GAP dan linear classifier.

SPPF-Attention pernah gagal pada MobileNetV3. Pengujian ulang ini dibenarkan
hanya sebagai kontrol interaksi dengan DSConv pada backbone EfficientNetV2-B0,
bukan sebagai tuning ulang hasil lama.

## Ablasi faktorial validation-only

| Kode | DSConv | SPPF-Attention | Pool/classifier |
|---|---:|---:|---|
| BE2G | tidak | tidak | GAP + linear + CE |
| HCD1 | ya | tidak | GAP + linear + CE |
| HCS1 | tidak | ya | GAP + linear + CE |
| HCDS1 | ya | ya | GAP + linear + CE |

BE2H (EfficientNetV2-B0 + HBP + CE) hanya menjadi baseline performa lama, bukan
bagian faktorial. Semua model memakai input 224, split bersih/grouped, scheduler,
augmentasi, dan seed yang sama.

## Gate fail-fast

Screening pertama memakai seed 123 dan validation. Test terkunci. Setiap
perbandingan PASS bila:

- Macro-F1 naik;
- Hard-F1 naik;
- Worst-F1 tidak turun lebih dari 1 poin.

HCDS1 final hanya PASS jika sekaligus:

1. mengalahkan BE2H;
2. mengalahkan HCD1, membuktikan tambahan SPPF bermanfaat ketika DSConv aktif;
3. mengalahkan HCS1, membuktikan tambahan DSConv bermanfaat ketika SPPF aktif.

Jika final FAIL, eksperimen berhenti tanpa tiga seed, tanpa test, dan tanpa
mengubah bit/block/stage berdasarkan validation.

## Menjalankan

```bash
python -u -m bilinear_lmmd.experiments.run_hong_classification_screening \
  --data-root data/coffee17-clean/folds/fold_1 \
  --baseline-root outputs/backbone-results \
  --output-root outputs/hong-classification \
  --seeds 123 \
  --models HCD1 HCS1 HCDS1 \
  --evaluation-split val
```

Notebook Colab:
`notebooks/coffee17_hong_classification_colab.ipynb`.

## Batas klaim

- Hasil hanya menguji transfer komponen Hong ke klasifikasi satu-biji.
- Tidak membuktikan performa detector atau conveyor.
- Tidak membuktikan kepatuhan SNI 2907.
- Tidak boleh menyebut DSConv lebih cepat berdasarkan latency PyTorch simulasi.
- Tidak boleh memakai angka Hong sebagai target langsung karena task, label,
  backbone, resolusi, dan protokol split berbeda.
