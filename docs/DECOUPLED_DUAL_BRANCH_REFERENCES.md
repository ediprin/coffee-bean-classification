# Referensi Decoupled Dual-Branch GAP-HBP

Dokumen ini mencatat dasar literatur untuk eksperimen shared-early,
branched-late GAP-HBP. Arsitektur final merupakan sintesis terkontrol dari
beberapa paper; tidak boleh diklaim sebagai salinan satu metode tertentu.

## Referensi inti

### 1. Hierarchical Bilinear Pooling (HBP)

Yu et al., *Hierarchical Bilinear Pooling for Fine-Grained Visual
Recognition*, ECCV 2018.

- Dasar cabang HBP.
- Menggabungkan beberapa cross-layer bilinear feature untuk memodelkan
  hubungan bagian/fitur antar-layer.
- Sumber resmi:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.html

### 2. Detachable Second-Order Pooling (DSoP)

Li et al., *Detachable Second-Order Pooling: Toward High-Performance
First-Order Networks*, IEEE TNNLS 2022, DOI 10.1109/TNNLS.2021.3052829.

- Referensi paling dekat dengan tujuan efisiensi kita.
- Cabang auxiliary second-order dipasang pada beberapa stage CNN selama
  training untuk membantu first-order backbone mempelajari representasi yang
  lebih diskriminatif.
- Cabang second-order dapat dilepas setelah training sehingga inference tetap
  memakai first-order network.
- Sumber:
  https://pubmed.ncbi.nlm.nih.gov/33523818/
- Manuskrip:
  https://www4.comp.polyu.edu.hk/~cslzhang/paper/TNNLS-DSoP.pdf

### 3. On-the-Fly Native Ensemble (ONE)

Lan et al., *Knowledge Distillation by On-the-Fly Native Ensemble*, NeurIPS
2018.

- Mendukung pembagian shared low-level stages dan independent late branches.
- Setiap branch memiliki classifier sendiri.
- Gating component membentuk ensemble teacher selama satu training process.
- Relevan untuk versi GAP-HBP yang tetap memakai kedua cabang saat inference
  atau melakukan online transfer ke main branch.
- Sumber resmi:
  https://proceedings.neurips.cc/paper/2018/hash/94ef7214c4a90790186e255304f8fd1f-Abstract.html

### 4. Online KD with Diverse Peers (OKDDip)

Chen et al., *Online Knowledge Distillation with Diverse Peers*, AAAI 2020,
DOI 10.1609/aaai.v34i04.5746.

- Menunjukkan bahwa simple aggregation dapat membuat peer cepat homogen dan
  menghilangkan diversity.
- Memakai target aggregation yang berbeda untuk mempertahankan spesialisasi
  peer, kemudian mentransfer ensemble ke group leader.
- Menjadi dasar untuk mengaudit agreement GAP-HBP sebelum menambah diversity
  loss secara sembarang.
- Sumber resmi:
  https://ojs.aaai.org/index.php/AAAI/article/view/5746

### 5. Mixture of Granularity-Specific Experts

Zhang et al., *Learning a Mixture of Granularity-Specific Experts for
Fine-Grained Categorization*, ICCV 2019.

- Secara khusus membahas expert diversity pada data fine-grained yang
  terbatas.
- Menggunakan gradually-enhanced experts dan KL-based constraint agar expert
  mempelajari subproblem yang berbeda.
- Mendukung hipotesis bahwa GAP dan HBP harus mempertahankan spesialisasi,
  bukan dipaksa menjadi dua classifier identik.
- Sumber resmi:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Learning_a_Mixture_of_Granularity-Specific_Experts_for_Fine-Grained_Categorization_ICCV_2019_paper.html

## Referensi optimisasi konflik branch

### 6. PCGrad

Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020.

- Memproyeksikan gradient ketika dua objective memiliki dot product negatif.
- Relevan jika CE GAP dan CE HBP terbukti sering menghasilkan gradient yang
  berlawanan pada shared encoder.
- Bukan komponen default; gradient cosine harus diukur dahulu.
- Paper: https://arxiv.org/abs/2001.06782

### 7. GradNorm

Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in
Deep Multitask Networks*, ICML 2018.

- Menyeimbangkan magnitudo gradient beberapa loss secara dinamis.
- Relevan jika salah satu branch mendominasi shared encoder meskipun arah
  gradient tidak selalu berlawanan.
- Sumber resmi: https://proceedings.mlr.press/v80/chen18a.html

### 8. Cross-Stitch Networks

Misra et al., *Cross-Stitch Networks for Multi-Task Learning*, CVPR 2016.

- Mempelajari kombinasi shared dan task-specific representation.
- Menjadi referensi alternatif jika hard branch point ternyata terlalu kaku.
- Tidak dipakai pada versi pertama agar jumlah faktor eksperimen tetap kecil.
- Sumber resmi:
  https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html

## Referensi pooling/gating tambahan

Wang et al., *Global Gated Mixture of Second-order Pooling for Improving Deep
Convolutional Neural Networks*, NeurIPS 2018, menunjukkan input-dependent gate
untuk memilih beberapa second-order pooling candidate. Ini mendukung konsep
adaptive pooling, tetapi tidak langsung membuktikan bahwa gate GAP-HBP akan
stabil pada Coffee-17.

Sumber resmi:
https://papers.nips.cc/paper/2018/hash/17c276c8e723eb46aef576537e9d56d0-Abstract.html

## Sintesis arsitektur yang dapat diuji

```text
Input
  |
shared MobileNetV3 early/mid stages
  |-------------------------------|
late GAP branch               late HBP branch
  |                               |
GAP + classifier             HBP + classifier
  |                               |
logits_gap                    logits_hbp
  |-------------------------------|
     small fusion/gating head
              |
         fused logits
```

Objective awal yang paling mudah diaudit:

```text
L = CE_fused + lambda_gap * CE_gap + lambda_hbp * CE_hbp
```

Kedua auxiliary CE mencegah gate membuat satu branch mati. Input fusion dapat
di-detach sebagai modifikasi penelitian untuk membatasi gradient fusion agar
tidak menghomogenkan kedua expert. Modifikasi detach ini harus diuji dengan
ablation dan tidak boleh diatribusikan langsung kepada ONE atau DSoP.

## Urutan eksperimen yang disarankan

1. Baseline terpisah M0 (GAP) dan M1 (HBP) tetap menjadi kontrol.
2. D1: shared-early/branched-late, classifier masing-masing, fixed average
   logits. Ini menguji arsitektur tanpa kapasitas gate.
3. D2: D1 + small learned gate dengan auxiliary CE.
4. Audit agreement, branch-only accuracy, gate weight, dan cosine gradient.
5. Tambahkan PCGrad hanya jika gradient conflict terbukti sering terjadi.
6. Tambahkan diversity constraint hanya jika dua branch terbukti homogen.
7. D3 detachable: buang HBP branch saat inference dan nilai GAP main branch,
   mengikuti pertanyaan utama DSoP.

Screening satu seed harus memakai validation. Test tiga seed hanya dijalankan
setelah kriteria validation dikunci. Selain Macro-F1, Hard-F1, dan Worst-F1,
laporkan parameter aktif, model size, latency, serta biaya training.

## Batas klaim

- ONE, DSoP, PCGrad, dan GradNorm bukan paper khusus kopi.
- Decoupled GAP-HBP adalah adaptasi untuk hipotesis Coffee-17.
- Keuntungan tidak boleh diklaim berasal dari fusion bila capacity-matched
  control belum diuji.
- Jika D3 berhasil, klaim yang tepat adalah HBP sebagai training-time auxiliary
  supervision, bukan HBP sebagai inference head.
