# SNI classification dataset v2

## Status

**Dataset-manifest protocol only. Training is not authorized by this
document.**

This protocol converts the audited 21-class SNI instance-crop v1 dataset into
a statistically and semantically safer classification package without copying,
deleting, or re-encoding its 31,074 crop images.

## Problems addressed

The v1 labels combine three different concepts in one flat softmax:

1. coffee-bean condition;
2. object/material family;
3. large/medium/small grading.

The last concept is not a valid crop-only visual target. Every crop is resized
to the same network input size, and neither public dataset provides a physical
scale calibration. A small object can therefore occupy the same 224x224 tensor
as a large object.

The source annotations also provide one category per object. They do not prove
that every other defect is absent. A v2 manifest must not invent complete
multi-label negatives from these single-category annotations.

## Visual target

The primary crop-only target contains 15 classes:

- the existing 12 bean-condition classes;
- all coffee-skin sizes collapsed to `kulit_kopi`;
- all parchment-skin sizes collapsed to `kulit_tanduk`;
- all foreign-matter sizes collapsed to `benda_asing`.

The original 21-class label and size level remain in the manifest as metadata.
They are not destroyed.

## Hierarchy

Every sample receives one family:

```text
coffee_bean
coffee_skin
parchment_skin
foreign_matter
```

For bean samples, the manifest also provides partial attributes such as
`black`, `broken`, `one_hole`, and `multiple_holes`.

Attribute values are:

```text
 1 = explicitly positive in the source label
 0 = explicitly negative; currently only justified by a normal-bean label
-1 = unobserved, not negative
```

For example, `biji_hitam_pecah` has `black=1` and `broken=1`. Other attributes
remain unknown unless the label explicitly supports them.

## Split and leakage rules

The v2 package reuses the grouped v1 train/validation/test assignment. It
verifies:

- every crop file exists;
- every path stays inside the v1 root;
- every source group belongs to exactly one split;
- no exact crop hash crosses splits;
- no exact crop hash has conflicting labels;
- all classes and both public source datasets are present.

No random crop-level resplitting is allowed.

## Imbalance

No crop is removed or duplicated. The training manifest contains two
normalized sample-weight alternatives:

1. inverse square-root class frequency;
2. effective-number weighting.

The default recommendation is inverse square-root weighting. Validation and
test retain their natural grouped distributions and receive no weights.

## Independent evidence and cross-domain evaluation

Crop count is not treated as independent sample count. The audit reports both
crop count and unique source-group count for every visual class and split.

Two external protocols are generated:

```text
Adrian train/val -> Faruq external test
Faruq train/val  -> Adrian external test
```

Groups present in both domains are excluded. These protocols are intended for
future generalization evaluation and must not be used to tune on the external
test domain.

## Statistical readiness

The default strong-claim threshold is:

- at least 50 validation/test crops per visual class; and
- at least 20 independent source groups per visual class.

Failure does not corrupt the manifest. It means per-class model claims are not
yet statistically supported and more independent source images or a justified
class merge are required.

## Command

```bash
python -u -m bilinear_lmmd.data.preparation.prepare_sni_classification_v2 \
  --input-root /content/sni-instance-crops \
  --output-root /content/drive/MyDrive/coffee-sni-instance-crop-v1/classification-v2
```

The output is small because images remain in the audited v1 root:

```text
classification-v2/
  audit.json
  ontology.json
  manifests/
    all.csv
    train.csv
    train_weighted.csv
    val.csv
    test.csv
  cross_domain/
    adrian_detection_to_faruq_segmentation/
    faruq_segmentation_to_adrian_detection/
```

## Gate before training

Training remains blocked until:

1. `audit.json` has `status=complete`;
2. weak validation/test classes are reviewed;
3. the 15-class ontology is visually approved;
4. the selected weighting rule is frozen;
5. the internal or cross-domain evaluation question is frozen;
6. test remains locked.
