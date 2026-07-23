from __future__ import annotations


SNI_CLASSES = (
    "biji_berkulit_tanduk",
    "biji_berlubang_lebih_satu",
    "biji_berlubang_satu",
    "biji_bertutul_tutul",
    "biji_coklat",
    "biji_hitam",
    "biji_hitam_pecah",
    "biji_hitam_sebagian",
    "biji_muda",
    "biji_normal",
    "biji_pecah",
    "kopi_gelondong",
    "kulit_kopi_ukuran_besar",
    "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang",
    "kulit_tanduk_ukuran_besar",
    "kulit_tanduk_ukuran_kecil",
    "kulit_tanduk_ukuran_sedang",
    "tanah_batu_ranting_besar",
    "tanah_batu_ranting_kecil",
    "tanah_batu_ranting_sedang",
)

# Class order is deliberately contiguous per group. ImageFolder sorts the same
# canonical names lexicographically, so the conditional experts can reconstruct
# a normalized 21-class distribution by concatenating their local outputs.
SNI_GROUPS = {
    "kondisi_biji": SNI_CLASSES[0:12],
    "kulit_kopi": SNI_CLASSES[12:15],
    "kulit_tanduk": SNI_CLASSES[15:18],
    "benda_asing": SNI_CLASSES[18:21],
}
SNI_GROUP_NAMES = tuple(SNI_GROUPS)
SNI_GROUP_SIZES = tuple(len(classes) for classes in SNI_GROUPS.values())

# Metadata from SNI 01-2907-2008. These values are not class-loss weights and
# are not an output of the v1 classifier; they remain available for a separate
# grading system if that scope is ever authorized.
SNI_DEFECT_WEIGHTS = {
    "biji_hitam": 1.0,
    "biji_hitam_sebagian": 0.5,
    "biji_hitam_pecah": 0.5,
    "kopi_gelondong": 1.0,
    "biji_coklat": 0.25,
    "kulit_kopi_ukuran_besar": 1.0,
    "kulit_kopi_ukuran_sedang": 0.5,
    "kulit_kopi_ukuran_kecil": 0.2,
    "biji_berkulit_tanduk": 0.5,
    "kulit_tanduk_ukuran_besar": 0.5,
    "kulit_tanduk_ukuran_sedang": 0.2,
    "kulit_tanduk_ukuran_kecil": 0.1,
    "biji_pecah": 0.2,
    "biji_muda": 0.2,
    "biji_berlubang_satu": 0.1,
    "biji_berlubang_lebih_satu": 0.2,
    "biji_bertutul_tutul": 0.1,
    "tanah_batu_ranting_besar": 5.0,
    "tanah_batu_ranting_sedang": 2.0,
    "tanah_batu_ranting_kecil": 1.0,
    "biji_normal": 0.0,
}


def validate_sni_classes(classes: list[str] | tuple[str, ...]) -> None:
    observed = tuple(classes)
    if observed != SNI_CLASSES:
        raise ValueError(
            "Urutan kelas dataset tidak cocok dengan ontologi SNI v1. "
            f"Expected={list(SNI_CLASSES)}, observed={list(observed)}"
        )
