from __future__ import annotations
import json, zipfile
from pathlib import Path
import pytest
from PIL import Image
from bilinear_lmmd.data.preparation.audit_coffee17_provenance import audit_coffee17_provenance
from bilinear_lmmd.data.preparation.prepare_preprocessing_folds import prepare_preprocessing_folds
from bilinear_lmmd.data.preparation.materialize_preprocessing_development import materialize_preprocessing_development
from bilinear_lmmd.data.preparation.materialize_preprocessing_test import materialize_preprocessing_test

def _img(p:Path,c):
    p.parent.mkdir(parents=True,exist_ok=True); Image.new("RGB",(16,16),c).save(p)
def _zip(tmp:Path):
    src=tmp/"src"; counts={"A":7,"B":7}
    for ci,(cls,n) in enumerate(counts.items()):
        for i in range(n): _img(src/cls/f"{i}.png",(ci*100+i,i+10,200-i))
    z=tmp/"coffee.zip"
    with zipfile.ZipFile(z,"w") as b:
        for p in sorted(src.glob("*/*")): b.write(p,arcname=f"{p.parent.name}/{p.name}")
    return z,counts

def test_provenance_is_raw_only(tmp_path):
    z,c=_zip(tmp_path); e=tmp_path/"e"; raw=tmp_path/"raw"
    r=audit_coffee17_provenance(z,e,canonical_root=raw,expected_counts=c)
    assert r["decision"]=="PASS" and r["image_count"]==14 and r["split_created"] is False
    assert not (raw/"train").exists() and len(list(raw.glob("*/*.png")))==14

def test_folds_are_manifest_only_and_deterministic(tmp_path):
    z,c=_zip(tmp_path); e=tmp_path/"e"; raw=tmp_path/"raw"
    audit_coffee17_provenance(z,e,canonical_root=raw,expected_counts=c)
    a=prepare_preprocessing_folds(raw,e/"coffee17_provenance.json",tmp_path/"fa",folds=3,validation_ratio=.2)
    b=prepare_preprocessing_folds(raw,e/"coffee17_provenance.json",tmp_path/"fb",folds=3,validation_ratio=.2)
    assert a["fold_manifest_sha256"]==b["fold_manifest_sha256"]
    f=json.loads((tmp_path/"fa/fold_manifest.json").read_text())
    ids=[x for fold in f["assignments"].values() for x in fold["test"]]
    assert len(ids)==len(set(ids))==14 and not (tmp_path/"fa/fold_1").exists()

def test_development_never_materializes_test(tmp_path):
    z,c=_zip(tmp_path); e=tmp_path/"e"; raw=tmp_path/"raw"; folds=tmp_path/"folds"
    audit_coffee17_provenance(z,e,canonical_root=raw,expected_counts=c)
    prepare_preprocessing_folds(raw,e/"coffee17_provenance.json",folds,folds=3,validation_ratio=.2)
    d=tmp_path/"dev"; r=materialize_preprocessing_development(raw,folds/"clean_manifest.json",folds/"fold_manifest.json",d,fold=1)
    assert r["test_images_extracted"] is False and (d/"source/train").is_dir() and (d/"source/val").is_dir() and not (d/"source/test").exists()

def test_test_materializer_requires_explicit_authority(tmp_path):
    z,c=_zip(tmp_path); e=tmp_path/"e"; raw=tmp_path/"raw"; folds=tmp_path/"folds"
    audit_coffee17_provenance(z,e,canonical_root=raw,expected_counts=c)
    prepare_preprocessing_folds(raw,e/"coffee17_provenance.json",folds,folds=3,validation_ratio=.2)
    clean=json.loads((folds/"clean_manifest.json").read_text())
    auth=tmp_path/"authority.json"; auth.write_text(json.dumps({"decision":"AUTHORIZE_OOF_TEST_EVALUATION","clean_content_sha256":clean["clean_content_sha256"],"test_images_accessed":False,"further_primary_tuning_authorized":False}))
    with pytest.raises(RuntimeError,match="belum diotorisasi"):
        materialize_preprocessing_test(raw,folds/"clean_manifest.json",folds/"fold_manifest.json",auth,tmp_path/"test",fold=1,authorize_test=False)
    r=materialize_preprocessing_test(raw,folds/"clean_manifest.json",folds/"fold_manifest.json",auth,tmp_path/"test",fold=1,authorize_test=True)
    assert r["training_executed"] is False and r["test_images_accessed"] is True and (tmp_path/"test/source/test").is_dir()
