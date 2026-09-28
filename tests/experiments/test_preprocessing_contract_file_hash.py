import json
from bilinear_lmmd.experiments.preprocessing_contract import validate_development

def test_validate_development_uses_fold_manifest_file_sha(tmp_path):
    root=tmp_path/"dev"
    (root/"source/train/A").mkdir(parents=True)
    (root/"source/val/A").mkdir(parents=True)
    contract={
        "format":"bilinear_lmmd.coffee17.preprocessing_development.v1",
        "decision":"PASS",
        "fold":1,
        "clean_content_sha256":"clean",
        "fold_manifest_file_sha256":"fold-file-sha",
        "test_images_extracted":False,
        "training_executed":False,
    }
    path=root/"development_contract.json"
    path.write_text(json.dumps(contract),encoding="utf-8")
    result=validate_development(root,path)
    assert result["fold_manifest_sha256"]=="fold-file-sha"
