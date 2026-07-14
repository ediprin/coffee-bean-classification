import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.config import load_config
from bilinear_lmmd.models import (
    AdaptationModel,
    ArcMarginClassifier,
    FixedFusionCBAM,
    LGFCBAM,
    build_model,
)


def test_arcface_applies_margin_only_to_target_and_backpropagates():
    classifier = ArcMarginClassifier(
        in_features=8,
        num_classes=4,
        scale=16.0,
        margin=0.3,
    )
    embedding = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 2, 1])

    inference_logits = classifier(embedding)
    training_logits = classifier(embedding, labels)
    target = torch.arange(labels.shape[0])
    assert torch.all(training_logits[target, labels] < inference_logits[target, labels])

    non_target_mask = torch.ones_like(training_logits, dtype=torch.bool)
    non_target_mask[target, labels] = False
    assert torch.allclose(
        training_logits[non_target_mask],
        inference_logits[non_target_mask],
    )

    torch.nn.functional.cross_entropy(training_logits, labels).backward()
    assert embedding.grad is not None
    assert classifier.weight.grad is not None


def test_arcface_model_uses_margin_for_training_and_cosine_for_inference():
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="gap",
        out_indices=(4,),
        classifier="arcface",
        arcface_scale=16.0,
        arcface_margin=0.3,
        pretrained=False,
    )
    model.eval()
    images = torch.randn(2, 3, 96, 96)
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        inference = model(images)
        training = model(images, labels=labels)

    rows = torch.arange(labels.shape[0])
    assert inference.logits.shape == (2, 4)
    assert torch.all(training.logits[rows, labels] < inference.logits[rows, labels])


@pytest.mark.parametrize(
    ("config_path", "head", "classifier"),
    [
        ("configs/F0_mobilenetv3_gap_320_ce_source.yaml", "gap", "linear"),
        ("configs/F1_mobilenetv3_hbp_320_ce_source.yaml", "hbp", "linear"),
        ("configs/F2_mobilenetv3_gap_320_arcface_source.yaml", "gap", "arcface"),
        ("configs/F3_mobilenetv3_hbp_320_arcface_source.yaml", "hbp", "arcface"),
    ],
)
def test_finegrained_configs_form_controlled_ablation(
    config_path, head, classifier
):
    cfg = load_config(config_path)
    assert cfg["data"]["image_size"] == 320
    assert cfg["model"]["head"] == head
    assert cfg["model"]["classifier"] == classifier
    assert cfg["adaptation"]["method"] == "source_only"


@pytest.mark.parametrize(
    "head",
    [
        "gap",
        "bilinear",
        "hbp",
        "hbp_mlp",
        "gap_hbp_fusion",
        "hbp_residual_control",
        "gap_hbp_residual",
        "gap_fixed_cbam",
        "gap_lgf_cbam",
    ],
)
def test_mobilenetv3_output_shapes(head):
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head=head,
        out_indices=(1, 3, 4),
        projection_dim=32,
        pretrained=False,
        enable_domain_classifier=True,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 96, 96), domain_strength=1.0)
    assert output.logits.shape == (2, 4)
    assert output.embedding.ndim == 2
    assert output.domain_logits.shape == (2, 2)


def test_m0b_and_hbp_have_matched_embedding_classifier_and_capacity():
    bilinear_cfg = load_config("configs/M0b_mobilenetv3_bilinear_source.yaml")
    hbp_cfg = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    bilinear_cfg["model"]["pretrained"] = False
    hbp_cfg["model"]["pretrained"] = False
    bilinear = build_model(bilinear_cfg["model"])
    hbp = build_model(hbp_cfg["model"])

    assert bilinear.pool.output_dim == hbp.pool.output_dim == 1536
    assert bilinear.classifier.in_features == hbp.classifier.in_features == 1536
    bilinear_params = sum(parameter.numel() for parameter in bilinear.parameters())
    hbp_params = sum(parameter.numel() for parameter in hbp.parameters())
    assert abs(bilinear_params - hbp_params) / hbp_params < 0.01


def test_hbp_mlp_and_gap_hbp_fusion_have_matched_capacity():
    control_cfg = load_config("configs/M1c_mobilenetv3_hbp_mlp_source.yaml")
    fusion_cfg = load_config("configs/M1f_mobilenetv3_gap_hbp_fusion_source.yaml")
    control_cfg["model"]["pretrained"] = False
    fusion_cfg["model"]["pretrained"] = False
    control = build_model(control_cfg["model"])
    fusion = build_model(fusion_cfg["model"])

    control_params = sum(parameter.numel() for parameter in control.parameters())
    fusion_params = sum(parameter.numel() for parameter in fusion.parameters())
    assert abs(control_params - fusion_params) / fusion_params < 0.001


def test_gap_hbp_fusion_backpropagates_through_both_branches():
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="gap_hbp_fusion",
        out_indices=(1, 3, 4),
        projection_dim=32,
        fusion_hbp_dim=24,
        fusion_gap_dim=12,
        pretrained=False,
    )
    output = model(torch.randn(2, 3, 96, 96))
    output.logits.sum().backward()

    assert model.pool.hbp_projector[0].weight.grad is not None
    assert model.pool.gap_projector[0].weight.grad is not None


def test_residual_control_and_fusion_have_matched_capacity():
    control_cfg = load_config(
        "configs/M1rc_mobilenetv3_hbp_residual_control_source.yaml"
    )
    fusion_cfg = load_config(
        "configs/M1r_mobilenetv3_gap_hbp_residual_source.yaml"
    )
    control_cfg["model"]["pretrained"] = False
    fusion_cfg["model"]["pretrained"] = False
    control = build_model(control_cfg["model"])
    fusion = build_model(fusion_cfg["model"])

    control_params = sum(parameter.numel() for parameter in control.parameters())
    fusion_params = sum(parameter.numel() for parameter in fusion.parameters())
    assert abs(control_params - fusion_params) / fusion_params < 0.001


def test_residual_fusion_keeps_full_hbp_embedding_unchanged():
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="gap_hbp_residual",
        out_indices=(1, 3, 4),
        projection_dim=32,
        residual_gap_dim=12,
        pretrained=False,
    )
    model.eval()
    with torch.no_grad():
        features = model.encoder(torch.randn(2, 3, 96, 96))
        hbp = model.pool.hbp(features)
        fused = model.pool(features)

    assert fused.shape[1] == hbp.shape[1] + 12
    assert torch.equal(fused[:, : hbp.shape[1]], hbp)


def test_lgf_cbam_starts_as_fixed_fusion_and_gates_sum_to_one():
    fixed = FixedFusionCBAM(channels=12, reduction=4)
    learnable = LGFCBAM(channels=12, reduction=4)
    learnable.paths.load_state_dict(fixed.paths.state_dict())
    feature = torch.randn(3, 12, 9, 9)

    fixed_output = fixed(feature)
    learnable_output = learnable(feature)

    assert torch.allclose(fixed_output, learnable_output)
    assert learnable.last_gate_weights is not None
    assert torch.allclose(
        learnable.last_gate_weights,
        torch.full((3, 2), 0.5),
    )
    assert torch.allclose(
        learnable.last_gate_weights.sum(dim=1), torch.ones(3)
    )


def test_lgf_cbam_backpropagates_into_attention_and_gate():
    module = LGFCBAM(channels=12, reduction=4)
    output = module(torch.randn(2, 12, 9, 9))
    output.square().mean().backward()

    assert module.paths.channel_mlp[0].weight.grad is not None
    assert module.paths.spatial.weight.grad is not None
    assert module.gate_mlp[-1].weight.grad is not None


def test_fixed_and_lgf_models_share_same_seed_initialization():
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (4,),
        "attention_reduction": 4,
        "pretrained": False,
    }
    torch.manual_seed(17)
    fixed = AdaptationModel(head="gap_fixed_cbam", **kwargs)
    torch.manual_seed(17)
    learnable = AdaptationModel(head="gap_lgf_cbam", **kwargs)

    for fixed_parameter, learnable_parameter in zip(
        fixed.encoder.parameters(), learnable.encoder.parameters()
    ):
        assert torch.equal(fixed_parameter, learnable_parameter)
    for fixed_parameter, learnable_parameter in zip(
        fixed.pool.attention.paths.parameters(),
        learnable.pool.attention.paths.parameters(),
    ):
        assert torch.equal(fixed_parameter, learnable_parameter)
    assert torch.equal(fixed.classifier.weight, learnable.classifier.weight)
    assert torch.equal(fixed.classifier.bias, learnable.classifier.bias)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/M0a_mobilenetv3_gap_fixed_cbam_source.yaml",
        "configs/M0lgf_mobilenetv3_gap_lgf_cbam_source.yaml",
    ],
)
def test_attention_gap_configs_use_only_deepest_feature(config_path):
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 96, 96))

    assert tuple(cfg["model"]["out_indices"]) == (4,)
    assert output.logits.shape == (2, 17)
    assert output.embedding.shape[1] == model.pool.output_dim
