import pytest
import torch

pytest.importorskip("timm")

from bilinear_lmmd.config import load_config
from bilinear_lmmd.models import (
    AdaptationModel,
    ArcMarginClassifier,
    FixedFusionCBAM,
    CapacityResidualHBP,
    PointwiseResidualCapacity,
    LGFCBAM,
    SPPFAttention,
    SPPFAttentionGAP,
    SPPFAttentionHBP,
    SpatiallyPreservedHBP,
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
    ("config_path", "head", "classifier", "image_size"),
    [
        ("configs/A2_mobilenetv3_gap_224_arcface_source.yaml", "gap", "arcface", 224),
        ("configs/A3_mobilenetv3_hbp_224_arcface_source.yaml", "hbp", "arcface", 224),
        ("configs/F0_mobilenetv3_gap_320_ce_source.yaml", "gap", "linear", 320),
        ("configs/F1_mobilenetv3_hbp_320_ce_source.yaml", "hbp", "linear", 320),
        ("configs/F2_mobilenetv3_gap_320_arcface_source.yaml", "gap", "arcface", 320),
        ("configs/F3_mobilenetv3_hbp_320_arcface_source.yaml", "hbp", "arcface", 320),
    ],
)
def test_finegrained_configs_form_controlled_ablation(
    config_path, head, classifier, image_size
):
    cfg = load_config(config_path)
    assert cfg["data"]["image_size"] == image_size
    assert cfg["model"]["head"] == head
    assert cfg["model"]["classifier"] == classifier
    assert cfg["adaptation"]["method"] == "source_only"


@pytest.mark.parametrize(
    "head",
    [
        "gap",
        "bilinear",
        "hbp",
        "hbp_moe",
        "sp_hbp",
        "sppf_attention_gap",
        "sppf_attention_hbp",
        "capacity_residual_hbp",
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


def test_sp_hbp_preserves_hbp_shape_capacity_and_backpropagates():
    baseline = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="hbp",
        out_indices=(1, 3, 4),
        projection_dim=32,
        pretrained=False,
    )
    candidate = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="sp_hbp",
        out_indices=(1, 3, 4),
        projection_dim=32,
        hbp_spatial_size=14,
        pretrained=False,
    )

    assert isinstance(candidate.pool, SpatiallyPreservedHBP)
    assert candidate.pool.spatial_size == 14
    assert candidate.pool.output_dim == baseline.pool.output_dim == 96
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )

    output = candidate(torch.randn(2, 3, 96, 96))
    output.logits.sum().backward()
    assert output.embedding.shape == (2, 96)
    assert candidate.pool.projections[0][0].weight.grad is not None


def test_sp_hbp_rejects_invalid_grid_size():
    with pytest.raises(ValueError, match="lebih besar dari nol"):
        SpatiallyPreservedHBP([8, 12, 16], projection_dim=4, spatial_size=0)


def test_sp_hbp_config_is_controlled_against_hbp():
    baseline_cfg = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    candidate_cfg = load_config("configs/M1s_mobilenetv3_sp_hbp_source.yaml")

    assert baseline_cfg["model"]["head"] == "hbp"
    assert candidate_cfg["model"]["head"] == "sp_hbp"
    assert candidate_cfg["model"]["hbp_spatial_size"] == 14
    for key in (
        "backbone",
        "out_indices",
        "projection_dim",
        "classifier",
        "dropout",
    ):
        assert baseline_cfg["model"][key] == candidate_cfg["model"][key]
    assert baseline_cfg["data"] == candidate_cfg["data"]
    assert baseline_cfg["adaptation"] == candidate_cfg["adaptation"]


def test_hierarchical_hbp_adds_only_parent_head_and_backpropagates():
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="hbp",
        out_indices=(1, 3, 4),
        projection_dim=32,
        hierarchy_num_parents=2,
        pretrained=False,
    )
    output = model(torch.randn(2, 3, 96, 96))

    assert output.logits.shape == (2, 4)
    assert output.parent_logits.shape == (2, 2)
    output.parent_logits.sum().backward()
    assert model.parent_classifier.weight.grad is not None
    assert model.pool.projections[0][0].weight.grad is not None


def test_hierarchical_hbp_preserves_same_seed_core_initialization():
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "head": "hbp",
        "out_indices": (1, 3, 4),
        "projection_dim": 32,
        "pretrained": False,
    }
    torch.manual_seed(23)
    baseline = AdaptationModel(**kwargs)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(23)
    candidate = AdaptationModel(hierarchy_num_parents=2, **kwargs)

    for baseline_parameter, candidate_parameter in zip(
        baseline.encoder.parameters(), candidate.encoder.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    for baseline_parameter, candidate_parameter in zip(
        baseline.pool.parameters(), candidate.pool.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    assert torch.equal(baseline.classifier.bias, candidate.classifier.bias)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())


def test_hierarchical_config_is_controlled_against_hbp():
    baseline = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    candidate = load_config("configs/H1_mobilenetv3_hbp_hierarchical_source.yaml")

    assert candidate["hierarchy"]["enabled"] is True
    assert candidate["hierarchy"]["weight"] == 0.2
    assert candidate["model"]["hierarchy_num_parents"] == 14
    for key in ("backbone", "head", "out_indices", "projection_dim", "classifier", "dropout"):
        assert baseline["model"][key] == candidate["model"][key]
    assert baseline["data"] == candidate["data"]
    assert baseline["adaptation"] == candidate["adaptation"]


def test_sppf_attention_preserves_shape_and_backpropagates():
    module = SPPFAttention(channels=16, reduction=4)
    feature = torch.randn(2, 16, 9, 9, requires_grad=True)

    output = module(feature)
    assert output.shape == feature.shape
    output.square().mean().backward()
    assert feature.grad is not None
    assert module.reduce[0].weight.grad is not None
    assert module.channel_mlp[0].weight.grad is not None
    assert module.spatial.weight.grad is not None


def test_sppf_attention_gap_pools_refined_deep_feature_and_backpropagates():
    module = SPPFAttentionGAP(channels=16, attention_reduction=4)
    feature = torch.randn(2, 16, 9, 9, requires_grad=True)

    output = module([feature])
    assert output.shape == (2, 16)
    output.square().mean().backward()
    assert feature.grad is not None
    assert module.attention.reduce[0].weight.grad is not None


def test_sppf_attention_gap_preserves_same_seed_gap_initialization():
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (4,),
        "pretrained": False,
    }
    torch.manual_seed(37)
    baseline = AdaptationModel(head="gap", **kwargs)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(37)
    candidate = AdaptationModel(head="sppf_attention_gap", **kwargs)

    assert isinstance(candidate.pool, SPPFAttentionGAP)
    for baseline_parameter, candidate_parameter in zip(
        baseline.encoder.parameters(), candidate.encoder.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    assert torch.equal(baseline.classifier.bias, candidate.classifier.bias)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())


def test_sppf_attention_gap_config_is_controlled_against_gap():
    baseline = load_config("configs/M0_mobilenetv3_gap_source.yaml")
    candidate = load_config(
        "configs/S0_mobilenetv3_sppf_attention_gap_source.yaml"
    )

    assert baseline["model"]["head"] == "gap"
    assert candidate["model"]["head"] == "sppf_attention_gap"
    assert baseline["model"]["out_indices"] == candidate["model"]["out_indices"] == [4]
    for key in ("backbone", "classifier", "dropout"):
        assert baseline["model"][key] == candidate["model"][key]
    assert baseline["data"] == candidate["data"]
    assert baseline["adaptation"] == candidate["adaptation"]


def test_sppf_attention_hbp_preserves_same_seed_core_initialization():
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (1, 3, 4),
        "projection_dim": 32,
        "pretrained": False,
    }
    torch.manual_seed(29)
    baseline = AdaptationModel(head="hbp", **kwargs)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(29)
    candidate = AdaptationModel(head="sppf_attention_hbp", **kwargs)

    assert isinstance(candidate.pool, SPPFAttentionHBP)
    for baseline_parameter, candidate_parameter in zip(
        baseline.encoder.parameters(), candidate.encoder.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    for baseline_parameter, candidate_parameter in zip(
        baseline.pool.parameters(), candidate.pool.hbp.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    assert torch.equal(baseline.classifier.bias, candidate.classifier.bias)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())


def test_sppf_attention_hbp_config_is_controlled_against_hbp():
    baseline = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    candidate = load_config(
        "configs/S1_mobilenetv3_sppf_attention_hbp_source.yaml"
    )

    assert candidate["model"]["head"] == "sppf_attention_hbp"
    for key in ("backbone", "out_indices", "projection_dim", "classifier", "dropout"):
        assert baseline["model"][key] == candidate["model"][key]
    assert baseline["data"] == candidate["data"]
    assert baseline["adaptation"] == candidate["adaptation"]


def test_pointwise_capacity_control_preserves_shape_and_backpropagates():
    module = PointwiseResidualCapacity(channels=16, hidden_channels=21)
    feature = torch.randn(2, 16, 9, 9, requires_grad=True)

    output = module(feature)
    assert output.shape == feature.shape
    output.square().mean().backward()
    assert feature.grad is not None
    assert module.refine[0].weight.grad is not None
    assert module.refine[3].weight.grad is not None


def test_capacity_control_matches_sppf_parameters_and_core_initialization():
    baseline_cfg = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    control_cfg = load_config(
        "configs/C1_mobilenetv3_capacity_residual_hbp_source.yaml"
    )
    candidate_cfg = load_config(
        "configs/S1_mobilenetv3_sppf_attention_hbp_source.yaml"
    )
    for cfg in (baseline_cfg, control_cfg, candidate_cfg):
        cfg["model"]["pretrained"] = False

    torch.manual_seed(31)
    baseline = build_model(baseline_cfg["model"])
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(31)
    control = build_model(control_cfg["model"])
    torch.manual_seed(31)
    candidate = build_model(candidate_cfg["model"])

    assert isinstance(control.pool, CapacityResidualHBP)
    for baseline_parameter, control_parameter in zip(
        baseline.encoder.parameters(), control.encoder.parameters()
    ):
        assert torch.equal(baseline_parameter, control_parameter)
    for baseline_parameter, control_parameter in zip(
        baseline.pool.parameters(), control.pool.hbp.parameters()
    ):
        assert torch.equal(baseline_parameter, control_parameter)
    assert torch.equal(baseline.classifier.weight, control.classifier.weight)
    assert torch.equal(baseline.classifier.bias, control.classifier.bias)
    assert torch.equal(baseline_rng, torch.random.get_rng_state())

    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    candidate_parameters = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    assert abs(control_parameters - candidate_parameters) / candidate_parameters < 0.0001


def test_capacity_control_config_is_controlled_against_sppf():
    control = load_config(
        "configs/C1_mobilenetv3_capacity_residual_hbp_source.yaml"
    )
    candidate = load_config(
        "configs/S1_mobilenetv3_sppf_attention_hbp_source.yaml"
    )

    assert control["model"]["head"] == "capacity_residual_hbp"
    assert control["model"]["capacity_hidden_dim"] == 1259
    for key in ("backbone", "out_indices", "projection_dim", "classifier", "dropout"):
        assert control["model"][key] == candidate["model"][key]
    assert control["data"] == candidate["data"]
    assert control["adaptation"] == candidate["adaptation"]


def test_hbp_moe_outputs_experts_gate_and_backpropagates():
    model = AdaptationModel(
        backbone="mobilenetv3_small_050",
        num_classes=4,
        head="hbp_moe",
        out_indices=(1, 3, 4),
        projection_dim=32,
        moe_local_dim=16,
        moe_gate_hidden=8,
        moe_hbp_prior=0.8,
        pretrained=False,
    )
    model.eval()
    output = model(torch.randn(2, 3, 96, 96))

    assert output.expert_logits is not None
    assert set(output.expert_logits) == {"hbp_global", "local_gmp"}
    assert output.gate_weights is not None
    assert output.gate_weights.shape == (2, 2)
    assert torch.allclose(
        output.gate_weights,
        torch.tensor([[0.8, 0.2], [0.8, 0.2]]),
    )
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(2))

    output.logits.sum().backward()
    assert model.pool.projections[0][0].weight.grad is not None
    assert model.local_expert.projector[0].weight.grad is not None
    assert model.expert_gate[-1].weight.grad is not None


def test_hbp_moe_preserves_same_seed_hbp_and_classifier_initialization():
    kwargs = {
        "backbone": "mobilenetv3_small_050",
        "num_classes": 4,
        "out_indices": (1, 3, 4),
        "projection_dim": 32,
        "pretrained": False,
    }
    torch.manual_seed(19)
    baseline = AdaptationModel(head="hbp", **kwargs)
    torch.manual_seed(19)
    candidate = AdaptationModel(head="hbp_moe", **kwargs)

    for baseline_parameter, candidate_parameter in zip(
        baseline.encoder.parameters(), candidate.encoder.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    for baseline_parameter, candidate_parameter in zip(
        baseline.pool.parameters(), candidate.pool.parameters()
    ):
        assert torch.equal(baseline_parameter, candidate_parameter)
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)
    assert torch.equal(baseline.classifier.bias, candidate.classifier.bias)


def test_hbp_moe_config_is_controlled_against_hbp():
    baseline = load_config("configs/M1_mobilenetv3_hbp_source.yaml")
    candidate = load_config("configs/E1_mobilenetv3_hbp_local_moe_source.yaml")

    assert candidate["model"]["head"] == "hbp_moe"
    for key in ("backbone", "out_indices", "projection_dim", "classifier", "dropout"):
        assert baseline["model"][key] == candidate["model"][key]
    assert baseline["data"] == candidate["data"]
    assert baseline["adaptation"] == candidate["adaptation"]


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


@pytest.mark.parametrize(
    ("config_path", "backbone", "head"),
    [
        ("configs/U0_usk_resnet18_gap_source.yaml", "resnet18", "gap"),
        ("configs/U1_usk_mobilenetv2_gap_source.yaml", "mobilenetv2_100", "gap"),
        ("configs/U2_usk_mobilenetv3_gap_source.yaml", "mobilenetv3_large_100", "gap"),
        ("configs/U3_usk_mobilenetv3_hbp_source.yaml", "mobilenetv3_large_100", "hbp"),
    ],
)
def test_usk_configs_use_controlled_four_class_protocol(
    config_path, backbone, head
):
    cfg = load_config(config_path)
    assert cfg["model"]["backbone"] == backbone
    assert cfg["model"]["head"] == head
    assert cfg["model"]["num_classes"] == 4
    assert cfg["data"]["image_size"] == 256
    assert cfg["data"]["rotation_angles"] == [0]
    assert cfg["training"]["epochs"] == 25
    assert cfg["adaptation"]["method"] == "source_only"
    assert cfg["evaluation"]["hard_groups"] == {
        "peaberry_premium": ["Peaberry", "Premium"]
    }


@pytest.mark.parametrize(
    ("config_path", "head"),
    [
        ("configs/R0_roast_mobilenetv3_gap_source.yaml", "gap"),
        ("configs/R1_roast_mobilenetv3_hbp_source.yaml", "hbp"),
    ],
)
def test_roast_configs_isolate_hbp_effect(config_path, head):
    cfg = load_config(config_path)
    assert cfg["model"]["backbone"] == "mobilenetv3_large_100"
    assert cfg["model"]["head"] == head
    assert cfg["model"]["num_classes"] == 4
    assert cfg["data"]["image_size"] == 224
    assert cfg["adaptation"]["method"] == "source_only"
    assert cfg["evaluation"]["hard_groups"] == {
        "light_medium": ["Light", "Medium"]
    }
