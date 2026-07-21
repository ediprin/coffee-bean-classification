import torch

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.modeling.models import (
    MatrixPowerNormalizedCovariancePooling,
    build_model,
)


def test_mpncov_shape_finite_gradient_and_translation_invariance() -> None:
    head = MatrixPowerNormalizedCovariancePooling(
        channels=3,
        reduction_dim=3,
        iterations=7,
        epsilon=1.0e-6,
    )
    head.eval()
    with torch.no_grad():
        head.reduction[0].weight.zero_()
        for index in range(3):
            head.reduction[0].weight[index, index, 0, 0] = 1.0

    feature = (torch.rand(2, 3, 4, 5) + 1.0).requires_grad_()
    shifted = feature.detach() + torch.tensor([2.0, 3.0, 4.0]).view(1, 3, 1, 1)
    output = head([feature])
    shifted_output = head([shifted])

    assert output.shape == (2, 6)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.detach(), shifted_output, atol=2.0e-5)
    output.sum().backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert head.reduction[0].weight.grad is not None


def test_newton_schulz_matches_symmetric_eigendecomposition() -> None:
    head = MatrixPowerNormalizedCovariancePooling(
        channels=3,
        reduction_dim=3,
        iterations=10,
        epsilon=1.0e-8,
    )
    matrix = torch.tensor(
        [[[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]]
    )
    result = head._matrix_square_root(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    reference = eigenvectors.bmm(torch.diag_embed(eigenvalues.sqrt())).bmm(
        eigenvectors.transpose(1, 2)
    )
    assert torch.allclose(result, reference, atol=2.0e-4, rtol=2.0e-4)


def test_covariance_configs_are_controlled_and_buildable() -> None:
    paths = {
        "COV0": "configs/covariance/COV0_efficientnetv2_gap_source.yaml",
        "COV1": "configs/covariance/COV1_efficientnetv2_hbp_source.yaml",
        "COV2": "configs/covariance/COV2_efficientnetv2_mpncov_source.yaml",
    }
    configs = {code: load_config(path) for code, path in paths.items()}
    assert {cfg["model"]["backbone"] for cfg in configs.values()} == {
        "tf_efficientnetv2_b0.in1k"
    }
    assert {cfg["data"]["image_size"] for cfg in configs.values()} == {224}
    assert {cfg["adaptation"]["method"] for cfg in configs.values()} == {
        "source_only"
    }
    assert {cfg["training"]["classification_loss"] for cfg in configs.values()} == {
        "cross_entropy"
    }
    assert configs["COV2"]["model"]["mpncov_reduction_dim"] == 128
    assert configs["COV2"]["model"]["mpncov_iterations"] == 5

    model_cfg = dict(configs["COV2"]["model"])
    model_cfg["pretrained"] = False
    model = build_model(model_cfg)
    assert isinstance(model.pool, MatrixPowerNormalizedCovariancePooling)
    assert model.pool.output_dim == 128 * 129 // 2
