from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bilinear_lmmd.core.config import load_config
from bilinear_lmmd.data.loaders import ObjectCentricCrop, _transforms


def _bean_image() -> Image.Image:
    image = Image.new("RGB", (120, 100), (244, 242, 239))
    draw = ImageDraw.Draw(image)
    draw.ellipse((38, 30, 82, 70), fill=(92, 126, 56))
    return image


def test_object_crop_extracts_rgb_bean_with_square_margin():
    cropped = ObjectCentricCrop(0.10)(_bean_image())

    assert cropped.mode == "RGB"
    assert cropped.width == cropped.height
    assert cropped.width < 80
    array = np.asarray(cropped)
    assert (array[..., 1] < 180).any()
    assert (array[0].mean(axis=0) > 200).all()


def test_object_crop_rejects_empty_foreground():
    image = Image.new("RGB", (64, 64), "white")

    with pytest.raises(ValueError, match="Mask biji kosong"):
        ObjectCentricCrop()(image)


def test_object_crop_validation_transform_keeps_expected_tensor_shape():
    transform = _transforms(
        224,
        train=False,
        rotation_angles=[0],
        object_crop=True,
        object_crop_margin=0.10,
    )

    tensor = transform(_bean_image())

    assert tensor.shape == (3, 224, 224)


def test_object_crop_configs_preserve_model_factorial():
    gap = load_config("configs/coffee17/O0_mobilenetv3_gap_object_crop_source.yaml")
    hbp = load_config("configs/coffee17/O1_mobilenetv3_hbp_object_crop_source.yaml")

    assert gap["data"]["object_crop"] is True
    assert hbp["data"]["object_crop"] is True
    assert gap["data"]["object_crop_margin"] == 0.10
    assert gap["model"]["head"] == "gap"
    assert hbp["model"]["head"] == "hbp"
