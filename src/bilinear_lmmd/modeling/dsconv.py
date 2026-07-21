from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DistributionShiftConv2d(nn.Module):
    """Quantization-aware Distribution Shifting Convolution.

    The operator follows Nascimento et al. (ICCV 2019) and the formulation
    reused by Hong et al. (2026): a low-bit Variable Quantized Kernel (VQK)
    is reconstructed with learnable Kernel Distribution Shift (KDS) and
    Channel Distribution Shift (CDS) parameters.

    This module simulates the reconstructed kernel in ordinary PyTorch
    ``conv2d`` so it remains trainable and portable. Consequently, parameter
    accuracy can be studied here, but integer-kernel latency must not be
    claimed without a dedicated deployment backend.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        bias: bool = False,
        bits: int = 4,
        block_size: int = 128,
        dynamic_same_padding: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel DSConv harus lebih besar dari nol.")
        if bits < 2 or bits > 8:
            raise ValueError("dsconv_bits harus berada pada rentang 2..8.")
        if block_size <= 0:
            raise ValueError("dsconv_block_size harus lebih besar dari nol.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(int(value) for value in kernel_size)
        self.stride = tuple(int(value) for value in stride)
        self.padding = tuple(int(value) for value in padding)
        self.dilation = tuple(int(value) for value in dilation)
        self.bits = int(bits)
        self.block_size = int(block_size)
        self.dynamic_same_padding = bool(dynamic_same_padding)
        self.num_blocks = math.ceil(self.in_channels / self.block_size)
        self.quant_min = -(2 ** (self.bits - 1))
        self.quant_max = 2 ** (self.bits - 1) - 1

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        self.bias = (
            nn.Parameter(torch.empty(self.out_channels)) if bias else None
        )
        shift_shape = (
            self.out_channels,
            self.num_blocks,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.kds_scale = nn.Parameter(torch.ones(shift_shape))
        self.kds_bias = nn.Parameter(torch.zeros(shift_shape))
        self.cds_scale = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.cds_bias = nn.Parameter(torch.zeros(self.out_channels, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.initialize_shifts_from_weight()

    @torch.no_grad()
    def initialize_shifts_from_weight(self) -> None:
        """Initialize KDS by least squares against the current FP kernel."""
        for block_index, (start, stop) in enumerate(self.block_ranges()):
            block = self.weight[:, start:stop]
            maximum = block.abs().amax(dim=1, keepdim=True)
            step = (maximum / float(self.quant_max)).clamp_min(1.0e-8)
            quantized = torch.round(block / step).clamp(
                self.quant_min, self.quant_max
            )
            numerator = (block * quantized).sum(dim=1)
            denominator = quantized.square().sum(dim=1).clamp_min(1.0e-8)
            self.kds_scale[:, block_index].copy_(numerator / denominator)
        self.kds_bias.zero_()
        self.cds_scale.fill_(1.0)
        self.cds_bias.zero_()

    @classmethod
    def from_conv2d(
        cls,
        convolution: nn.Conv2d,
        *,
        bits: int = 4,
        block_size: int = 128,
    ) -> "DistributionShiftConv2d":
        if convolution.groups != 1:
            raise ValueError("DSConv adaptation hanya mengganti convolution groups=1.")
        module = cls(
            convolution.in_channels,
            convolution.out_channels,
            tuple(convolution.kernel_size),
            stride=tuple(convolution.stride),
            padding=tuple(convolution.padding),
            dilation=tuple(convolution.dilation),
            bias=convolution.bias is not None,
            bits=bits,
            block_size=block_size,
            dynamic_same_padding=(
                convolution.__class__.__name__ == "Conv2dSame"
            ),
        )
        module = module.to(
            device=convolution.weight.device,
            dtype=convolution.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(convolution.weight)
            if module.bias is not None and convolution.bias is not None:
                module.bias.copy_(convolution.bias)
            module.initialize_shifts_from_weight()
        return module

    def block_ranges(self) -> list[tuple[int, int]]:
        return [
            (
                block_index * self.block_size,
                min((block_index + 1) * self.block_size, self.in_channels),
            )
            for block_index in range(self.num_blocks)
        ]

    def _quantized_block(self, block: Tensor) -> Tensor:
        maximum = block.detach().abs().amax(dim=1, keepdim=True)
        step = (maximum / float(self.quant_max)).clamp_min(1.0e-8)
        scaled = block / step
        rounded = torch.round(scaled).clamp(self.quant_min, self.quant_max)
        # Straight-through estimator: integer values in forward, gradients to
        # the latent full-precision kernel in backward.
        return scaled + (rounded - scaled).detach()

    def reconstructed_weight(self) -> Tensor:
        reconstructed = []
        for block_index, (start, stop) in enumerate(self.block_ranges()):
            quantized = self._quantized_block(self.weight[:, start:stop])
            scale = self.kds_scale[:, block_index].unsqueeze(1)
            bias = self.kds_bias[:, block_index].unsqueeze(1)
            reconstructed.append(scale * quantized + bias)
        kernel = torch.cat(reconstructed, dim=1)
        return self.cds_scale * kernel + self.cds_bias

    def forward(self, inputs: Tensor) -> Tensor:
        padding = self.padding
        if self.dynamic_same_padding:
            input_height, input_width = inputs.shape[-2:]
            effective_height = (
                (self.kernel_size[0] - 1) * self.dilation[0] + 1
            )
            effective_width = (
                (self.kernel_size[1] - 1) * self.dilation[1] + 1
            )
            output_height = math.ceil(input_height / self.stride[0])
            output_width = math.ceil(input_width / self.stride[1])
            pad_height = max(
                (output_height - 1) * self.stride[0]
                + effective_height
                - input_height,
                0,
            )
            pad_width = max(
                (output_width - 1) * self.stride[1]
                + effective_width
                - input_width,
                0,
            )
            if pad_height or pad_width:
                inputs = F.pad(
                    inputs,
                    (
                        pad_width // 2,
                        pad_width - pad_width // 2,
                        pad_height // 2,
                        pad_height - pad_height // 2,
                    ),
                )
            padding = (0, 0)
        return F.conv2d(
            inputs,
            self.reconstructed_weight(),
            self.bias,
            self.stride,
            padding,
            self.dilation,
            groups=1,
        )

    def theoretical_kernel_bits(self) -> int:
        """Storage bits for VQK plus FP32 KDS/CDS deployment tensors."""
        vqk = self.weight.numel() * self.bits
        shifts = (
            self.kds_scale.numel()
            + self.kds_bias.numel()
            + self.cds_scale.numel()
            + self.cds_bias.numel()
        ) * 32
        if self.bias is not None:
            shifts += self.bias.numel() * 32
        return int(vqk + shifts)


def replace_spatial_convolutions_with_dsconv(
    root: nn.Module,
    *,
    stage_prefixes: Sequence[str],
    bits: int = 4,
    block_size: int = 128,
) -> list[str]:
    """Replace full spatial convolutions in selected backbone stages."""
    prefixes = tuple(str(prefix).rstrip(".") for prefix in stage_prefixes)
    if not prefixes:
        raise ValueError("Minimal satu dsconv_stage_prefix harus diberikan.")
    replacements: list[tuple[str, nn.Conv2d]] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1 or tuple(module.kernel_size) == (1, 1):
            continue
        if not any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        replacements.append((name, module))

    replaced_names = []
    rng_state = torch.random.get_rng_state()
    try:
        for name, convolution in replacements:
            parent_name, _, child_name = name.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            parent._modules[child_name] = DistributionShiftConv2d.from_conv2d(
                convolution,
                bits=bits,
                block_size=block_size,
            )
            replaced_names.append(name)
    finally:
        # Replacing a pretrained layer must not alter the paired classifier's
        # same-seed initialization in the subsequent model constructor.
        torch.random.set_rng_state(rng_state)
    if not replaced_names:
        raise ValueError(
            "Tidak ada spatial convolution groups=1 yang cocok dengan "
            f"prefix DSConv {list(prefixes)}."
        )
    return replaced_names
