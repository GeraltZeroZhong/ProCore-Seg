"""Reusable sparse convolution building blocks built on MinkowskiEngine."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "MinkowskiEngine is required for the sparse U-Net components"
    ) from exc


@dataclass(slots=True)
class BlockConfig:
    in_channels: int
    out_channels: int
    dimension: int = 3


class SparseConvBlock(nn.Module):
    """Convolution -> BatchNorm -> ReLU convenience block."""

    def __init__(self, in_channels: int, out_channels: int, dimension: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        return self.net(x)


class SparseResidualBlock(nn.Module):
    """Two convolutional layers with a residual connection."""

    def __init__(self, config: BlockConfig):
        super().__init__()
        self.conv1 = SparseConvBlock(config.in_channels, config.out_channels, config.dimension)
        self.conv2 = SparseConvBlock(config.out_channels, config.out_channels, config.dimension)
        if config.in_channels != config.out_channels:
            self.skip = ME.MinkowskiConvolution(
                config.in_channels,
                config.out_channels,
                kernel_size=1,
                stride=1,
                dimension=config.dimension,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity


class SparseDownsample(nn.Module):
    """Strided sparse convolution used for downsampling."""

    def __init__(self, in_channels: int, out_channels: int, dimension: int = 3):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            dimension=dimension,
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        return self.conv(x)


class SparseUpsample(nn.Module):
    """Transpose convolution block used for upsampling."""

    def __init__(self, in_channels: int, out_channels: int, dimension: int = 3):
        super().__init__()
        self.deconv = ME.MinkowskiConvolutionTranspose(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            dimension=dimension,
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        return self.deconv(x)


def build_residual_block(in_channels: int, out_channels: int, dimension: int = 3) -> nn.Module:
    return SparseResidualBlock(BlockConfig(in_channels, out_channels, dimension))
