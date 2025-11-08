from __future__ import annotations

"""Sparse U-Net component library built on MinkowskiEngine.

This module defines reusable neural network building blocks tailored for
MinkowskiEngine sparse tensors. All components are fully deterministic and do
not perform any global state mutations. They operate on inputs that share the
same :class:`MinkowskiEngine.CoordinateManager` and are intended to be composed
into encoder/decoder style architectures.

The utility function :func:`kaiming_init_module` can be used to initialise an
entire module hierarchy with Kaiming normal weights and batch-normalisation
parameters suitable for ReLU activations.
"""

from dataclasses import dataclass
from typing import List, Optional

import MinkowskiEngine as ME
import torch
from torch import nn


__all__ = [
    "ConvBNReLU",
    "ResidualBlock",
    "Downsample",
    "Upsample",
    "UNetSpec",
    "SparseUNetEncoder",
    "SparseUNetDecoder",
    "kaiming_init_module",
]


class ConvBNReLU(nn.Module):
    """Minkowski convolution followed by batch norm and ReLU.

    Parameters
    ----------
    in_ch:
        Number of input feature channels.
    out_ch:
        Number of output feature channels.
    k:
        Convolution kernel size. Padding is chosen automatically to preserve
        spatial resolution for ``stride == 1``.
    stride:
        Convolution stride. Strided convolutions produce sparser outputs and
        therefore increase the tensor stride in the coordinate manager.
    dilation:
        Convolution dilation factor.

    Notes
    -----
    The operation requires that the input and output tensors are associated
    with the same :class:`~MinkowskiEngine.CoordinateManager`. When used within
    a module, this is handled automatically as long as tensors originate from
    the same parent sparse tensor.
    """

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, dilation: int = 1
    ) -> None:
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k,
            stride=stride,
            dilation=dilation,
            bias=False,
            dimension=3,
        )
        self.bn = ME.MinkowskiBatchNorm(out_ch)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Apply convolution, batch-normalisation, and ReLU sequentially."""

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block composed of two Conv-BN-ReLU stages.

    The block preserves spatial resolution while allowing optional stride or
    channel projection on the residual branch. When ``stride`` is greater than
    one, the output tensor has increased stride within the coordinate manager.
    """

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, dilation: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=k, stride=stride, dilation=dilation)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=k,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=3,
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_ch)
        self.relu = ME.MinkowskiReLU(inplace=True)
        needs_projection = stride != 1 or in_ch != out_ch
        self.proj: Optional[nn.Module]
        if needs_projection:
            self.proj = ME.MinkowskiConvolution(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=stride,
                dilation=1,
                bias=False,
                dimension=3,
            )
        else:
            self.proj = None

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Return ``x + f(x)`` with shared coordinate manager.

        The input tensor must share the same coordinate manager as the block's
        internal convolutions, which is the case when tensors descend from the
        same computation graph. Projection is applied when the residual shape or
        stride changes.
        """

        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        out = self.relu(out)
        return out


class Downsample(nn.Module):
    """Halve spatial resolution with a stride-2 Minkowski convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=3,
        )
        self.bn = ME.MinkowskiBatchNorm(out_ch)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Return a stride-2 sparse tensor sharing the coordinate manager."""

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Upsample(nn.Module):
    """Double spatial resolution with a transpose Minkowski convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.deconv = ME.MinkowskiConvolutionTranspose(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=3,
        )
        self.bn = ME.MinkowskiBatchNorm(out_ch)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Return an upsampled sparse tensor sharing the coordinate manager."""

        out = self.deconv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


@dataclass(frozen=True)
class UNetSpec:
    """Configuration describing a sparse U-Net backbone."""

    in_channels: int = 8
    base_channels: int = 32
    depth: int = 4
    bottleneck_channels: Optional[int] = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.depth < 1:
            raise ValueError("depth must be at least 1")


class SparseUNetEncoder(nn.Module):
    """Sparse U-Net encoder producing multi-scale features.

    The encoder applies two residual blocks per resolution level followed by a
    stride-2 down-sampling convolution (except at the bottleneck stage). The
    :meth:`forward` method returns a list ``[L0, L1, ..., L_depth]`` where each
    element is a :class:`MinkowskiEngine.SparseTensor` at progressively coarser
    strides. The final entry corresponds to the bottleneck features.
    """

    def __init__(self, spec: UNetSpec) -> None:
        super().__init__()
        self.spec = spec
        channels = [spec.base_channels * (2 ** i) for i in range(spec.depth)]
        bottleneck_channels = spec.bottleneck_channels or spec.base_channels * (2 ** spec.depth)

        self.level_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_ch = spec.in_channels
        for level, ch in enumerate(channels):
            blocks = nn.ModuleList(
                [ResidualBlock(prev_ch, ch, stride=1), ResidualBlock(ch, ch, stride=1)]
            )
            self.level_blocks.append(blocks)
            prev_ch = ch
            next_ch = bottleneck_channels if level == spec.depth - 1 else channels[level + 1]
            self.downsamples.append(Downsample(prev_ch, next_ch))
            prev_ch = next_ch

        self.bottleneck_blocks = nn.ModuleList(
            [ResidualBlock(prev_ch, bottleneck_channels), ResidualBlock(bottleneck_channels, bottleneck_channels)]
        )

    def forward(self, x: ME.SparseTensor) -> List[ME.SparseTensor]:
        """Return encoder activations for skip connections.

        Parameters
        ----------
        x:
            Input sparse tensor with ``spec.in_channels`` features. The tensor
            must share a coordinate manager with all subsequent tensors; this is
            naturally satisfied when tensors originate from this encoder.

        Returns
        -------
        list of :class:`MinkowskiEngine.SparseTensor`
            Feature maps ``[L0, ..., L_depth]`` ordered from finest to coarsest
            resolution. Each level increases the tensor stride due to the
            down-sampling operations.
        """

        features: List[ME.SparseTensor] = []
        out = x
        for level, blocks in enumerate(self.level_blocks):
            for block in blocks:
                out = block(out)
            features.append(out)
            out = self.downsamples[level](out)

        for block in self.bottleneck_blocks:
            out = block(out)
        features.append(out)
        return features


class SparseUNetDecoder(nn.Module):
    """Sparse U-Net decoder consuming encoder features.

    The decoder performs transpose convolutions to progressively increase
    spatial resolution. After each upsampling stage, the upsampled features are
    concatenated with the corresponding encoder activation using
    :func:`MinkowskiEngine.cat`, followed by two residual blocks that refine the
    fused features. The output tensor matches the L0 resolution of the encoder.
    """

    def __init__(self, spec: UNetSpec, out_channels: int) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        self.spec = spec
        self.out_channels = out_channels

        channels = [spec.base_channels * (2 ** i) for i in range(spec.depth)]
        bottleneck_channels = spec.bottleneck_channels or spec.base_channels * (2 ** spec.depth)

        self.upsamples = nn.ModuleList()
        self.level_blocks = nn.ModuleList()

        current_ch = bottleneck_channels
        for idx, ch in enumerate(reversed(channels)):
            self.upsamples.append(Upsample(current_ch, ch))
            is_final = idx == len(channels) - 1
            out_ch = out_channels if is_final else ch
            block1 = ResidualBlock(ch + ch, ch)
            block2 = ResidualBlock(ch, out_ch)
            self.level_blocks.append(nn.ModuleList([block1, block2]))
            current_ch = out_ch

    def forward(self, enc_feats: List[ME.SparseTensor]) -> ME.SparseTensor:
        """Decode encoder features back to the input resolution.

        Parameters
        ----------
        enc_feats:
            Encoder feature list generated by :class:`SparseUNetEncoder`. The
            list must contain ``spec.depth + 1`` tensors ordered from finest to
            coarsest resolution.

        Returns
        -------
        :class:`MinkowskiEngine.SparseTensor`
            Output tensor at L0 resolution containing ``out_channels`` features.
        """

        if len(enc_feats) != self.spec.depth + 1:
            raise ValueError(
                f"Expected {self.spec.depth + 1} encoder features, got {len(enc_feats)}"
            )

        out = enc_feats[-1]
        encoder_levels = enc_feats[:-1]
        for idx, (upsample, blocks) in enumerate(zip(self.upsamples, self.level_blocks)):
            out = upsample(out)
            skip = encoder_levels[-(idx + 1)]
            out = ME.cat([out, skip])
            for block in blocks:
                out = block(out)
        return out


def kaiming_init_module(module: nn.Module) -> None:
    """Apply Kaiming initialisation to all convolutional layers in ``module``.

    Convolution weights are initialised using ``kaiming_normal_`` with ReLU
    non-linearity assumptions. Batch-normalisation gains are set to one and
    biases to zero. Projection layers inside residual blocks are initialised in
    the same manner. The operation is deterministic and does not alter global
    random seeds.
    """

    for submodule in module.modules():
        if isinstance(submodule, (ME.MinkowskiConvolution, ME.MinkowskiConvolutionTranspose)):
            nn.init.kaiming_normal_(submodule.kernel, mode="fan_out", nonlinearity="relu")
            if getattr(submodule, "bias", None) is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, ME.MinkowskiBatchNorm):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)


def _self_check_channels() -> bool:
    """Internal consistency check used by unit tests.

    The function constructs an encoder/decoder pair and verifies that channel
    dimensions propagate as expected. It is intentionally lightweight and is
    not executed on module import.
    """

    spec = UNetSpec(in_channels=4, base_channels=8, depth=2)
    encoder = SparseUNetEncoder(spec)
    decoder = SparseUNetDecoder(spec, out_channels=spec.base_channels)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    feats = torch.zeros((1, spec.in_channels), dtype=torch.float32)
    x = ME.SparseTensor(feats, coordinates=coords)
    feats_list = encoder(x)
    assert len(feats_list) == spec.depth + 1
    out = decoder(feats_list)
    assert out.F.shape[1] == spec.base_channels
    return True


if __name__ == "__main__":
    try:
        spec = UNetSpec()
        encoder = SparseUNetEncoder(spec)
        decoder = SparseUNetDecoder(spec, out_channels=spec.base_channels)
        kaiming_init_module(encoder)
        kaiming_init_module(decoder)

        batch_size = 2
        num_points = 64
        coords = torch.randint(low=0, high=5, size=(num_points, 4), dtype=torch.int32)
        coords[:, 0] = torch.randint(0, batch_size, (num_points,), dtype=torch.int32)
        feats = torch.randn(num_points, spec.in_channels)
        x = ME.SparseTensor(feats, coordinates=coords)

        encoder_feats = encoder(x)
        assert len(encoder_feats) == spec.depth + 1
        decoder_out = decoder(encoder_feats)
        assert decoder_out.F.shape[1] == spec.base_channels
        print("Sparse U-Net components smoke test passed.")
    except ImportError:  # pragma: no cover - ME may be unavailable in some envs.
        print("MinkowskiEngine not available; skipping component smoke test.")

