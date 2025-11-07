"""Sparse U-Net segmentation model with skip connections."""

from __future__ import annotations

from typing import List

import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("MinkowskiEngine is required for the segmentation U-Net") from exc

from .sparse_unet_components import SparseDownsample, SparseUpsample, build_residual_block


class SegmentationEncoder(nn.Module):
    def __init__(self, channels: List[int], dimension: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = channels[0]
        for out_channels in channels[1:]:
            self.blocks.append(build_residual_block(in_channels, out_channels, dimension))
            self.downsamples.append(SparseDownsample(out_channels, out_channels, dimension))
            in_channels = out_channels

    def forward(self, x: ME.SparseTensor) -> List[ME.SparseTensor]:  # type: ignore[name-defined]
        skips: List[ME.SparseTensor] = []
        for block, down in zip(self.blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = down(x)
        skips.append(x)
        return skips


class SegmentationDecoder(nn.Module):
    def __init__(self, channels: List[int], dimension: int = 3):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        in_channels = channels[0]
        for out_channels in channels[1:]:
            self.upsamples.append(SparseUpsample(in_channels, out_channels, dimension))
            self.blocks.append(build_residual_block(out_channels * 2, out_channels, dimension))
            in_channels = out_channels

    def forward(
        self, x: ME.SparseTensor, skips: List[ME.SparseTensor]
    ) -> ME.SparseTensor:  # type: ignore[name-defined]
        for upsample, block, skip in zip(self.upsamples, self.blocks, reversed(skips)):
            x = upsample(x)
            x = ME.cat(x, skip)
            x = block(x)
        return x


class SparseSegmentationUNet(nn.Module):
    """Segmentation network operating on sparse tensors."""

    def __init__(self, input_channels: int = 8, base_channels: int = 32, depth: int = 4, num_classes: int = 2):
        super().__init__()
        encoder_channels = [input_channels] + [base_channels * (2 ** i) for i in range(depth)]
        decoder_channels = list(reversed(encoder_channels[1:]))
        self.encoder = SegmentationEncoder(encoder_channels)
        self.decoder = SegmentationDecoder([encoder_channels[-1]] + decoder_channels)
        self.head = ME.MinkowskiConvolution(
            decoder_channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            dimension=3,
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        skips = self.encoder(x)
        bottleneck = skips.pop()
        decoded = self.decoder(bottleneck, skips)
        return self.head(decoded)

    def load_encoder_weights(self, state_dict):
        self.encoder.load_state_dict(state_dict, strict=False)
