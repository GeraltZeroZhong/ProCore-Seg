"""Sparse autoencoder used for self-supervised pretraining."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("MinkowskiEngine is required for the autoencoder") from exc

from .sparse_unet_components import SparseDownsample, SparseUpsample, build_residual_block


class Encoder(nn.Module):
    def __init__(self, channels: List[int], dimension: int = 3):
        super().__init__()
        blocks = []
        in_channels = channels[0]
        for out_channels in channels[1:]:
            blocks.append(build_residual_block(in_channels, out_channels, dimension))
            blocks.append(SparseDownsample(out_channels, out_channels, dimension))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels: List[int], dimension: int = 3):
        super().__init__()
        blocks = []
        in_channels = channels[0]
        for out_channels in channels[1:]:
            blocks.append(SparseUpsample(in_channels, out_channels, dimension))
            blocks.append(build_residual_block(out_channels, out_channels, dimension))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        for block in self.blocks:
            x = block(x)
        return x


class SparseAutoencoder(nn.Module):
    """Simple sparse convolutional autoencoder."""

    def __init__(self, input_channels: int = 8, base_channels: int = 32, depth: int = 4):
        super().__init__()
        encoder_channels = [input_channels] + [base_channels * (2 ** i) for i in range(depth)]
        decoder_channels = list(reversed(encoder_channels[1:]))
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder([encoder_channels[-1]] + decoder_channels)
        self.head = ME.MinkowskiConvolution(
            decoder_channels[-1],
            3,
            kernel_size=1,
            stride=1,
            dimension=3,
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # type: ignore[name-defined]
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return self.head(decoded)

    def reconstruct_coordinates(self, x: ME.SparseTensor) -> torch.Tensor:  # type: ignore[name-defined]
        output = self.forward(x)
        return output.F
