from __future__ import annotations

"""Sparse autoencoder model for Stage-1 self-supervised pretraining.

The autoencoder utilises a :class:`~procore_seg.02_model_architecture.sparse_unet_components.SparseUNetEncoder`
to produce hierarchical features from sparse 3D inputs. A lightweight decoder
upsamples bottleneck features back to the input resolution without skip
connections to promote abstraction. The final head predicts bounded offsets
from voxel centres which are subsequently converted to reconstructed Cartesian
coordinates.
"""

from dataclasses import dataclass
from typing import Dict

import MinkowskiEngine as ME
import torch
from torch import nn

from .sparse_unet_components import (
    ConvBNReLU,
    SparseUNetEncoder,
    UNetSpec,
    Upsample,
    kaiming_init_module,
)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`SparseAutoencoder`.

    Parameters
    ----------
    in_channels:
        Number of input feature channels (must be positive).
    base_channels:
        Base channel count for the U-Net encoder (must be positive).
    depth:
        Number of encoder down-sampling stages. Total encoder outputs equal
        ``depth + 1`` including the bottleneck.
    quantization_size:
        Spatial quantisation size in Å. Must match the featuriser used to
        generate sparse tensors.
    head_activation:
        Activation applied to the head output before computing offsets. Only
        ``"tanh"`` (default) and ``None`` are supported.
    """

    in_channels: int = 8
    base_channels: int = 32
    depth: int = 4
    quantization_size: float = 1.0
    head_activation: str | None = "tanh"

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.depth < 1:
            raise ValueError("depth must be at least one")
        if self.quantization_size <= 0.0:
            raise ValueError("quantization_size must be positive")
        if self.head_activation not in ("tanh", None):
            raise ValueError("head_activation must be either 'tanh' or None")


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with MinkowskiEngine U-Net encoder.

    Notes
    -----
    * Inputs must be :class:`MinkowskiEngine.SparseTensor` instances sharing a
      coordinate manager across the encoder and decoder stages.
    * Deterministic execution depends on the caller configuring
      ``torch.backends.cudnn.deterministic`` and seeds appropriately.
    * Automatic mixed precision (AMP) is supported through PyTorch's autocast
      mechanism without additional handling in this module.
    """

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = UNetSpec(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            depth=cfg.depth,
        )

        self.encoder = SparseUNetEncoder(self.spec)

        channels = [self.spec.base_channels * (2 ** i) for i in range(self.spec.depth)]
        bottleneck_channels = self.spec.bottleneck_channels or self.spec.base_channels * (
            2 ** self.spec.depth
        )

        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_ch = bottleneck_channels
        for ch in reversed(channels):
            self.decoder_upsamples.append(Upsample(current_ch, ch))
            self.decoder_blocks.append(ConvBNReLU(ch, ch))
            current_ch = ch

        self.head = ME.MinkowskiConvolution(
            in_channels=current_ch,
            out_channels=3,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3,
        )

        kaiming_init_module(self)

    def forward(self, x: ME.SparseTensor) -> Dict[str, torch.Tensor | ME.SparseTensor]:
        """Forward pass returning offsets and reconstructed coordinates.

        Parameters
        ----------
        x:
            Input sparse tensor with ``cfg.in_channels`` features. The tensor
            must originate from the same coordinate manager throughout the
            network.

        Returns
        -------
        dict
            Dictionary containing ``"offsets"`` (sparse tensor of bounded
            offsets) and ``"points"`` (dense tensor of reconstructed Cartesian
            coordinates).
        """

        enc_feats = self.encoder(x)
        out = enc_feats[-1]
        for up, block in zip(self.decoder_upsamples, self.decoder_blocks):
            out = block(up(out))

        raw_offsets = self.head(out)
        features = raw_offsets.F
        if self.cfg.head_activation == "tanh":
            activated = torch.tanh(features)
        else:
            activated = features

        scale = self.cfg.quantization_size / 2.0
        delta = torch.clamp(activated * scale, min=-scale, max=scale)
        offsets = raw_offsets.replace_feature(delta)

        coords = x.C.to(delta.dtype)
        spatial = coords[:, 1:4]
        voxel_center = spatial * self.cfg.quantization_size
        points = voxel_center + delta

        return {"offsets": offsets, "points": points}

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state dictionary containing encoder parameters only."""

        return self.encoder.state_dict()


if __name__ == "__main__":
    try:
        cfg = AutoencoderConfig()
        model = SparseAutoencoder(cfg)
        batch_size = 2
        num_points = 64
        coords = torch.randint(low=0, high=5, size=(num_points, 4), dtype=torch.int32)
        coords[:, 0] = torch.randint(0, batch_size, (num_points,), dtype=torch.int32)
        feats = torch.randn(num_points, cfg.in_channels)
        x = ME.SparseTensor(feats, coordinates=coords)
        outputs = model(x)
        offsets = outputs["offsets"]
        points = outputs["points"]
        assert isinstance(offsets, ME.SparseTensor)
        assert offsets.F.shape[1] == 3
        assert points.shape == (offsets.F.shape[0], 3)
        print("Sparse autoencoder smoke test passed.")
    except ImportError:  # pragma: no cover
        print("MinkowskiEngine not available; skipping autoencoder smoke test.")

