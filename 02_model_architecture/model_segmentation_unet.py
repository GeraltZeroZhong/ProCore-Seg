from __future__ import annotations

"""Sparse U-Net architecture for Stage-2 semantic segmentation."""

from dataclasses import dataclass

import MinkowskiEngine as ME
import torch
from torch import nn

from .sparse_unet_components import (
    SparseUNetDecoder,
    SparseUNetEncoder,
    UNetSpec,
    kaiming_init_module,
)


@dataclass
class SegmentationConfig:
    """Configuration for :class:`SparseSegmentationUNet`.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    base_channels:
        Base number of channels in the encoder.
    depth:
        Number of encoder down-sampling levels (``depth + 1`` feature maps).
    num_classes:
        Number of output semantic classes (must be at least two).
    """

    in_channels: int = 8
    base_channels: int = 32
    depth: int = 4
    num_classes: int = 2

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.depth < 1:
            raise ValueError("depth must be at least one")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")


class SparseSegmentationUNet(nn.Module):
    """Sparse semantic segmentation model with skip connections.

    The model mirrors the encoder topology used during Stage-1 pretraining. The
    decoder fuses encoder skip connections to recover fine-grained spatial
    detail. Callers are responsible for configuring deterministic execution via
    PyTorch global flags when required.
    """

    def __init__(self, cfg: SegmentationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = UNetSpec(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            depth=cfg.depth,
        )

        self.encoder = SparseUNetEncoder(self.spec)
        self.decoder = SparseUNetDecoder(self.spec, out_channels=self.spec.base_channels)
        self.head = ME.MinkowskiConvolution(
            in_channels=self.spec.base_channels,
            out_channels=cfg.num_classes,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=3,
        )

        kaiming_init_module(self)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """Return logits at the input (L0) stride with ``cfg.num_classes`` channels."""

        enc_feats = self.encoder(x)
        decoded = self.decoder(enc_feats)
        logits = self.head(decoded)
        return logits

    def load_pretrained_encoder(self, state_dict: dict, strict: bool = True) -> None:
        """Load encoder weights from a state dictionary.

        ``state_dict`` may either contain raw encoder parameters or be prefixed
        with ``"encoder."`` as produced by :func:`torch.nn.Module.state_dict` on
        the full segmentation model.
        """

        encoder_state = {}
        encoder_keys = set(self.encoder.state_dict().keys())
        for key, value in state_dict.items():
            trimmed = key
            if key.startswith("encoder."):
                trimmed = key[len("encoder.") :]
            if trimmed in encoder_keys:
                encoder_state[trimmed] = value

        self.encoder.load_state_dict(encoder_state, strict=strict)

    @staticmethod
    def smoke_test() -> bool:
        """Lightweight forward pass test used by unit tests."""

        cfg = SegmentationConfig()
        model = SparseSegmentationUNet(cfg)
        batch_size = 2
        num_points = 64
        coords = torch.randint(low=0, high=5, size=(num_points, 4), dtype=torch.int32)
        coords[:, 0] = torch.randint(0, batch_size, (num_points,), dtype=torch.int32)
        feats = torch.randn(num_points, cfg.in_channels)
        x = ME.SparseTensor(feats, coordinates=coords)
        logits = model(x)
        assert logits.F.shape[1] == cfg.num_classes
        assert logits.coordinate_map_key == x.coordinate_map_key
        return True


if __name__ == "__main__":
    try:
        SparseSegmentationUNet.smoke_test()
        print("Sparse segmentation U-Net smoke test passed.")
    except ImportError:  # pragma: no cover
        print("MinkowskiEngine not available; skipping segmentation smoke test.")

