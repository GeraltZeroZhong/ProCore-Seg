"""Custom loss functions tailored for noisy protein domain labels."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class DensityAwareChamferDistance(nn.Module):
    """Chamfer distance with optional per-point density weighting."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        predicted_density: Optional[torch.Tensor] = None,
        target_density: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if predicted.numel() == 0 or target.numel() == 0:
            return predicted.new_tensor(0.0)
        diff_matrix = torch.cdist(predicted, target, p=2)
        min_pred, _ = diff_matrix.min(dim=1)
        min_target, _ = diff_matrix.min(dim=0)
        if predicted_density is not None:
            min_pred = min_pred * torch.sigmoid(predicted_density)
        if target_density is not None:
            min_target = min_target * torch.sigmoid(target_density)
        loss = torch.cat([min_pred, min_target], dim=0)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class DensityWeightedCrossEntropy(nn.Module):
    """Cross entropy with per-point weighting derived from residue density."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        density_index: int = -1,
    ) -> torch.Tensor:
        if logits.shape[0] != targets.numel():
            raise ValueError("Logits and targets shape mismatch")
        if features.shape[0] != targets.numel():
            raise ValueError("Features and targets shape mismatch")
        densities = features[:, density_index]
        weights = torch.ones_like(densities)
        positive_mask = targets.view(-1) == 1
        weights[positive_mask] = torch.sigmoid(densities[positive_mask] - self.threshold)
        ce = F.cross_entropy(logits, targets.view(-1), reduction="none")
        weighted = ce * weights
        return weighted.mean()
