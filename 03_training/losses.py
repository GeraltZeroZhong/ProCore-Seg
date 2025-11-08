from __future__ import annotations

"""Loss functions used during ProCore-Seg training."""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class DensityAwareChamferDistance(nn.Module):
    """Weighted symmetric Chamfer distance supporting optional chunking."""

    def __init__(
        self,
        pred_weight: float = 1.0,
        gt_weight: float = 1.0,
        eps: float = 1e-9,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if pred_weight <= 0.0:
            raise ValueError("pred_weight must be positive")
        if gt_weight <= 0.0:
            raise ValueError("gt_weight must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided")

        self.pred_weight = float(pred_weight)
        self.gt_weight = float(gt_weight)
        self.eps = float(eps)
        self.chunk_size = chunk_size

    def _chunked_min_dist(
        self, src: torch.Tensor, dst: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        num_src = src.size(0)
        min_dist = torch.empty(num_src, device=src.device, dtype=src.dtype)
        for start in range(0, num_src, chunk_size):
            end = min(start + chunk_size, num_src)
            chunk = src[start:end]
            dists = torch.cdist(chunk, dst, p=2)
            min_vals, _ = torch.min(dists, dim=1)
            min_dist[start:end] = min_vals.square()
        return min_dist

    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        w_pred: Optional[torch.Tensor] = None,
        w_gt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred_points.ndim != 2 or pred_points.size(1) != 3:
            raise ValueError("pred_points must have shape (P, 3)")
        if gt_points.ndim != 2 or gt_points.size(1) != 3:
            raise ValueError("gt_points must have shape (G, 3)")
        if pred_points.numel() == 0 or gt_points.numel() == 0:
            raise ValueError("pred_points and gt_points must be non-empty")

        if w_pred is None:
            w_pred = torch.ones(pred_points.size(0), device=pred_points.device, dtype=pred_points.dtype)
        if w_gt is None:
            w_gt = torch.ones(gt_points.size(0), device=gt_points.device, dtype=gt_points.dtype)

        w_pred = w_pred.reshape(-1).to(device=pred_points.device, dtype=pred_points.dtype)
        w_gt = w_gt.reshape(-1).to(device=gt_points.device, dtype=gt_points.dtype)

        chunk_size = self.chunk_size
        if chunk_size is None:
            d_pred = torch.cdist(pred_points, gt_points, p=2).square()
            min_pred, _ = torch.min(d_pred, dim=1)
            min_gt, _ = torch.min(d_pred, dim=0)
        else:
            min_pred = self._chunked_min_dist(pred_points, gt_points, chunk_size)
            min_gt = self._chunked_min_dist(gt_points, pred_points, chunk_size)

        weighted_pred = (w_pred * min_pred).sum() / w_pred.sum().clamp_min(self.eps)
        weighted_gt = (w_gt * min_gt).sum() / w_gt.sum().clamp_min(self.eps)
        return self.pred_weight * weighted_pred + self.gt_weight * weighted_gt


class DensityWeightedCrossEntropy(nn.Module):
    """Cross entropy loss modulated by atomic density priors."""

    def __init__(
        self,
        density_feature_idx: int = 7,
        T: float = 0.0,
        tau: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.density_feature_idx = density_feature_idx
        self.T = float(T)
        self.tau = float(tau)
        self.reduction = reduction
        self.register_buffer("_class_weights", class_weights if class_weights is not None else None)

    def forward(
        self,
        logits_per_voxel: torch.Tensor,
        atom2voxel: torch.LongTensor,
        targets_atom: torch.LongTensor,
        features_atom: torch.Tensor,
    ) -> torch.Tensor:
        if logits_per_voxel.ndim != 2:
            raise ValueError("logits_per_voxel must have shape (V, C)")
        if features_atom.ndim != 2:
            raise ValueError("features_atom must have shape (N, D)")
        if not (0 <= self.density_feature_idx < features_atom.size(1)):
            raise ValueError("density_feature_idx out of bounds for features_atom")
        if atom2voxel.shape[0] != targets_atom.shape[0]:
            raise ValueError("atom2voxel and targets_atom must be the same length")
        if atom2voxel.shape[0] != features_atom.shape[0]:
            raise ValueError("atom2voxel and features_atom must be the same length")

        logits_atom = logits_per_voxel.index_select(0, atom2voxel)
        weight = self._class_weights if self._class_weights is not None else None
        ce = F.cross_entropy(logits_atom, targets_atom, weight=weight, reduction="none")

        density = features_atom[:, self.density_feature_idx]
        scaled = (density - self.T) / self.tau
        w_pos = torch.sigmoid(scaled)
        weights = torch.where(targets_atom == 1, w_pos, torch.ones_like(w_pos))

        weighted = weights * ce
        if self.reduction == "mean":
            return weighted.mean()
        if self.reduction == "sum":
            return weighted.sum()
        return weighted
