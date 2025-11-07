"""Evaluation metrics for protein domain segmentation."""

from __future__ import annotations

import numpy as np


def core_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the intersection-over-union for the core domain class."""

    pred_binary = pred.astype(bool)
    target_binary = target.astype(bool)
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred_binary = pred.astype(bool)
    target_binary = target.astype(bool)
    tp = np.logical_and(pred_binary, target_binary).sum()
    fp = np.logical_and(pred_binary, np.logical_not(target_binary)).sum()
    fn = np.logical_and(np.logical_not(pred_binary), target_binary).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(precision), float(recall)
