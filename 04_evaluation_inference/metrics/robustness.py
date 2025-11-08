from __future__ import annotations

"""Robustness metrics assessing prediction consistency."""

from typing import Dict, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    if torch is not None and isinstance(x, torch.Tensor):  # pragma: no cover
        return np.ascontiguousarray(x.detach().cpu().numpy())
    return np.ascontiguousarray(np.array(x))


def _validate_binary_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = to_numpy(a)
    b = to_numpy(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Predictions must be 1-D arrays.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Predictions must have identical lengths.")
    if not np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.int64, copy=False)
    if not np.issubdtype(b.dtype, np.integer):
        b = b.astype(np.int64, copy=False)
    if not np.array_equal(a, a.astype(bool)):
        raise ValueError("Predictions must be binary {0,1}.")
    if not np.array_equal(b, b.astype(bool)):
        raise ValueError("Predictions must be binary {0,1}.")
    return a, b


def consistency_iou(y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """IoU between two binary prediction vectors of shape ``(N,)``."""

    a, b = _validate_binary_pair(y_pred_a, y_pred_b)
    intersection = float(np.dot(a, b))
    union = float(a.sum() + b.sum() - intersection)
    if union == 0.0:
        return 1.0
    return intersection / union


def agreement_rate(y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """Fraction of identical labels between two binary prediction vectors."""

    a, b = _validate_binary_pair(y_pred_a, y_pred_b)
    if a.size == 0:
        raise ValueError("Predictions must not be empty.")
    return float(np.mean(a == b))


def batch_consistency_report(preds_list: Sequence[np.ndarray]) -> Dict[str, float]:
    """Compute pairwise consistency statistics across perturbations.

    Parameters
    ----------
    preds_list : Sequence[np.ndarray]
        Sequence of ``K`` binary prediction vectors, each of shape ``(N,)``.

    Returns
    -------
    Dict[str, float]
        Dictionary containing pairwise mean/min IoU and agreement values.
    """

    if not preds_list:
        raise ValueError("preds_list must contain at least one prediction array.")
    preds: list[np.ndarray] = []
    for idx, p_raw in enumerate(preds_list):
        p = to_numpy(p_raw)
        if p.ndim != 1:
            raise ValueError("Each prediction array must be 1-D.")
        if idx == 0:
            length = p.shape[0]
        elif p.shape[0] != length:
            raise ValueError("All predictions must have the same length.")
        if np.issubdtype(p.dtype, np.integer):
            if not np.all((p == 0) | (p == 1)):
                raise ValueError("Predictions must be binary {0,1}.")
            p_int = p.astype(np.int64, copy=False)
        else:
            p_float = p.astype(np.float64, copy=False)
            if not np.all(np.isin(p_float, (0.0, 1.0))):
                raise ValueError("Predictions must be binary {0,1}.")
            p_int = p_float.astype(np.int64, copy=False)
        preds.append(p_int)

    k = len(preds)
    if k == 1:
        return {
            "pairwise_mean_iou": 1.0,
            "pairwise_mean_agreement": 1.0,
            "pairwise_min_iou": 1.0,
            "pairwise_min_agreement": 1.0,
        }

    pairwise_ious = []
    pairwise_agree = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = preds[i], preds[j]
            pairwise_ious.append(consistency_iou(a, b))
            pairwise_agree.append(agreement_rate(a, b))

    pairwise_ious_arr = np.asarray(pairwise_ious, dtype=np.float64)
    pairwise_agree_arr = np.asarray(pairwise_agree, dtype=np.float64)

    return {
        "pairwise_mean_iou": float(np.mean(pairwise_ious_arr)),
        "pairwise_mean_agreement": float(np.mean(pairwise_agree_arr)),
        "pairwise_min_iou": float(np.min(pairwise_ious_arr)),
        "pairwise_min_agreement": float(np.min(pairwise_agree_arr)),
    }


if __name__ == "__main__":  # pragma: no cover - sanity checks
    preds = [
        np.array([0, 1, 1, 0, 1]),
        np.array([0, 1, 0, 0, 1]),
        np.array([1, 1, 0, 0, 1]),
    ]
    report = batch_consistency_report(preds)
    assert report["pairwise_min_iou"] <= report["pairwise_mean_iou"]
    assert report["pairwise_min_agreement"] <= report["pairwise_mean_agreement"]
    print("Robustness metrics sanity checks passed.")
