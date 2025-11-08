from __future__ import annotations

"""Calibration metrics for probabilistic predictions."""

from typing import Dict, Tuple

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


def _prepare_probs(probs: np.ndarray) -> np.ndarray:
    probs = to_numpy(probs)
    if probs.ndim == 1:
        return probs.astype(np.float64, copy=False)
    if probs.ndim != 2:
        raise ValueError("probs must be 1-D or 2-D.")
    return probs.astype(np.float64, copy=False)


def _prepare_labels(y_true: np.ndarray) -> np.ndarray:
    y_true = to_numpy(y_true)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1-D.")
    if not np.issubdtype(y_true.dtype, np.integer):
        y_true = y_true.astype(np.int64, copy=False)
    return y_true


def brier_score(y_true: np.ndarray, probs: np.ndarray, positive_class: int = 1) -> float:
    """Compute the Brier score for binary or multi-class probabilities.

    For binary predictions with probabilities ``probs`` of shape ``(N,)`` the
    score reduces to ``mean((probs - y_true)^2)``. For multi-class probabilities
    ``probs`` of shape ``(N, C)`` the score averages squared errors across
    classes and examples.
    """

    y_true = _prepare_labels(y_true)
    probs_arr = _prepare_probs(probs)
    if y_true.shape[0] != probs_arr.shape[0]:
        raise ValueError("y_true and probs must have matching first dimension.")

    if probs_arr.ndim == 1:
        target = (y_true == positive_class).astype(np.float64)
        diff = probs_arr - target
        return float(np.mean(diff * diff))

    n, c = probs_arr.shape
    if (y_true < 0).any() or (y_true >= c).any():
        raise ValueError("y_true contains class indices outside probability range.")
    target = np.zeros((n, c), dtype=np.float64)
    target[np.arange(n), y_true] = 1.0
    diff = probs_arr - target
    return float(np.mean(np.sum(diff * diff, axis=1) / c))


def ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Expected Calibration Error (ECE) with equal-width bins in ``[0, 1]``."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    y_true = _prepare_labels(y_true)
    probs_arr = _prepare_probs(probs)
    if y_true.shape[0] != probs_arr.shape[0]:
        raise ValueError("y_true and probs must align in length.")

    if probs_arr.ndim == 1:
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("Binary calibration expects labels in {0,1}.")
        confidences = np.clip(probs_arr, 0.0, 1.0)
        correctness = (y_true == 1).astype(np.float64)
    else:
        if (y_true < 0).any() or (y_true >= probs_arr.shape[1]).any():
            raise ValueError("y_true contains invalid class indices.")
        confidences = np.max(probs_arr, axis=1)
        predicted = np.argmax(probs_arr, axis=1)
        correctness = (predicted == y_true).astype(np.float64)
        confidences = np.clip(confidences, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1], right=True)
    bin_confidence = np.zeros(n_bins, dtype=np.float64)
    bin_accuracy = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.count_nonzero(mask))
        bin_counts[b] = count
        if count == 0:
            continue
        bin_confidence[b] = float(np.mean(confidences[mask]))
        bin_accuracy[b] = float(np.mean(correctness[mask]))

    total = float(np.sum(bin_counts))
    if total == 0:
        raise ValueError("No samples provided for calibration metrics.")

    ece_value = float(
        np.sum((np.abs(bin_confidence - bin_accuracy)) * (bin_counts / total))
    )
    details = {
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
    }
    return ece_value, details


def mce(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error across bins."""

    ece_value, details = ece(y_true, probs, n_bins=n_bins)
    diff = np.abs(details["bin_confidence"] - details["bin_accuracy"])
    return float(np.max(diff))


def nll(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood for binary or multi-class probabilities."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    y_true = _prepare_labels(y_true)
    probs_arr = _prepare_probs(probs)
    if y_true.shape[0] != probs_arr.shape[0]:
        raise ValueError("y_true and probs must align.")

    eps = float(eps)
    if probs_arr.ndim == 1:
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("Binary NLL expects labels in {0,1}.")
        probs_clipped = np.clip(probs_arr, eps, 1.0 - eps)
        targets = (y_true == 1).astype(np.float64)
        loss = -(targets * np.log(probs_clipped) + (1 - targets) * np.log(1 - probs_clipped))
        return float(np.mean(loss))

    if (y_true < 0).any() or (y_true >= probs_arr.shape[1]).any():
        raise ValueError("y_true contains invalid class indices.")
    probs_clipped = np.clip(probs_arr, eps, 1.0)
    probs_clipped /= probs_clipped.sum(axis=1, keepdims=True)
    chosen = probs_clipped[np.arange(y_true.shape[0]), y_true]
    loss = -np.log(chosen)
    return float(np.mean(loss))


if __name__ == "__main__":  # pragma: no cover - sanity checks
    y_true = np.array([0, 1, 1, 0])
    probs = np.array([0.1, 0.9, 0.8, 0.2])
    ece_value, details = ece(y_true, probs)
    assert ece_value < 1.0
    assert details["bin_counts"].sum() == len(y_true)
    assert np.isclose(brier_score(y_true, probs), np.mean((probs - y_true) ** 2))

    probs_mc = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_mc = np.array([0, 1])
    nll_value = nll(y_mc, probs_mc)
    assert nll_value >= 0

    print("Calibration metrics sanity checks passed.")
