from __future__ import annotations

"""Classification metrics for ProCore-Seg evaluation.

The functions in this module operate on NumPy arrays and will transparently
convert PyTorch tensors to NumPy without tracking gradients. All metrics are
implemented with deterministic, numerically stable operations and do not rely
on external machine-learning libraries.
"""

from typing import Any, Dict, Tuple

import numpy as np

try:  # Optional dependency for tensor inputs
    import torch
except Exception:  # pragma: no cover - torch may be absent in environment
    torch = None  # type: ignore


def to_numpy(x: Any) -> np.ndarray:
    """Convert list-like or tensor inputs to a contiguous :class:`np.ndarray`.

    Parameters
    ----------
    x:
        Input array-like object (NumPy array, PyTorch tensor, list, tuple).

    Returns
    -------
    np.ndarray
        Contiguous NumPy array view/copy of the input with ``dtype`` preserved
        when possible.

    Examples
    --------
    >>> import numpy as _np
    >>> to_numpy([0, 1, 2]).dtype == _np.int64
    True
    """

    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    if torch is not None and isinstance(x, torch.Tensor):  # pragma: no cover - optional
        return np.ascontiguousarray(x.detach().cpu().numpy())
    return np.ascontiguousarray(np.array(x))


def _validate_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if not np.issubdtype(y_true.dtype, np.integer):
        y_true = y_true.astype(np.int64, copy=False)
    if not np.issubdtype(y_pred.dtype, np.integer):
        y_pred = y_pred.astype(np.int64, copy=False)
    return y_true, y_pred


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute the confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of shape ``(N,)`` containing integer class indices in
        ``[0, num_classes - 1]``.
    num_classes : int
        Number of classes ``C``.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(C, C)`` with dtype ``int64`` where element
        ``[i, j]`` counts examples with true class ``i`` and predicted class
        ``j``.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    y_true, y_pred = _validate_labels(y_true, y_pred)
    if (y_true < 0).any() or (y_true >= num_classes).any():
        raise ValueError("y_true contains class indices outside valid range.")
    if (y_pred < 0).any() or (y_pred >= num_classes).any():
        raise ValueError("y_pred contains class indices outside valid range.")

    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(mat, (y_true, y_pred), 1)
    return mat


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall accuracy.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of shape ``(N,)`` with integer labels.

    Returns
    -------
    float
        Fraction of matching labels in ``[0, 1]`` as ``float64``.
    """

    y_true, y_pred = _validate_labels(y_true, y_pred)
    if y_true.size == 0:
        raise ValueError("y_true must not be empty.")
    return float(np.mean(y_true == y_pred))


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom = np.asarray(denom, dtype=np.float64)
    numer = np.asarray(numer, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom != 0)
    return out


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    average: str = "binary",
) -> Dict[str, float]:
    """Compute precision, recall and F1-score with multiple averaging modes.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of shape ``(N,)`` containing integer labels.
    num_classes : int
        Number of classes present.
    average : {"binary", "macro", "weighted", "micro"}, default="binary"
        Averaging strategy. Binary averaging treats class ``1`` as the positive
        class and requires ``num_classes == 2``.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``precision``, ``recall`` and ``f1``.
    """

    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    average = average.lower()
    if average not in {"binary", "macro", "weighted", "micro"}:
        raise ValueError("Invalid average. Choose from 'binary', 'macro', 'weighted', 'micro'.")

    if average == "binary":
        if num_classes != 2:
            raise ValueError("Binary averaging requires num_classes == 2.")
        idx = 1
        precision = float(_safe_div(tp[idx], tp[idx] + fp[idx]))
        recall = float(_safe_div(tp[idx], tp[idx] + fn[idx]))
        denom = precision + recall
        f1 = float(0.0 if denom == 0 else 2 * precision * recall / denom)
        return {"precision": precision, "recall": recall, "f1": f1}

    if average == "micro":
        tp_total = float(tp.sum())
        fp_total = float(fp.sum())
        fn_total = float(fn.sum())
        precision = 0.0 if tp_total + fp_total == 0 else tp_total / (tp_total + fp_total)
        recall = 0.0 if tp_total + fn_total == 0 else tp_total / (tp_total + fn_total)
        denom = precision + recall
        f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
        return {"precision": precision, "recall": recall, "f1": f1}

    per_class_precision = _safe_div(tp, tp + fp)
    per_class_recall = _safe_div(tp, tp + fn)
    per_class_f1 = _safe_div(2 * per_class_precision * per_class_recall, per_class_precision + per_class_recall)

    if average == "macro":
        precision = float(np.nan_to_num(per_class_precision).mean())
        recall = float(np.nan_to_num(per_class_recall).mean())
        f1 = float(np.nan_to_num(per_class_f1).mean())
        return {"precision": precision, "recall": recall, "f1": f1}

    weights = support
    total_weight = float(weights.sum())
    if total_weight == 0:
        precision = recall = f1 = 0.0
    else:
        precision = float(np.nan_to_num(per_class_precision * weights).sum() / total_weight)
        recall = float(np.nan_to_num(per_class_recall * weights).sum() / total_weight)
        f1 = float(np.nan_to_num(per_class_f1 * weights).sum() / total_weight)
    return {"precision": precision, "recall": recall, "f1": f1}


def pr_curve_binary(
    y_true: np.ndarray,
    probs_pos: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels of shape ``(N,)`` with values in ``{0, 1}``.
    probs_pos : np.ndarray
        Positive-class probabilities of shape ``(N,)`` in ``[0, 1]``.
    num_thresholds : int, optional
        Maximum number of thresholds to evaluate. If the number of unique
        scores exceeds this value, thresholds are subsampled at quantile
        positions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precision array of length ``T + 1``, recall array of length ``T + 1``
        and descending thresholds array of length ``T``.
    """

    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive.")
    y_true = to_numpy(y_true)
    probs = to_numpy(probs_pos)
    if y_true.ndim != 1 or probs.ndim != 1:
        raise ValueError("y_true and probs_pos must be 1-D arrays.")
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probs_pos must have the same length.")
    if y_true.size == 0:
        raise ValueError("Inputs must be non-empty.")
    if not np.issubdtype(y_true.dtype, np.integer):
        y_true = y_true.astype(np.int64, copy=False)
    if not np.array_equal(y_true, y_true.astype(bool)):
        raise ValueError("y_true must contain binary labels {0, 1}.")

    probs = probs.astype(np.float64, copy=False)
    if np.any(np.isnan(probs)):
        raise ValueError("probs_pos must not contain NaNs.")

    sort_idx = np.argsort(-probs, kind="mergesort")
    probs_sorted = probs[sort_idx]
    y_sorted = y_true[sort_idx]

    total_pos = np.sum(y_sorted)
    tp_cum = np.cumsum(y_sorted, dtype=np.float64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.float64)

    if probs_sorted.size == 0:
        return np.array([1.0]), np.array([0.0]), np.array([], dtype=np.float64)

    distinct_mask = np.r_[True, probs_sorted[1:] != probs_sorted[:-1]]
    threshold_indices = np.flatnonzero(distinct_mask)
    if threshold_indices.size > num_thresholds:
        positions = np.linspace(0, threshold_indices.size - 1, num_thresholds, dtype=int)
        threshold_indices = threshold_indices[positions]

    thresholds = probs_sorted[threshold_indices]

    precision_values = [1.0]
    recall_values = [0.0]
    for idx in threshold_indices:
        tp = tp_cum[idx]
        fp = fp_cum[idx]
        denom = tp + fp
        precision = 1.0 if denom == 0 else tp / denom
        recall = 0.0 if total_pos == 0 else tp / total_pos
        precision_values.append(float(precision))
        recall_values.append(float(recall))

    precision_arr = np.asarray(precision_values, dtype=np.float64)
    recall_arr = np.asarray(recall_values, dtype=np.float64)

    return precision_arr, recall_arr, thresholds.astype(np.float64, copy=False)


def pr_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute the area under a precision-recall curve.

    Parameters
    ----------
    precision, recall : np.ndarray
        Arrays of identical length ``T``. Recall values must be sorted in
        non-decreasing order.

    Returns
    -------
    float
        Scalar area under the precision-recall curve in ``[0, 1]``.
    """

    precision = to_numpy(precision).astype(np.float64, copy=False)
    recall = to_numpy(recall).astype(np.float64, copy=False)
    if precision.shape != recall.shape:
        raise ValueError("precision and recall must have the same shape.")
    if precision.ndim != 1:
        raise ValueError("precision and recall must be 1-D arrays.")
    if precision.size == 0:
        raise ValueError("precision and recall must be non-empty.")
    if np.any(np.diff(recall) < -1e-12):
        raise ValueError("recall must be sorted in non-decreasing order.")

    area = float(np.trapz(precision, recall))
    return max(0.0, min(1.0, area))


def fbeta_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
    average: str = "binary",
    num_classes: int = 2,
) -> float:
    """Compute the generalized :math:`F_{\beta}` score.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of shape ``(N,)`` with integer labels.
    beta : float, optional
        Balance parameter (``beta > 0``). Values ``< 1`` weigh precision more
        heavily than recall.
    average : str, optional
        Averaging strategy identical to :func:`precision_recall_f1`.
    num_classes : int, optional
        Number of classes for non-binary averaging.

    Returns
    -------
    float
        F-beta score in ``[0, 1]``.
    """

    if beta <= 0:
        raise ValueError("beta must be positive.")

    metrics = precision_recall_f1(y_true, y_pred, num_classes=num_classes, average=average)
    precision = metrics["precision"]
    recall = metrics["recall"]
    denom = (beta ** 2) * precision + recall
    if denom == 0.0:
        return 0.0
    return float((1 + beta ** 2) * precision * recall / denom)


if __name__ == "__main__":  # pragma: no cover - sanity checks
    # Binary example
    y_t = np.array([0, 1, 1, 0, 1, 0, 1])
    y_p = np.array([0, 1, 0, 0, 1, 1, 1])
    cm = confusion_matrix(y_t, y_p, num_classes=2)
    assert cm.tolist() == [[3, 1], [1, 2]]
    metrics = precision_recall_f1(y_t, y_p, num_classes=2)
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 3)
    assert np.isclose(metrics["f1"], 2 / 3)

    probs = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4, 0.95])
    prec, rec, thr = pr_curve_binary(y_t, probs)
    assert np.all(np.diff(rec) >= -1e-12)
    auc = pr_auc(prec, rec)
    assert 0.0 <= auc <= 1.0

    # Multi-class example
    y_t = np.array([0, 1, 2, 2, 1, 0])
    y_p = np.array([0, 2, 2, 1, 1, 0])
    metrics_macro = precision_recall_f1(y_t, y_p, num_classes=3, average="macro")
    metrics_micro = precision_recall_f1(y_t, y_p, num_classes=3, average="micro")
    assert metrics_macro["f1"] <= 1.0
    assert metrics_micro["precision"] == metrics_micro["recall"]

    print("Classification metrics sanity checks passed.")
