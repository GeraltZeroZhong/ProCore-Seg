from __future__ import annotations

"""Intersection-over-Union metrics for atom-level segmentation."""

from typing import Dict, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def to_numpy(x) -> np.ndarray:
    """Convert array-like inputs to contiguous :class:`np.ndarray`.

    Parameters
    ----------
    x : Any
        Array-like structure (NumPy, PyTorch, list).

    Returns
    -------
    np.ndarray
        Contiguous array with preserved dtype when feasible.
    """

    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    if torch is not None and isinstance(x, torch.Tensor):  # pragma: no cover
        return np.ascontiguousarray(x.detach().cpu().numpy())
    return np.ascontiguousarray(np.array(x))


def _validate_binary_mask(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have identical length.")
    if not np.issubdtype(y_true.dtype, np.integer):
        y_true = y_true.astype(np.int64, copy=False)
    if not np.issubdtype(y_pred.dtype, np.integer):
        y_pred = y_pred.astype(np.int64, copy=False)
    if not np.array_equal(y_true, y_true.astype(bool)):
        raise ValueError("y_true must contain binary labels {0, 1}.")
    if not np.array_equal(y_pred, y_pred.astype(bool)):
        raise ValueError("y_pred must contain binary labels {0, 1}.")
    return y_true, y_pred


def iou_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary IoU between two masks.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Binary arrays of shape ``(N,)`` containing ``{0, 1}``.

    Returns
    -------
    float
        Intersection-over-Union in ``[0, 1]``. When both masks are empty the IoU
        is defined as ``1.0``.
    """

    y_true, y_pred = _validate_binary_mask(y_true, y_pred)
    intersection = float(np.dot(y_true, y_pred))
    union = float(y_true.sum() + y_pred.sum() - intersection)
    if union == 0.0:
        return 1.0
    return intersection / union


def iou_per_chain(
    chains: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute IoU for each chain separately.

    Parameters
    ----------
    chains : Sequence[str]
        Chain identifier per atom; length ``N``.
    y_true, y_pred : np.ndarray
        Binary labels/predictions of shape ``(N,)``.

    Returns
    -------
    Dict[str, float]
        Mapping ``chain_id -> IoU``.
    """

    chains_arr = np.asarray(chains)
    if chains_arr.ndim != 1:
        raise ValueError("chains must be a 1-D sequence.")
    y_true, y_pred = _validate_binary_mask(y_true, y_pred)
    if chains_arr.shape[0] != y_true.shape[0]:
        raise ValueError("chains length must match y_true/y_pred length.")

    results: Dict[str, float] = {}
    for chain_id in np.unique(chains_arr):
        mask = chains_arr == chain_id
        results[chain_id] = iou_binary(y_true[mask], y_pred[mask])
    return results


def mean_iou_macro(
    chains: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-average IoU across chains.

    Parameters
    ----------
    chains : Sequence[str]
        Chain identifier per atom; length ``N``.
    y_true, y_pred : np.ndarray
        Binary arrays of shape ``(N,)`` with values in ``{0, 1}``.

    Returns
    -------
    float
        Mean IoU across chains. Chains with no positives and no predicted
        positives are treated as perfectly overlapping (IoU ``= 1``).
    """

    per_chain = iou_per_chain(chains, y_true, y_pred)
    if not per_chain:
        raise ValueError("No chains provided.")
    return float(np.mean(list(per_chain.values())))


def core_mask_from_density(
    density: np.ndarray,
    T: Optional[float] = None,
    strategy: str = "median",
) -> np.ndarray:
    """Derive a boolean core mask from density values.

    Parameters
    ----------
    density : np.ndarray
        Atom densities of shape ``(N,)``.
    T : float, optional
        Explicit threshold used when ``strategy == 'fixed'``. When ``None`` the
        threshold is computed from the chosen strategy.
    strategy : {"median", "p25", "p50", "p75", "fixed"}
        Rule for computing the threshold. Percentile-based strategies apply to
        ``density``.

    Returns
    -------
    np.ndarray
        Boolean mask where ``True`` denotes core atoms (``density >= threshold``).
    """

    density = to_numpy(density).astype(np.float64, copy=False)
    if density.ndim != 1:
        raise ValueError("density must be 1-D.")
    if density.size == 0:
        raise ValueError("density must not be empty.")

    strategy = strategy.lower()
    if strategy not in {"median", "p25", "p50", "p75", "fixed"}:
        raise ValueError("Invalid strategy for core mask.")

    if strategy == "fixed":
        if T is None:
            raise ValueError("Threshold T must be provided when strategy='fixed'.")
        threshold = float(T)
    else:
        percentile_map = {"median": 50.0, "p25": 25.0, "p50": 50.0, "p75": 75.0}
        percentile = percentile_map[strategy]
        try:
            threshold = float(np.percentile(density, percentile, method="linear"))
        except TypeError:  # pragma: no cover - fallback for older NumPy
            threshold = float(np.percentile(density, percentile, interpolation="linear"))
        if T is not None:
            threshold = float(T)

    return density >= threshold


def core_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    density: np.ndarray,
    T: Optional[float] = None,
    strategy: str = "median",
) -> float:
    """Compute IoU restricted to core atoms defined by ``density``.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Binary arrays of shape ``(N,)``.
    density : np.ndarray
        Density scores of shape ``(N,)``.
    T : float, optional
        Explicit threshold overriding ``strategy`` when provided.
    strategy : {"median", "p25", "p50", "p75", "fixed"}
        Strategy for deriving the core mask.

    Returns
    -------
    float
        IoU computed on atoms selected by the core mask. If no atom is selected
        the IoU is defined as ``1.0``.
    """

    y_true_arr, y_pred_arr = _validate_binary_mask(y_true, y_pred)
    mask = core_mask_from_density(density, T=T, strategy=strategy)
    if mask.shape[0] != y_true_arr.shape[0]:
        raise ValueError("density length must match predictions.")
    if not np.any(mask):
        return 1.0
    return iou_binary(y_true_arr[mask], y_pred_arr[mask])


def iou_and_core_iou_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    density: np.ndarray,
) -> Dict[str, float]:
    """Generate IoU report including multiple core density thresholds.

    Returns a dictionary with keys ``iou``, ``core_iou_p25``, ``core_iou_p50``
    and ``core_iou_p75`` representing the IoU computed on all atoms and on
    cores defined via the 25th, 50th and 75th percentiles of ``density``.
    """

    base = iou_binary(y_true, y_pred)
    density = to_numpy(density)
    report = {"iou": base}
    for strategy, key in [("p25", "core_iou_p25"), ("p50", "core_iou_p50"), ("p75", "core_iou_p75")]:
        report[key] = core_iou(y_true, y_pred, density, strategy=strategy)
    return report


if __name__ == "__main__":  # pragma: no cover - sanity checks
    y_true = np.array([0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0])
    chains = np.array(["A", "A", "A", "B", "B", "B", "B"])
    density = np.linspace(0.1, 0.7, num=y_true.size)

    assert np.isclose(iou_binary(y_true, y_true), 1.0)
    assert np.isclose(iou_binary(y_true, np.zeros_like(y_true)), 0.0)
    per_chain = iou_per_chain(chains, y_true, y_pred)
    assert set(per_chain) == {"A", "B"}
    report = iou_and_core_iou_report(y_true, y_pred, density)
    assert 0.0 <= report["iou"] <= 1.0
    assert 0.0 <= report["core_iou_p50"] <= 1.0

    print("IoU metrics sanity checks passed.")
