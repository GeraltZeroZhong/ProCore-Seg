from __future__ import annotations

"""Statistical utilities for evaluation reports."""

from typing import Callable, Dict, Optional, Tuple

import math

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


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional data array.
    stat : Callable[[np.ndarray], float], optional
        Statistic function applied to resampled data.
    n_boot : int, optional
        Number of bootstrap samples.
    alpha : float, optional
        Significance level (two-sided).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        Statistic computed on original data and the lower/upper percentile
        bounds.
    """

    values_arr = to_numpy(values).astype(np.float64, copy=False)
    if values_arr.ndim != 1:
        raise ValueError("values must be a 1-D array.")
    n = values_arr.size
    if n == 0:
        raise ValueError("values must not be empty.")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")

    statistic = float(stat(values_arr))
    if n == 1:
        return statistic, statistic, statistic

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values_arr, size=n, replace=True)
        boot_stats[i] = float(stat(sample))

    try:
        lower = float(np.percentile(boot_stats, 100 * (alpha / 2), method="linear"))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2), method="linear"))
    except TypeError:  # pragma: no cover - fallback for NumPy < 1.22
        lower = float(np.percentile(boot_stats, 100 * (alpha / 2), interpolation="linear"))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2), interpolation="linear"))
    return statistic, lower, upper


def _rank_abs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_vals = values[order]
    ranks = np.empty_like(sorted_vals, dtype=np.float64)
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and math.isclose(sorted_vals[j], sorted_vals[i]):
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[i:j] = rank
        i = j
    unsorted = np.empty_like(ranks)
    unsorted[order] = ranks
    return unsorted


def paired_wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test using a normal approximation.

    Parameters
    ----------
    a, b : np.ndarray
        Paired measurements of shape ``(N,)``. Arrays must have identical
        length.

    Returns
    -------
    Dict[str, float]
        Dictionary with statistic ``W``, normal approximation ``z`` score,
        two-sided p-value ``p`` and effective sample size ``n`` after removing
        zero differences.
    """

    a_arr = to_numpy(a).astype(np.float64, copy=False)
    b_arr = to_numpy(b).astype(np.float64, copy=False)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must share the same shape.")
    if a_arr.ndim != 1:
        raise ValueError("a and b must be 1-D arrays.")

    diff = b_arr - a_arr
    mask = diff != 0
    diff = diff[mask]
    n_eff = diff.size
    if n_eff == 0:
        return {"W": 0.0, "z": math.nan, "p": math.nan, "n": 0.0}

    abs_diff = np.abs(diff)
    ranks = _rank_abs(abs_diff)
    signs = np.sign(diff)
    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    W = w_plus

    if n_eff < 2:
        return {"W": W, "z": math.nan, "p": math.nan, "n": float(n_eff)}

    mean_w = n_eff * (n_eff + 1) / 4.0
    tie_counts = []
    order = np.argsort(abs_diff)
    sorted_abs = abs_diff[order]
    i = 0
    while i < n_eff:
        j = i + 1
        while j < n_eff and math.isclose(sorted_abs[j], sorted_abs[i]):
            j += 1
        tie_counts.append(j - i)
        i = j
    tie_correction = sum(t * (t * t - 1) for t in tie_counts) / 48.0
    var_w = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0 - tie_correction
    if var_w <= 0:
        return {"W": W, "z": math.nan, "p": math.nan, "n": float(n_eff)}

    sigma = math.sqrt(var_w)
    continuity = 0.5 * math.copysign(1.0, W - mean_w) if W != mean_w else 0.0
    z = (W - mean_w - continuity) / sigma
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return {"W": W, "z": z, "p": p, "n": float(n_eff)}


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's :math:`d_z` effect size for paired samples.

    Parameters
    ----------
    a, b : np.ndarray
        Paired measurements of shape ``(N,)``.

    Returns
    -------
    float
        Effect size defined as ``mean(b - a) / std(b - a)``. Returns ``NaN``
        when fewer than two non-zero differences exist.
    """

    a_arr = to_numpy(a).astype(np.float64, copy=False)
    b_arr = to_numpy(b).astype(np.float64, copy=False)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must share the same shape.")
    if a_arr.ndim != 1:
        raise ValueError("a and b must be 1-D.")
    diff = b_arr - a_arr
    if diff.size < 2:
        return math.nan
    std = np.std(diff, ddof=1)
    if std == 0.0:
        return math.nan
    return float(np.mean(diff) / std)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cliff's delta effect size in ``[-1, 1]``.

    Parameters
    ----------
    a, b : np.ndarray
        Independent samples of shape ``(N,)`` and ``(M,)`` respectively.

    Returns
    -------
    float
        Cliff's delta ``δ`` measuring ordinal dominance. Positive values
        indicate that ``a`` tends to be larger than ``b``.
    """

    a_arr = to_numpy(a).astype(np.float64, copy=False)
    b_arr = to_numpy(b).astype(np.float64, copy=False)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")
    n = a_arr.size
    m = b_arr.size
    if n == 0 or m == 0:
        raise ValueError("Inputs must be non-empty.")

    combined = np.concatenate([a_arr, b_arr])
    unique_vals, inverse = np.unique(combined, return_inverse=True)
    counts_a = np.bincount(inverse[:n], minlength=unique_vals.size)
    counts_b = np.bincount(inverse[n:], minlength=unique_vals.size)

    cum_b_less = np.cumsum(counts_b) - counts_b
    greater = float(np.sum(counts_a * cum_b_less))
    less = float(np.sum(counts_a * (m - counts_b - cum_b_less)))

    return (greater - less) / (n * m)


def summarize_with_ci(
    values: np.ndarray,
    name: str = "metric",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Summarise values with bootstrap confidence interval.

    Returns a dictionary containing the mean of ``values`` (stored under the
    provided ``name``) and a percentile bootstrap confidence interval.
    """

    stat, ci_low, ci_high = bootstrap_ci(values, stat=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)
    values_arr = to_numpy(values)
    return {name: stat, "ci_low": ci_low, "ci_high": ci_high, "n": int(values_arr.size)}


if __name__ == "__main__":  # pragma: no cover - sanity checks
    rng = np.random.default_rng(0)
    data = rng.normal(size=20)
    stat, lo, hi = bootstrap_ci(data, seed=0)
    assert lo <= stat <= hi

    a = rng.normal(size=15)
    b = a + 0.5
    wilcoxon = paired_wilcoxon_signed_rank(a, b)
    assert wilcoxon["n"] <= 15
    assert np.isnan(wilcoxon["p"]) or wilcoxon["p"] <= 0.05

    delta = cliffs_delta(a, b)
    assert -1 <= delta <= 1

    print("Statistical utilities sanity checks passed.")
