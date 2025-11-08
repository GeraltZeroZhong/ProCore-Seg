from __future__ import annotations

"""Boundary quality metrics for residue-level segmentation."""

from typing import Dict, List, Sequence, Tuple

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


def residues_from_atoms(
    chains: Sequence[str],
    resseq: Sequence[int],
    icode: Sequence[str],
    y_atom: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate atom-level annotations to residue level by majority vote.

    Parameters
    ----------
    chains, resseq, icode : Sequence
        Atom-level identifiers of equal length ``N``.
    y_atom : Sequence[int]
        Binary atom annotations of length ``N`` with values in ``{0, 1}``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Residue-level ``(chain, resseq, icode, label)`` arrays sorted by
        ``(chain, resseq, icode)``. Labels are integers ``0`` or ``1`` with ties
        resolved in favour of the positive class.
    """

    chains_arr = np.asarray(chains)
    resseq_arr = np.asarray(resseq)
    icode_arr = np.asarray(icode)
    y_atom_arr = to_numpy(y_atom)

    if not (chains_arr.shape == resseq_arr.shape == icode_arr.shape == y_atom_arr.shape):
        raise ValueError("All atom-level inputs must share the same shape.")

    keys = list(zip(chains_arr.tolist(), resseq_arr.tolist(), icode_arr.tolist()))
    order = np.lexsort((icode_arr, resseq_arr, chains_arr))

    residues: List[Tuple[str, int, str, int]] = []
    i = 0
    while i < len(order):
        idx = order[i]
        key = keys[idx]
        members = [idx]
        j = i + 1
        while j < len(order) and keys[order[j]] == key:
            members.append(order[j])
            j += 1
        votes = y_atom_arr[members]
        positives = np.count_nonzero(votes)
        negatives = votes.size - positives
        residue_label = 1 if positives >= negatives else 0
        residues.append((key[0], key[1], key[2], residue_label))
        i = j

    chains_res = np.array([r[0] for r in residues])
    resseq_res = np.array([r[1] for r in residues])
    icode_res = np.array([r[2] for r in residues])
    y_res = np.array([r[3] for r in residues], dtype=np.int64)
    return chains_res, resseq_res, icode_res, y_res


def boundary_indices(y_res: np.ndarray) -> np.ndarray:
    """Return indices of transitions between consecutive residue labels.

    Parameters
    ----------
    y_res : np.ndarray
        Residue-level labels of shape ``(R,)`` with integer values ``0`` or ``1``.

    Returns
    -------
    np.ndarray
        Sorted indices ``i`` in ``[0, R-2]`` where ``y_res[i] != y_res[i+1]``.
    """

    y_res = to_numpy(y_res)
    if y_res.ndim != 1:
        raise ValueError("y_res must be 1-D.")
    if y_res.size < 2:
        return np.array([], dtype=np.int64)
    if not np.issubdtype(y_res.dtype, np.integer):
        y_res = y_res.astype(np.int64, copy=False)
    diffs = np.diff(y_res)
    indices = np.flatnonzero(diffs != 0)
    return indices.astype(np.int64, copy=False)


def _match_boundaries(true_idx: np.ndarray, pred_idx: np.ndarray, k: int) -> Tuple[int, int, int]:
    true_idx = np.sort(true_idx.astype(np.int64, copy=False))
    pred_idx = np.sort(pred_idx.astype(np.int64, copy=False))
    matches = 0
    i = j = 0
    while i < true_idx.size and j < pred_idx.size:
        diff = pred_idx[j] - true_idx[i]
        if abs(diff) <= k:
            matches += 1
            i += 1
            j += 1
        elif pred_idx[j] < true_idx[i]:
            j += 1
        else:
            i += 1
    return matches, true_idx.size, pred_idx.size


def boundary_f1_at_k(y_true_res: np.ndarray, y_pred_res: np.ndarray, k: int) -> float:
    """Compute boundary F1 within ±k residues tolerance.

    Parameters
    ----------
    y_true_res, y_pred_res : np.ndarray
        Residue-level labels of shape ``(R,)`` with values in ``{0, 1}``.
    k : int
        Tolerance window in residues.

    Returns
    -------
    float
        F1-score of matched boundary points.
    """

    if k < 0:
        raise ValueError("k must be non-negative.")
    true_idx = boundary_indices(y_true_res)
    pred_idx = boundary_indices(y_pred_res)
    matches, n_true, n_pred = _match_boundaries(true_idx, pred_idx, k)
    if n_true == 0 and n_pred == 0:
        return 1.0
    precision = 0.0 if n_pred == 0 else matches / n_pred
    recall = 0.0 if n_true == 0 else matches / n_true
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2 * precision * recall / denom


def _dilated_boundary_mask(length: int, boundary_idx: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if boundary_idx.size == 0:
        return mask
    for idx in boundary_idx:
        center = idx + 1
        start = max(0, center - k)
        end = min(length, center + k + 1)
        mask[start:end] = True
    return mask


def boundary_iou_at_k(y_true_res: np.ndarray, y_pred_res: np.ndarray, k: int) -> float:
    """Compute IoU between dilated boundary bands within ±k residues.

    Parameters
    ----------
    y_true_res, y_pred_res : np.ndarray
        Residue-level labels of shape ``(R,)`` with values in ``{0, 1}``.
    k : int
        Half-width of the dilation band in residues.

    Returns
    -------
    float
        IoU between dilated boundary regions. If both regions are empty the
        IoU is defined as ``1.0``.
    """

    if k < 0:
        raise ValueError("k must be non-negative.")
    y_true_res = to_numpy(y_true_res)
    y_pred_res = to_numpy(y_pred_res)
    if y_true_res.ndim != 1 or y_pred_res.ndim != 1:
        raise ValueError("y_true_res and y_pred_res must be 1-D.")
    if y_true_res.shape[0] != y_pred_res.shape[0]:
        raise ValueError("Residue sequences must share the same length.")
    n = y_true_res.shape[0]
    true_mask = _dilated_boundary_mask(n, boundary_indices(y_true_res), k)
    pred_mask = _dilated_boundary_mask(n, boundary_indices(y_pred_res), k)
    intersection = np.count_nonzero(true_mask & pred_mask)
    union = np.count_nonzero(true_mask | pred_mask)
    if union == 0:
        return 1.0
    return intersection / union


def boundary_report(
    chains: Sequence[str],
    resseq: Sequence[int],
    icode: Sequence[str],
    y_true_atom: Sequence[int],
    y_pred_atom: Sequence[int],
    ks: Sequence[int] = (2, 4, 6),
) -> Dict[str, float]:
    """Full boundary report across tolerance values ``ks``.

    Parameters
    ----------
    chains, resseq, icode : Sequence
        Atom-level identifiers of equal length ``N``.
    y_true_atom, y_pred_atom : Sequence[int]
        Atom-level binary labels of length ``N``.
    ks : Sequence[int], optional
        Sequence of tolerance values in residues.

    Returns
    -------
    Dict[str, float]
        Mapping of metric names ``boundary_f1@k`` and ``boundary_iou@k`` to
        scalar scores.
    """

    chains_res, resseq_res, icode_res, y_true_res = residues_from_atoms(chains, resseq, icode, y_true_atom)
    _, _, _, y_pred_res = residues_from_atoms(chains, resseq, icode, y_pred_atom)

    if not ks:
        raise ValueError("ks must contain at least one tolerance value.")

    ks = tuple(int(k) for k in ks)
    for k in ks:
        if k < 0:
            raise ValueError("Tolerance k must be non-negative.")

    report: Dict[str, float] = {}
    unique_chains, chain_indices = np.unique(chains_res, return_inverse=True)

    for k in ks:
        matches_total = 0
        true_total = 0
        pred_total = 0
        intersection = 0
        union = 0
        for chain_id in range(unique_chains.size):
            mask = chain_indices == chain_id
            y_true_chain = y_true_res[mask]
            y_pred_chain = y_pred_res[mask]
            true_idx = boundary_indices(y_true_chain)
            pred_idx = boundary_indices(y_pred_chain)
            matches, n_true, n_pred = _match_boundaries(true_idx, pred_idx, k)
            matches_total += matches
            true_total += n_true
            pred_total += n_pred
            n_chain = y_true_chain.size
            true_mask = _dilated_boundary_mask(n_chain, true_idx, k)
            pred_mask = _dilated_boundary_mask(n_chain, pred_idx, k)
            intersection += int(np.count_nonzero(true_mask & pred_mask))
            union += int(np.count_nonzero(true_mask | pred_mask))
        if true_total == 0 and pred_total == 0:
            f1 = 1.0
        else:
            precision = 0.0 if pred_total == 0 else matches_total / pred_total
            recall = 0.0 if true_total == 0 else matches_total / true_total
            denom = precision + recall
            f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
        iou = 1.0 if union == 0 else intersection / union
        report[f"boundary_f1@{k}"] = float(f1)
        report[f"boundary_iou@{k}"] = float(iou)
    return report


if __name__ == "__main__":  # pragma: no cover - sanity checks
    chains = ["A", "A", "A", "A", "A", "A"]
    resseq = [1, 1, 1, 2, 2, 3]
    icode = [" ", " ", " ", " ", " ", " "]
    y_true_atom = [0, 0, 1, 1, 1, 0]
    y_pred_atom = [0, 1, 1, 1, 0, 0]

    chains_res, resseq_res, icode_res, y_true_res = residues_from_atoms(chains, resseq, icode, y_true_atom)
    _, _, _, y_pred_res = residues_from_atoms(chains, resseq, icode, y_pred_atom)
    assert chains_res.shape[0] == y_true_res.shape[0]
    f1_k2 = boundary_f1_at_k(y_true_res, y_pred_res, k=2)
    f1_k4 = boundary_f1_at_k(y_true_res, y_pred_res, k=4)
    assert f1_k4 >= f1_k2
    report = boundary_report(chains, resseq, icode, y_true_atom, y_pred_atom, ks=(2,))
    assert "boundary_f1@2" in report

    print("Boundary metrics sanity checks passed.")
