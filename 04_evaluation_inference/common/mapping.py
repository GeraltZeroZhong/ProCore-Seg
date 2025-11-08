from __future__ import annotations

"""Atom-to-voxel mapping utilities shared across evaluation and inference."""

from typing import Optional, Tuple

import numpy as np


def quantize_coords(coords_angstrom: np.ndarray, voxel_size: float) -> np.ndarray:
    """Quantise Euclidean coordinates into an integer lattice."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    coords = np.asarray(coords_angstrom, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_angstrom must have shape (N, 3)")
    quantised = np.floor(coords / voxel_size).astype(np.int32)
    return quantised


def unique_with_inverse(int_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lexicographically ordered unique integer coordinates with indices."""

    coords = np.asarray(int_coords, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("int_coords must have shape (N, 3)")
    if coords.size == 0:
        empty = np.empty((0, 3), dtype=np.int32)
        return empty, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    sorted_coords = coords[order]
    diffs = np.diff(sorted_coords, axis=0)
    unique_mask = np.ones(len(sorted_coords), dtype=bool)
    unique_mask[1:] = np.any(diffs != 0, axis=1)
    unique_coords = sorted_coords[unique_mask]
    index = order[unique_mask]
    inverse = np.empty(len(coords), dtype=np.int64)
    inverse[order] = np.cumsum(unique_mask) - 1
    return unique_coords.astype(np.int32), index.astype(np.int64), inverse


def reduce_features_by_inverse(
    features: np.ndarray,
    inverse: np.ndarray,
    V: Optional[int] = None,
    reduction: str = "mean",
) -> np.ndarray:
    """Reduce features for atoms that map to the same voxel."""

    feats = np.asarray(features)
    if feats.ndim != 2:
        raise ValueError("features must be a 2D array")
    inv = np.asarray(inverse, dtype=np.int64)
    if inv.ndim != 1:
        raise ValueError("inverse must be 1D")
    if feats.shape[0] != inv.shape[0]:
        raise ValueError("features and inverse must align")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    max_index = int(inv.max()) if inv.size else -1
    V = int(V) if V is not None else (max_index + 1)
    if V < 0:
        raise ValueError("V must be non-negative")
    if inv.size and V <= max_index:
        raise ValueError("V must exceed the maximum index in inverse")
    voxel_feats = np.zeros((V, feats.shape[1]), dtype=feats.dtype)
    if inv.size:
        np.add.at(voxel_feats, inv, feats)
        if reduction == "mean":
            counts = np.bincount(inv, minlength=V).reshape(-1, 1)
            counts[counts == 0] = 1
            voxel_feats = voxel_feats / counts
    return voxel_feats


def voxel_centers_from_int(int_coords_unique: np.ndarray, voxel_size: float) -> np.ndarray:
    """Compute real-valued voxel centres from integer coordinates."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    coords = np.asarray(int_coords_unique, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("int_coords_unique must have shape (V, 3)")
    return (coords.astype(np.float32) * float(voxel_size)).astype(np.float32)


def try_build_me_sparse(
    int_coords_unique: np.ndarray,
    voxel_features: np.ndarray,
    device: str = "cpu",
):
    """Attempt to construct a MinkowskiEngine sparse tensor; return ``None`` if unavailable."""

    try:
        import torch
        import MinkowskiEngine as ME  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None

    coords = np.asarray(int_coords_unique, dtype=np.int32)
    feats = np.asarray(voxel_features)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("int_coords_unique must have shape (V, 3)")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError("voxel_features must have shape (V, D)")
    batch_col = np.zeros((coords.shape[0], 1), dtype=np.int32)
    coords_with_batch = np.concatenate([batch_col, coords], axis=1)
    coordinates = torch.as_tensor(coords_with_batch, dtype=torch.int32, device=device)
    features = torch.as_tensor(feats, dtype=torch.float32, device=device)
    return ME.SparseTensor(features=features, coordinates=coordinates)  # type: ignore[return-value]


def me_batched_coordinates(int_coords_list: list[np.ndarray]) -> Optional[np.ndarray]:
    """Construct ME batched coordinates if MinkowskiEngine is installed."""

    try:
        import MinkowskiEngine as ME  # type: ignore  # noqa: F401
    except ImportError:  # pragma: no cover - optional dependency
        return None
    coords_with_batch: list[np.ndarray] = []
    for batch_index, coords in enumerate(int_coords_list):
        arr = np.asarray(coords, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Each coordinate array must have shape (N, 3)")
        batch_col = np.full((arr.shape[0], 1), batch_index, dtype=np.int32)
        coords_with_batch.append(np.concatenate([batch_col, arr], axis=1))
    if not coords_with_batch:
        return np.empty((0, 4), dtype=np.int32)
    return np.vstack(coords_with_batch)


if __name__ == "__main__":
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.1, 0.0, 0.0]], dtype=np.float32)
    voxel_size = 0.5
    int_coords = quantize_coords(coords, voxel_size)
    unique, index, inverse = unique_with_inverse(int_coords)
    features = np.arange(len(coords) * 2, dtype=np.float32).reshape(len(coords), 2)
    reduced_mean = reduce_features_by_inverse(features, inverse, V=unique.shape[0], reduction="mean")
    reduced_sum = reduce_features_by_inverse(features, inverse, V=unique.shape[0], reduction="sum")
    print("Unique:", unique)
    print("Index:", index)
    print("Inverse:", inverse)
    print("Reduced mean:", reduced_mean)
    print("Reduced sum:", reduced_sum)
