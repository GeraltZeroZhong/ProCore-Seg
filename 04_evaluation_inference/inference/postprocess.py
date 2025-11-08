from __future__ import annotations

"""Post-processing utilities for ProCore-Seg inference outputs."""

from typing import Tuple

import numpy as np
import torch


def temperature_scale(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Return temperature-scaled logits without modifying the input tensor."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim < 1:
        raise ValueError("logits tensor must have at least one dimension")
    if T <= 0:
        raise ValueError("Temperature T must be positive")
    return logits / T


def to_atom_space(
    logits_voxel: torch.Tensor,
    atom2voxel: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project voxel logits to atom space using ``atom2voxel`` mapping."""

    if not isinstance(logits_voxel, torch.Tensor):
        raise TypeError("logits_voxel must be a torch.Tensor")
    if not isinstance(atom2voxel, torch.Tensor):
        raise TypeError("atom2voxel must be a torch.Tensor")
    if logits_voxel.ndim != 2:
        raise ValueError("logits_voxel must have shape (V, C)")
    if atom2voxel.ndim != 1:
        raise ValueError("atom2voxel must be a 1-D tensor")
    if logits_voxel.shape[0] == 0:
        raise ValueError("logits_voxel must contain at least one voxel")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits_scaled = temperature_scale(logits_voxel, temperature)
    probs_voxel = torch.softmax(logits_scaled, dim=1)
    if atom2voxel.numel() == 0:
        empty_probs = probs_voxel.new_empty((0, probs_voxel.shape[1])).to(torch.float32)
        empty_pred = probs_voxel.new_empty((0,), dtype=torch.int64)
        return empty_probs, empty_pred

    max_index = atom2voxel.max().item()
    if max_index >= probs_voxel.shape[0]:
        raise ValueError("atom2voxel contains indices out of range")

    probs_atom = probs_voxel[atom2voxel.long()]
    pred_atom = probs_atom.argmax(dim=1)
    return probs_atom.to(torch.float32), pred_atom.to(torch.int64)


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along ``axis``."""

    if x.ndim == 0:
        raise ValueError("softmax requires at least one dimension")
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp = np.exp(shifted)
    exp_sum = np.sum(exp, axis=axis, keepdims=True)
    if np.any(exp_sum == 0):
        raise ValueError("softmax encountered zero denominator")
    return exp / exp_sum


def argmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return ``np.argmax`` as ``np.int64``."""

    return np.asarray(np.argmax(x, axis=axis), dtype=np.int64)


if __name__ == "__main__":  # pragma: no cover - simple self-check
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    mapping = torch.tensor([0, 1, 1, 0])
    probs, labels = to_atom_space(logits, mapping)
    assert probs.shape == (4, 2)
    assert labels.shape == (4,)
    arr = np.array([[0.0, 1.0], [1.0, 0.0]])
    sm = softmax_numpy(arr, axis=1)
    assert sm.shape == arr.shape
    print("postprocess self-check passed")
