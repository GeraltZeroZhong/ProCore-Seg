from __future__ import annotations

"""Data loading utilities for ProCore-Seg training.

This module provides :class:`ProteinVoxelDataset` to load curated HDF5 protein
entries and a :func:`collate_sparse_batch` function that converts atom-level
features into :class:`MinkowskiEngine.SparseTensor` batches. Quantisation is
performed lazily during collation which keeps the raw atomic coordinates
available for reconstruction losses and per-atom supervision. The collate
function additionally returns mappings between the original atoms and the
quantised voxels to support Stage-1 and Stage-2 training objectives.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import MinkowskiEngine as ME
import numpy as np
import torch


@dataclass(frozen=True)
class Sample:
    """Single protein entry containing atom-level data."""

    coords: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    meta: Dict[str, torch.Tensor | list | str]


@dataclass(frozen=True)
class Batch:
    """Batch representation containing sparse tensors and per-atom mappings."""

    sparse_inputs: ME.SparseTensor
    inverse_map_atom2voxel: torch.LongTensor
    atom_features: torch.Tensor
    atom_labels: torch.Tensor
    atom_coords: torch.Tensor
    voxel_coords_float: torch.Tensor
    batch_row_splits: torch.Tensor


class ProteinVoxelDataset(torch.utils.data.Dataset[Sample]):
    """Dataset providing :class:`Sample` objects from curated HDF5 files."""

    def __init__(self, root: str | Path, ids: Optional[Iterable[str]] = None) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist")

        if ids is not None:
            selected_ids = [str(i).strip() for i in ids if str(i).strip()]
            if not selected_ids:
                raise ValueError("ids iterable is empty after stripping values")
            paths = []
            for entry_id in selected_ids:
                path = self.root / f"{entry_id}.h5"
                if not path.exists():
                    raise FileNotFoundError(f"Expected dataset file '{path}' not found")
                paths.append(path)
        else:
            paths = sorted(self.root.glob("*.h5"))
            if not paths:
                raise FileNotFoundError(f"No '.h5' files discovered under '{self.root}'")

        self._paths: List[Path] = sorted(paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Sample:
        path = self._paths[index]
        with h5py.File(path, "r") as handle:
            coords_np = np.asarray(handle["coords"], dtype=np.float32)
            features_np = np.asarray(handle["features"], dtype=np.float32)
            labels_np = np.asarray(handle["labels"], dtype=np.int64).reshape(-1)

            meta: Dict[str, torch.Tensor | list | str] = {}
            if "meta" in handle:
                meta_group = handle["meta"]
                for key, value in meta_group.items():
                    data = value[()]
                    if isinstance(data, bytes):
                        meta[key] = data.decode("utf-8")
                    elif np.isscalar(data):
                        meta[key] = torch.as_tensor(data)
                    else:
                        array = np.asarray(data)
                        if array.dtype.kind in {"S", "U"}:
                            meta[key] = array.tolist()
                        else:
                            meta[key] = torch.from_numpy(array)
            for key, value in handle.attrs.items():
                if isinstance(value, bytes):
                    meta[key] = value.decode("utf-8")
                elif np.isscalar(value):
                    meta[key] = torch.as_tensor(value)
                else:
                    array = np.asarray(value)
                    if array.dtype.kind in {"S", "U"}:
                        meta[key] = array.tolist()
                    else:
                        meta[key] = torch.from_numpy(array)

        coords = torch.from_numpy(coords_np)
        features = torch.from_numpy(features_np)
        labels = torch.from_numpy(labels_np)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords to have shape (N, 3); received {coords.shape}")
        if features.ndim != 2 or features.shape[1] != 8:
            raise ValueError(
                f"Expected features to have shape (N, 8); received {features.shape}"
            )
        if labels.ndim != 1:
            raise ValueError(f"Expected labels to have shape (N,); received {labels.shape}")
        if coords.shape[0] != features.shape[0] or coords.shape[0] != labels.shape[0]:
            raise ValueError("coords, features, and labels must share the same first dimension")

        return Sample(coords=coords, features=features, labels=labels, meta=meta)


def _reduce_voxel_features(
    features: torch.Tensor, inverse: torch.LongTensor, num_voxels: int, reduction: str
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be either 'mean' or 'sum'")

    dtype = features.dtype
    device = features.device
    dim = features.shape[1]
    expanded_inverse = inverse.unsqueeze(1).expand(-1, dim)

    reduced = torch.zeros((num_voxels, dim), dtype=dtype, device=device)
    reduced.scatter_add_(0, expanded_inverse, features)

    counts = torch.bincount(inverse, minlength=num_voxels).to(dtype=dtype, device=device)
    counts = counts.clamp_min(1).unsqueeze(1)

    if reduction == "mean":
        reduced = reduced / counts
    else:  # sum
        reduced[:, 6:] = reduced[:, 6:] / counts

    return reduced


def collate_sparse_batch(
    samples: List[Sample],
    voxel_size: float = 1.0,
    quantize_reduction: str = "mean",
) -> Batch:
    """Collate samples into a MinkowskiEngine sparse batch with mappings."""

    if not samples:
        raise ValueError("collate_sparse_batch received an empty list of samples")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    atom_features_list: List[torch.Tensor] = []
    atom_labels_list: List[torch.Tensor] = []
    atom_coords_list: List[torch.Tensor] = []
    inverse_maps: List[torch.LongTensor] = []
    voxel_features_list: List[torch.Tensor] = []
    voxel_coords_list: List[torch.Tensor] = []
    coord_list: List[torch.IntTensor] = []
    splits = [0]
    voxel_offset = 0

    for batch_idx, sample in enumerate(samples):
        coords = sample.coords
        features = sample.features
        labels = sample.labels

        num_atoms = coords.shape[0]
        if num_atoms == 0:
            raise ValueError("Samples must contain at least one atom")

        # Augmentation hook: coordinate or feature perturbations can be inserted here.

        int_coords = torch.floor(coords / voxel_size).to(torch.int32)
        quantized, _, inverse = ME.utils.sparse_quantize(
            int_coords, return_index=True, return_inverse=True
        )

        if not isinstance(quantized, torch.Tensor):
            raise RuntimeError("sparse_quantize did not return tensor coordinates")

        num_voxels = quantized.shape[0]
        if num_voxels == 0:
            raise ValueError("Quantisation produced an empty set of voxels")

        reduced_features = _reduce_voxel_features(features, inverse, num_voxels, quantize_reduction)

        voxel_coords_float = quantized.to(dtype=torch.float32) * voxel_size

        atom_features_list.append(features)
        atom_labels_list.append(labels)
        atom_coords_list.append(coords)

        inverse_maps.append(inverse.to(dtype=torch.long) + voxel_offset)
        voxel_features_list.append(reduced_features)
        voxel_coords_list.append(voxel_coords_float)
        coord_list.append(quantized)

        splits.append(splits[-1] + num_atoms)
        voxel_offset += num_voxels

    batched_coords = ME.utils.batched_coordinates(coord_list)
    voxel_features = torch.cat(voxel_features_list, dim=0)
    sparse_inputs = ME.SparseTensor(features=voxel_features, coordinates=batched_coords)

    inverse_map_atom2voxel = torch.cat(inverse_maps, dim=0)
    atom_features = torch.cat(atom_features_list, dim=0)
    atom_labels = torch.cat(atom_labels_list, dim=0)
    atom_coords = torch.cat(atom_coords_list, dim=0)
    voxel_coords_float = torch.cat(voxel_coords_list, dim=0)
    batch_row_splits = torch.tensor(splits, dtype=torch.long)

    if inverse_map_atom2voxel.numel() == 0:
        raise ValueError("inverse_map_atom2voxel must not be empty")
    if inverse_map_atom2voxel.max().item() >= sparse_inputs.F.shape[0]:
        raise RuntimeError("Inverse map indices exceed sparse tensor feature rows")

    return Batch(
        sparse_inputs=sparse_inputs,
        inverse_map_atom2voxel=inverse_map_atom2voxel,
        atom_features=atom_features,
        atom_labels=atom_labels,
        atom_coords=atom_coords,
        voxel_coords_float=voxel_coords_float,
        batch_row_splits=batch_row_splits,
    )


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Set global random seeds for deterministic experiments."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.deterministic = deterministic_cudnn
        cudnn_backend.benchmark = not deterministic_cudnn
