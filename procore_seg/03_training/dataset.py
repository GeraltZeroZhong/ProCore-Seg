"""Dataset and collation utilities for sparse protein point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("MinkowskiEngine is required for the training dataset") from exc

VOXEL_SIZE = 1.5


@dataclass(slots=True)
class ProteinSample:
    coords: np.ndarray
    features: np.ndarray
    labels: np.ndarray


class ProteinVoxelDataset(Dataset[ProteinSample]):
    """Load processed HDF5 files and expose them as MinkowskiEngine tensors."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Processed dataset directory not found: {self.root}")
        self.files = sorted(self.root.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No processed files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> ProteinSample:
        file_path = self.files[index]
        with h5py.File(file_path, "r") as handle:
            coords = handle["coords"][()]
            features = handle["features"][()]
            labels = handle["labels"][()]
        return ProteinSample(coords=coords, features=features, labels=labels)


def quantize_coords(coords: np.ndarray, voxel_size: float = VOXEL_SIZE) -> torch.IntTensor:
    scaled = torch.from_numpy(coords).float() / voxel_size
    return torch.floor(scaled).int()


def protein_collate_fn(batch: Sequence[ProteinSample], voxel_size: float = VOXEL_SIZE):
    coords_list: List[torch.IntTensor] = []
    feats_list: List[torch.FloatTensor] = []
    labels_list: List[torch.LongTensor] = []
    original_coords: List[torch.FloatTensor] = []
    for sample in batch:
        coords_list.append(quantize_coords(sample.coords, voxel_size))
        feats_list.append(torch.from_numpy(sample.features).float())
        labels_list.append(torch.from_numpy(sample.labels).long())
        original_coords.append(torch.from_numpy(sample.coords).float())
    coords, features = ME.utils.sparse_collate(coords_list, feats_list)
    labels = torch.cat(labels_list, dim=0)
    return {
        "coords": coords,
        "features": features,
        "labels": labels,
        "original_coords": torch.cat(original_coords, dim=0),
        "batch_indices": torch.tensor(
            [i for i, sample in enumerate(batch) for _ in range(len(sample.labels))], dtype=torch.long
        ),
    }
