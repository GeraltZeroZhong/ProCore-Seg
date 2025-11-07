"""Inference helpers for the trained segmentation model."""

from __future__ import annotations

from importlib import util
from pathlib import Path

import numpy as np
import torch

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("MinkowskiEngine is required for inference") from exc


def _load_module(name: str, path: Path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
FEATURIZER = _load_module("atom_point_featurizer", ROOT / "01_data_curation" / "03_atom_point_featurizer.py")
DATASET = _load_module("training_dataset", ROOT / "03_training" / "dataset.py")
MODELS = _load_module("model_segmentation_unet", ROOT / "02_model_architecture" / "model_segmentation_unet.py")


def load_model(weights_path: Path, input_channels: int, device: torch.device):
    model = MODELS.SparseSegmentationUNet(input_channels=input_channels)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_labels(pdb_path: Path, weights_path: Path, device: str | torch.device = "cpu") -> np.ndarray:
    device = torch.device(device)
    data = FEATURIZER.featurize_structure(pdb_path, {})
    if data["coords"].size == 0:
        return np.array([], dtype=np.int64)
    model = load_model(weights_path, data["features"].shape[1], device)
    quantized = DATASET.quantize_coords(data["coords"], DATASET.VOXEL_SIZE)
    coords = ME.utils.batched_coordinates([quantized.to(device)])
    features = torch.from_numpy(data["features"]).float().to(device)
    sparse_tensor = ME.SparseTensor(features, coordinates=coords)
    logits = model(sparse_tensor)
    probed = logits.features_at_coordinates(coords)
    labels = probed.argmax(dim=1).cpu().numpy()
    return labels
