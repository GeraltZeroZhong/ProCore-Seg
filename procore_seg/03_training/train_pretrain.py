"""Self-supervised pretraining script for the sparse autoencoder."""

from __future__ import annotations

import argparse
from importlib import util
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to load configuration files") from exc

try:
    import MinkowskiEngine as ME
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("MinkowskiEngine is required for training") from exc

from .dataset import ProteinVoxelDataset, protein_collate_fn
from .losses import DensityAwareChamferDistance


def _load_module(name: str, relative_path: Path):
    spec = util.spec_from_file_location(name, relative_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODELS = _load_module(
    "model_pretrain_autoencoder",
    Path(__file__).resolve().parents[1] / "02_model_architecture" / "model_pretrain_autoencoder.py",
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("encoder_weights.pth"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset_dir = Path(config["processed_dataset_dir"])
    dataset = ProteinVoxelDataset(dataset_dir)
    loader = DataLoader(
        dataset,
        batch_size=config["ssp"].get("batch_size", 4),
        shuffle=True,
        collate_fn=protein_collate_fn,
    )

    model = MODELS.SparseAutoencoder(input_channels=dataset[0].features.shape[1])
    model.to(args.device)
    criterion = DensityAwareChamferDistance()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["ssp"].get("learning_rate", 1e-3))

    for epoch in range(config["ssp"].get("epochs", 1)):
        model.train()
        running_loss = 0.0
        for batch in loader:
            coords = batch["coords"].to(args.device)
            features = batch["features"].to(args.device)
            original = batch["original_coords"].to(args.device)
            sparse_tensor = ME.SparseTensor(features, coordinates=coords).to(args.device)
            optimizer.zero_grad()
            reconstructed = model.reconstruct_coordinates(sparse_tensor)
            loss = criterion(reconstructed, original)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    torch.save(model.encoder.state_dict(), args.output)


if __name__ == "__main__":
    main()
