"""Supervised fine-tuning script using noisy density-weighted labels."""

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
from .losses import DensityWeightedCrossEntropy


def _load_module(name: str, relative_path: Path):
    spec = util.spec_from_file_location(name, relative_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODELS = _load_module(
    "model_segmentation_unet",
    Path(__file__).resolve().parents[1] / "02_model_architecture" / "model_segmentation_unet.py",
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--encoder-weights", type=Path, default=Path("encoder_weights.pth"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("segmenter_final.pth"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = ProteinVoxelDataset(Path(config["processed_dataset_dir"]))
    loader = DataLoader(
        dataset,
        batch_size=config["segmentation"].get("batch_size", 2),
        shuffle=True,
        collate_fn=protein_collate_fn,
    )

    model = MODELS.SparseSegmentationUNet(input_channels=dataset[0].features.shape[1])
    model.to(args.device)
    if args.encoder_weights.exists():
        state_dict = torch.load(args.encoder_weights, map_location=args.device)
        model.load_encoder_weights(state_dict)
    criterion = DensityWeightedCrossEntropy(threshold=config["segmentation"].get("density_threshold", 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["segmentation"].get("learning_rate", 5e-4))

    for epoch in range(config["segmentation"].get("epochs", 1)):
        model.train()
        running_loss = 0.0
        for batch in loader:
            coords = batch["coords"].to(args.device)
            features = batch["features"].to(args.device)
            labels = batch["labels"].to(args.device)
            sparse_tensor = ME.SparseTensor(features, coordinates=coords).to(args.device)
            optimizer.zero_grad()
            logits = model(sparse_tensor)
            probed = logits.features_at_coordinates(coords)
            loss = criterion(probed, labels, features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
