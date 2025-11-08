from __future__ import annotations

"""Self-supervised Stage-1 training for the ProCore-Seg sparse autoencoder."""

import argparse
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Batch, ProteinVoxelDataset, collate_sparse_batch, set_seed
from .losses import DensityAwareChamferDistance


LOGGER = logging.getLogger("procore_seg.pretrain")


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean")


def setup_logging() -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = [handler]


_PACKAGE_NAME = "procore_seg_internal.model_architecture"


def _json_safe(value):
    """Return a JSON-serialisable representation of the provided value."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def build_checkpoint_meta(voxel_size: float, cfg: dict) -> dict:
    """Construct a metadata dictionary for persisted checkpoints."""

    import json
    import platform

    import torch

    try:  # pragma: no cover - optional dependency
        minkowski_version = getattr(__import__("MinkowskiEngine"), "__version__", "unknown")
    except ImportError:  # pragma: no cover - MinkowskiEngine optional
        minkowski_version = "unknown"

    sanitized_cfg = _json_safe(cfg)
    # Ensure serialisability by round-tripping through JSON where possible.
    try:
        sanitized_cfg = json.loads(json.dumps(sanitized_cfg))
    except TypeError:  # pragma: no cover - defensive fallback
        sanitized_cfg = _json_safe({key: str(value) for key, value in cfg.items()})

    return {
        "voxel_size": float(voxel_size),
        "model": {
            "in_channels": cfg.get("in_channels", 8),
            "base_channels": cfg.get("base_channels", 32),
            "depth": cfg.get("depth", 4),
            "arch": "SparseAutoencoder",
        },
        "train_config": sanitized_cfg,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "cpu",
            "minkowski": minkowski_version,
        },
    }


def _ensure_architecture_package() -> Tuple[Path, ModuleType]:
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "02_model_architecture"
    if not module_dir.exists():
        raise FileNotFoundError("02_model_architecture directory not found relative to script")

    package = sys.modules.get(_PACKAGE_NAME)
    if package is None:
        package = ModuleType(_PACKAGE_NAME)
        package.__path__ = [str(module_dir)]  # type: ignore[attr-defined]
        sys.modules[_PACKAGE_NAME] = package
    return module_dir, package


def _load_architecture_module(name: str) -> ModuleType:
    module_dir, package = _ensure_architecture_package()
    full_name = f"{_PACKAGE_NAME}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    file_path = module_dir / f"{name}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Expected architecture module at '{file_path}'")

    import importlib.util

    spec = importlib.util.spec_from_file_location(full_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create spec for module '{full_name}'")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PACKAGE_NAME
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class AverageMeter:
    value: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value += val * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / self.count


def build_dataloaders(
    data_dir: Path,
    ids_file: Optional[Path],
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    voxel_size: float,
) -> Tuple[DataLoader[Batch], Optional[DataLoader[Batch]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")

    ids: Optional[List[str]] = None
    if ids_file is not None and ids_file.is_file():
        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
        if not ids:
            raise ValueError("ids_file provided but contains no valid identifiers")

    dataset = ProteinVoxelDataset(data_dir, ids=ids)
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty")

    val_size = 0
    if val_split > 0:
        val_size = max(1, int(round(total * val_split)))
        if val_size >= total:
            val_size = total - 1
        if val_size <= 0:
            raise ValueError("val_split is too small for the dataset size")

    generator = torch.Generator()
    generator.manual_seed(seed)

    if val_size > 0:
        train_size = total - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_subset = dataset
        val_subset = None

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_sparse_batch(batch, voxel_size=voxel_size),
        generator=generator,
        persistent_workers=num_workers > 0,
    )

    val_loader: Optional[DataLoader[Batch]] = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: collate_sparse_batch(batch, voxel_size=voxel_size),
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader


def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        torch.save(payload, tmp.name)
        temp_name = Path(tmp.name)
    os.replace(temp_name, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[_LRScheduler],
    scaler: Optional[GradScaler],
) -> Tuple[int, float, Optional[Dict[str, object]]]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_metric = float(checkpoint.get("best_metric", math.inf))
    meta = checkpoint.get("meta") if isinstance(checkpoint.get("meta"), dict) else None
    return start_epoch, best_metric, meta


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Batch],
    criterion: DensityAwareChamferDistance,
    optimizer: Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_max_norm: float,
    log_interval: int,
) -> float:
    model.train()
    running = AverageMeter()

    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc="train", leave=False)
    for step, batch in progress:
        sparse_inputs = batch.sparse_inputs.to(device)
        atom_coords = batch.atom_coords.to(device)
        atom_features = batch.atom_features.to(device)
        inverse_map = batch.inverse_map_atom2voxel.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            outputs = model(sparse_inputs)
            pred_points = outputs["points"]
            density = atom_features[:, 7]
            threshold = torch.median(density)
            density_tau = 1.0
            gt_weights = torch.sigmoid((density - threshold) / density_tau)

            num_voxels = pred_points.size(0)
            voxel_density = torch.zeros(num_voxels, device=device, dtype=pred_points.dtype)
            voxel_density.scatter_add_(0, inverse_map, density)
            counts = torch.bincount(inverse_map, minlength=num_voxels).to(device=device, dtype=pred_points.dtype)
            counts = counts.clamp_min(1)
            mean_density = voxel_density / counts
            pred_weights = torch.sigmoid((mean_density - threshold) / density_tau)

            loss = criterion(pred_points, atom_coords, pred_weights, gt_weights)

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_max_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
            optimizer.step()

        running.update(loss.item())
        if log_interval > 0 and step % log_interval == 0:
            LOGGER.info("step=%d loss=%.6f", step, loss.item())
        progress.set_postfix(loss=running.average)

    return running.average


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Batch],
    criterion: DensityAwareChamferDistance,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    running = AverageMeter()
    with torch.no_grad():
        progress = tqdm(dataloader, desc="val", leave=False)
        for batch in progress:
            sparse_inputs = batch.sparse_inputs.to(device)
            atom_coords = batch.atom_coords.to(device)
            atom_features = batch.atom_features.to(device)
            inverse_map = batch.inverse_map_atom2voxel.to(device)

            with autocast(enabled=amp_enabled):
                outputs = model(sparse_inputs)
                pred_points = outputs["points"]
                density = atom_features[:, 7]
                threshold = torch.median(density)
                density_tau = 1.0
                gt_weights = torch.sigmoid((density - threshold) / density_tau)

                num_voxels = pred_points.size(0)
                voxel_density = torch.zeros(num_voxels, device=device, dtype=pred_points.dtype)
                voxel_density.scatter_add_(0, inverse_map, density)
                counts = torch.bincount(inverse_map, minlength=num_voxels).to(
                    device=device, dtype=pred_points.dtype
                )
                counts = counts.clamp_min(1)
                mean_density = voxel_density / counts
                pred_weights = torch.sigmoid((mean_density - threshold) / density_tau)

                loss = criterion(pred_points, atom_coords, pred_weights, gt_weights)

            running.update(loss.item())
            progress.set_postfix(loss=running.average)
    return running.average


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1 self-supervised pretraining")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--grad-max-norm", type=float, default=1.0)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp) and device.type == "cuda"

    train_loader, val_loader = build_dataloaders(
        args.data_dir,
        args.ids_file,
        args.batch_size,
        args.val_split,
        args.num_workers,
        args.seed,
        args.voxel_size,
    )

    autoencoder_module = _load_architecture_module("model_pretrain_autoencoder")
    SparseAutoencoder = getattr(autoencoder_module, "SparseAutoencoder")
    AutoencoderConfig = getattr(autoencoder_module, "AutoencoderConfig")
    cfg = AutoencoderConfig(quantization_size=args.voxel_size)
    model: nn.Module = SparseAutoencoder(cfg).to(device)

    criterion = DensityAwareChamferDistance(chunk_size=args.chunk_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler: Optional[_LRScheduler] = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_val = math.inf
    if args.resume is not None and args.resume.is_file():
        LOGGER.info("Resuming from checkpoint %s", args.resume)
        start_epoch, best_val, resume_meta = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        if isinstance(resume_meta, dict):
            meta_voxel = resume_meta.get("voxel_size")
            if meta_voxel is not None:
                try:
                    meta_voxel_float = float(meta_voxel)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    LOGGER.warning("Checkpoint voxel_size metadata malformed: %s", meta_voxel)
                else:
                    LOGGER.info("Checkpoint voxel_size=%.4f", meta_voxel_float)
                    if not math.isclose(
                        meta_voxel_float,
                        float(args.voxel_size),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        LOGGER.warning(
                            "Checkpoint voxel_size %.4f differs from requested %.4f",
                            meta_voxel_float,
                            args.voxel_size,
                        )

    args_dict = vars(args)
    args_dict_serialisable = {k: (str(v) if isinstance(v, Path) else v) for k, v in args_dict.items()}

    for epoch in range(start_epoch, args.epochs):
        LOGGER.info("Epoch %d/%d", epoch + 1, args.epochs)
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.grad_max_norm,
            args.log_interval,
        )
        LOGGER.info("Train loss: %.6f", train_loss)

        val_loss = math.inf
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device, amp_enabled)
            LOGGER.info("Val loss: %.6f", val_loss)

        if scheduler is not None:
            scheduler.step()

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "best_metric": best_val,
            "config": args_dict_serialisable,
            "meta": build_checkpoint_meta(args.voxel_size, vars(args)),
        }
        save_checkpoint(args.checkpoint_dir / "last.pth", checkpoint_payload)

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            checkpoint_payload["best_metric"] = best_val
            save_checkpoint(args.checkpoint_dir / "best.pth", checkpoint_payload)
            encoder_state_dict = model.get_encoder_state_dict()
            encoder_state = {
                "epoch": epoch,
                "encoder_state": encoder_state_dict,
                "state_dict": encoder_state_dict,
                "config": args_dict_serialisable,
                "meta": build_checkpoint_meta(args.voxel_size, vars(args)),
            }
            save_checkpoint(args.checkpoint_dir / "encoder_only.pth", encoder_state)
        elif val_loader is None and epoch == args.epochs - 1:
            save_checkpoint(args.checkpoint_dir / "best.pth", checkpoint_payload)
            encoder_state_dict = model.get_encoder_state_dict()
            encoder_state = {
                "epoch": epoch,
                "encoder_state": encoder_state_dict,
                "state_dict": encoder_state_dict,
                "config": args_dict_serialisable,
                "meta": build_checkpoint_meta(args.voxel_size, vars(args)),
            }
            save_checkpoint(args.checkpoint_dir / "encoder_only.pth", encoder_state)

    LOGGER.info("Training completed. Best val loss: %.6f", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
