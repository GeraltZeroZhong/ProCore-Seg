from __future__ import annotations

"""Entry-point utilities for running batched ProCore-Seg inference.

This module loads a :class:`SparseSegmentationUNet` checkpoint, performs sparse
voxel inference over processed HDF5 structures, and writes per-entry NPZ
artifacts containing both voxel- and atom-level predictions. The script is
 designed to be deterministic and reproducible across CPU and GPU devices.
"""

import argparse
import importlib
import logging
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

try:  # pragma: no cover - MinkowskiEngine is an optional dependency during linting
    import MinkowskiEngine as ME
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "MinkowskiEngine is required for inference. Please ensure it is installed."
    ) from exc


# Placeholder for potential dataset utilities; maintained for backward compatibility.
_unused_collate = None

from .postprocess import temperature_scale

try:
    _SEGMENTATION_MODULE = importlib.import_module(
        "procore_seg.02_model_architecture.model_segmentation_unet"
    )
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Unable to import SparseSegmentationUNet architecture. Ensure the repository layout is intact."
    ) from exc

SegmentationConfig = getattr(_SEGMENTATION_MODULE, "SegmentationConfig")
SparseSegmentationUNet = getattr(_SEGMENTATION_MODULE, "SparseSegmentationUNet")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalInferConfig:
    """Configuration for :func:`batch_infer` execution."""

    model_path: Path
    data_dir: Path
    ids_file: Optional[Path]
    out_dir: Path
    batch_size: int
    workers: int
    device: torch.device
    amp: bool
    temperature: float
    voxel_size: float
    checkpoint_voxel_size: Optional[float]
    log_level: str
    seed: int


# Global worker state populated via :func:`_worker_initializer`.
_WORKER_MODEL: Optional[nn.Module] = None
_WORKER_DEVICE: Optional[torch.device] = None
_WORKER_VOXEL_SIZE: Optional[float] = None
_WORKER_AMP: bool = False
_WORKER_TEMPERATURE: float = 1.0
_WORKER_CKPT_VOXEL_SIZE: Optional[float] = None
_WORKER_SEED: Optional[int] = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for deterministic behaviour."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _decode_attr(value: Any) -> str:
    """Return ``value`` decoded to ``str`` where possible."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return "".join(v.decode("utf-8") for v in value.tolist())
    return str(value)


def discover_ids(data_dir: Path, ids_file: Optional[Path]) -> List[Tuple[str, Path]]:
    """Return a sorted list of ``(pdb_id, h5_path)`` tuples to process."""

    if ids_file is not None:
        if not ids_file.exists():
            raise FileNotFoundError(f"IDs file not found: {ids_file}")
        with ids_file.open("r", encoding="utf-8") as handle:
            ids = [line.strip().upper() for line in handle if line.strip()]
        paths: List[Tuple[str, Path]] = []
        for pdb_id in ids:
            candidate = data_dir / f"{pdb_id}.h5"
            if not candidate.exists():
                LOGGER.warning("Skipping missing HDF5 for %s: %s", pdb_id, candidate)
                continue
            paths.append((pdb_id, candidate))
    else:
        paths = []
        for h5_path in sorted(data_dir.glob("*.h5")):
            pdb_id = h5_path.stem.upper()
            paths.append((pdb_id, h5_path))
    if not paths:
        raise RuntimeError("No HDF5 files discovered for inference.")
    return paths


def _iter_batches(entries: Sequence[Tuple[str, Path]], batch_size: int) -> Iterator[Sequence[Tuple[str, Path]]]:
    """Yield ``entries`` in fixed-size batches preserving order."""

    batch_size = max(1, int(batch_size))
    for start in range(0, len(entries), batch_size):
        yield entries[start : start + batch_size]


def load_checkpoint(model_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load a serialized checkpoint from ``model_path``."""

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint file {model_path} is not a dictionary")
    model_state = state.get("model_state")
    if model_state is None:
        raise KeyError("Checkpoint missing 'model_state'")
    if not isinstance(model_state, dict):
        raise TypeError("checkpoint['model_state'] must be a dict")
    meta = state.get("meta", {})
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise TypeError("checkpoint['meta'] must be a dict if present")
    model_state_cpu = {k: v.cpu() for k, v in model_state.items()}
    return model_state_cpu, meta


def build_model_from_meta(meta: Dict[str, Any]) -> SparseSegmentationUNet:
    """Instantiate :class:`SparseSegmentationUNet` from metadata."""

    model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}
    in_channels = int(model_meta.get("in_channels", 8))
    base_channels = int(model_meta.get("base_channels", 32))
    depth = int(model_meta.get("depth", 4))
    num_classes = int(model_meta.get("num_classes", 2))
    seg_cfg = SegmentationConfig(
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
        num_classes=num_classes,
    )
    return SparseSegmentationUNet(seg_cfg)


def _decode_string_array(array: np.ndarray) -> np.ndarray:
    """Convert byte-string arrays to Unicode NumPy arrays."""

    if array.dtype.kind == "S":
        max_len = array.dtype.itemsize
        decoded = np.char.decode(array, "utf-8")
        return decoded.astype(f"<U{max_len}")
    return array


def _load_metadata_arrays(meta_group: h5py.Group) -> Dict[str, np.ndarray]:
    """Load relevant metadata arrays from ``meta_group``."""

    metadata: Dict[str, np.ndarray] = {}
    for key in ("chain", "resseq", "icode", "atom_name", "element"):
        if key in meta_group:
            value = meta_group[key][()]
            if isinstance(value, np.ndarray):
                value = _decode_string_array(value)
            if key == "resseq":
                metadata[key] = np.asarray(value, dtype=np.int32)
            else:
                metadata[key] = np.asarray(value)
    return metadata


def _ensure_numpy_float(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.float32:
        return array.astype(np.float32)
    return array


def infer_one(
    h5_path: Path,
    model: nn.Module,
    voxel_size: float,
    device: torch.device,
    amp: bool,
    temperature: float,
    checkpoint_voxel_size: Optional[float],
) -> Dict[str, Any]:
    """Run inference for a single processed HDF5 entry."""

    start_time = time.perf_counter()
    with h5py.File(h5_path, "r") as handle:
        coords = handle["coords"][()]
        features = handle["features"][()]
        labels = handle["labels"][()]
        labels = np.asarray(labels).reshape(-1)
        meta_group = handle.get("meta")
        metadata_arrays = _load_metadata_arrays(meta_group) if meta_group else {}
        pdb_id = _decode_attr(handle.attrs.get("pdb_id", h5_path.stem))
        cath_id = _decode_attr(handle.attrs.get("cath_id", "")) if "cath_id" in handle.attrs else ""

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords dataset in {h5_path} must have shape (N, 3)")
    if features.ndim != 2:
        raise ValueError(f"features dataset in {h5_path} must have shape (N, F)")
    if labels.shape[0] != coords.shape[0]:
        raise ValueError("labels and coords must have matching first dimension")
    for key, array in metadata_arrays.items():
        if isinstance(array, np.ndarray) and array.shape and array.shape[0] != coords.shape[0]:
            raise ValueError(f"Metadata array '{key}' length mismatch for {h5_path}")

    coords_tensor = torch.from_numpy(coords.astype(np.float32))
    int_coords = torch.floor(coords_tensor / voxel_size).to(torch.int32)
    _, uq_idx, inverse = ME.utils.sparse_quantize(
        int_coords,
        return_index=True,
        return_inverse=True,
    )
    inverse = inverse.to(torch.long)
    if uq_idx.numel() == 0:
        raise RuntimeError(f"No voxels generated for {h5_path}")

    features_tensor = torch.from_numpy(features.astype(np.float32))
    num_voxels = int(uq_idx.numel())
    feat_dim = features_tensor.shape[1]
    voxel_feats = torch.zeros(num_voxels, feat_dim, dtype=torch.float32)
    voxel_feats.index_add_(0, inverse, features_tensor)
    counts = torch.zeros(num_voxels, dtype=torch.float32)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    counts = torch.clamp(counts, min=1.0)
    voxel_feats = voxel_feats / counts.unsqueeze(1)

    coords_unique = int_coords[uq_idx]
    batched_coords = ME.utils.batched_coordinates([coords_unique])

    voxel_feats = voxel_feats.to(device)
    batched_coords = batched_coords.to(device)

    sparse_tensor = ME.SparseTensor(features=voxel_feats, coordinates=batched_coords, device=device)

    amp_enabled = amp and device.type == "cuda"
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits_sparse = model(sparse_tensor)
    logits_voxel = logits_sparse.F
    logits_voxel = temperature_scale(logits_voxel, temperature)
    probs_voxel = torch.softmax(logits_voxel, dim=1)
    pred_voxel = probs_voxel.argmax(dim=1)

    inverse_device = inverse.to(device)
    probs_atom = probs_voxel[inverse_device]
    pred_atom = pred_voxel[inverse_device]

    density_atom = (
        features_tensor[:, 7].cpu().numpy()
        if features_tensor.shape[1] > 7
        else np.zeros(coords.shape[0], dtype=np.float32)
    )

    payload: Dict[str, Any] = {
        "logits_voxel": logits_voxel.cpu().numpy(),
        "probs_voxel": probs_voxel.cpu().numpy(),
        "pred_voxel": pred_voxel.cpu().numpy(),
        "atom2voxel": inverse.cpu().numpy(),
        "probs_atom": probs_atom.cpu().numpy(),
        "pred_atom": pred_atom.cpu().numpy(),
        "labels_atom": labels.astype(np.int64),
        "density_atom": density_atom.astype(np.float32),
        "coords_atom": coords.astype(np.float32),
        "meta": {
            "pdb_id": pdb_id,
            "cath_id": cath_id,
            "voxel_size": float(voxel_size),
            "num_voxels": num_voxels,
            "num_atoms": int(coords.shape[0]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_voxel_size": float(checkpoint_voxel_size) if checkpoint_voxel_size is not None else None,
            "temperature": float(temperature),
        },
    }

    for key, value in metadata_arrays.items():
        payload[key] = value

    elapsed = time.perf_counter() - start_time
    LOGGER.info(
        "%s | atoms=%d voxels=%d device=%s amp=%s time=%.3fs",
        h5_path.stem,
        coords.shape[0],
        num_voxels,
        device.type,
        str(amp_enabled),
        elapsed,
    )
    return payload


def _worker_initializer(
    state_dict: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    voxel_size: float,
    device_str: str,
    amp: bool,
    temperature: float,
    checkpoint_voxel_size: Optional[float],
    seed: int,
) -> None:
    """Initialise global model state for worker processes."""

    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_VOXEL_SIZE, _WORKER_AMP, _WORKER_TEMPERATURE, _WORKER_CKPT_VOXEL_SIZE, _WORKER_SEED
    set_seed(seed)
    device = torch.device(device_str)
    model = build_model_from_meta(meta)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _WORKER_MODEL = model
    _WORKER_DEVICE = device
    _WORKER_VOXEL_SIZE = voxel_size
    _WORKER_AMP = amp and device.type == "cuda"
    _WORKER_TEMPERATURE = temperature
    _WORKER_CKPT_VOXEL_SIZE = checkpoint_voxel_size
    _WORKER_SEED = seed


def _worker_infer(args: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    pdb_id, path_str = args
    if _WORKER_MODEL is None or _WORKER_DEVICE is None or _WORKER_VOXEL_SIZE is None:
        raise RuntimeError("Worker model is not initialised")
    payload = infer_one(
        Path(path_str),
        _WORKER_MODEL,
        _WORKER_VOXEL_SIZE,
        _WORKER_DEVICE,
        _WORKER_AMP,
        _WORKER_TEMPERATURE,
        _WORKER_CKPT_VOXEL_SIZE,
    )
    return pdb_id, payload


def _write_npz_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=tmp_suffix) as tmp_file:
        np.savez_compressed(tmp_file, **payload)
        temp_name = Path(tmp_file.name)
    os.replace(temp_name, path)


def batch_infer(entries: Sequence[Tuple[str, Path]], cfg: EvalInferConfig, model: nn.Module, meta: Dict[str, Any]) -> None:
    """Run inference over ``entries`` using ``model`` and persist outputs."""

    if not entries:
        LOGGER.warning("No entries to process.")
        return

    device = cfg.device
    use_amp = cfg.amp and device.type == "cuda"
    model = model.to(device)
    model.eval()

    if device.type == "cuda":
        if cfg.workers != 1:
            LOGGER.warning("CUDA device in use; forcing workers=1 to avoid contention")
        iterator = tqdm(total=len(entries), desc="Inference", unit="entry")
        for batch in _iter_batches(entries, cfg.batch_size):
            for pdb_id, path in batch:
                payload = infer_one(
                    path,
                    model,
                    cfg.voxel_size,
                    device,
                    use_amp,
                    cfg.temperature,
                    cfg.checkpoint_voxel_size,
                )
                output_path = cfg.out_dir / f"{pdb_id}.npz"
                _write_npz_atomic(output_path, payload)
                iterator.update(1)
        iterator.close()
    else:
        if cfg.workers <= 1:
            iterator = tqdm(total=len(entries), desc="Inference", unit="entry")
            for batch in _iter_batches(entries, cfg.batch_size):
                for pdb_id, path in batch:
                    payload = infer_one(
                        path,
                        model,
                        cfg.voxel_size,
                        device,
                        use_amp,
                        cfg.temperature,
                        cfg.checkpoint_voxel_size,
                    )
                    output_path = cfg.out_dir / f"{pdb_id}.npz"
                    _write_npz_atomic(output_path, payload)
                    iterator.update(1)
            iterator.close()
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            init_args = (
                state_dict,
                meta,
                cfg.voxel_size,
                "cpu",
                False,
                cfg.temperature,
                cfg.checkpoint_voxel_size,
                cfg.seed,
            )
            iterator = tqdm(total=len(entries), desc="Inference", unit="entry")
            with ProcessPoolExecutor(max_workers=cfg.workers, initializer=_worker_initializer, initargs=init_args) as executor:
                futures = {
                    executor.submit(_worker_infer, (pdb_id, str(path))): pdb_id
                    for pdb_id, path in entries
                }
                for future in as_completed(futures):
                    pdb_id, payload = future.result()
                    output_path = cfg.out_dir / f"{pdb_id}.npz"
                    _write_npz_atomic(output_path, payload)
                    iterator.update(1)
            iterator.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProCore-Seg inference on processed structures.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the segmentation checkpoint (.pth)")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing processed HDF5 files")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional text file listing PDB IDs to process")
    parser.add_argument("--out-dir", required=True, type=Path, help="Destination directory for NPZ outputs")
    parser.add_argument("--batch", type=int, default=1, help="Number of entries to process per step (default: 1)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for CPU inference")
    parser.add_argument("--device", type=str, default="cpu", help="Target device identifier (cpu or cuda:X)")
    parser.add_argument("--amp", type=str, default="true", help="Enable mixed precision on CUDA (true/false)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for logits")
    parser.add_argument("--voxel-size", type=str, default="", help="Optional override for voxel size (Angstrom)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic execution")
    return parser.parse_args(argv)


def _parse_bool(value: str) -> bool:
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "yes", "y"}:
        return True
    if value_lower in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    set_seed(args.seed)

    amp_enabled = _parse_bool(args.amp)
    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid device string: {args.device}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
        amp_enabled = False
    if device.type != "cuda":
        amp_enabled = False

    model_state, meta = load_checkpoint(args.model)
    checkpoint_voxel_size = meta.get("voxel_size") if isinstance(meta, dict) else None
    if checkpoint_voxel_size is None:
        raise KeyError("Checkpoint metadata missing 'voxel_size'")
    checkpoint_voxel_size = float(checkpoint_voxel_size)

    voxel_override = args.voxel_size.strip()
    if voxel_override:
        try:
            voxel_size = float(voxel_override)
        except ValueError as exc:
            raise ValueError("--voxel-size must be a floating point number") from exc
        if not math.isclose(voxel_size, checkpoint_voxel_size, rel_tol=1e-5, abs_tol=1e-5):
            LOGGER.warning(
                "Using voxel size override %.4f (checkpoint reports %.4f)",
                voxel_size,
                checkpoint_voxel_size,
            )
    else:
        voxel_size = checkpoint_voxel_size

    model = build_model_from_meta(meta)
    model.load_state_dict(model_state)

    try:
        entries = discover_ids(args.data_dir, args.ids_file)
    except Exception as exc:
        LOGGER.error("Failed to discover HDF5 entries: %s", exc)
        return 1

    cfg = EvalInferConfig(
        model_path=args.model,
        data_dir=args.data_dir,
        ids_file=args.ids_file,
        out_dir=args.out_dir,
        batch_size=max(1, args.batch),
        workers=max(1, args.workers),
        device=device,
        amp=amp_enabled,
        temperature=float(args.temperature),
        voxel_size=voxel_size,
        checkpoint_voxel_size=checkpoint_voxel_size,
        log_level=args.log_level.upper(),
        seed=args.seed,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        batch_infer(entries, cfg, model, meta)
    except Exception as exc:
        LOGGER.exception("Inference failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
