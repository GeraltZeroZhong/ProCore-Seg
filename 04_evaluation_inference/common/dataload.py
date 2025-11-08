from __future__ import annotations

"""Data discovery and loading utilities for evaluation/inference."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np

from .utils import read_ids_file, safe_glob


def _normalise_id(identifier: str) -> str:
    ident = identifier.strip()
    if not ident:
        raise ValueError("Empty identifier encountered")
    return ident.upper()


def discover_h5(data_dir: Path, ids_file: Optional[Path] = None) -> Dict[str, Path]:
    """Discover processed HDF5 files under ``data_dir``."""

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    id_to_path: Dict[str, Path] = {}
    if ids_file is not None:
        ids = read_ids_file(ids_file)
        for pdb_id in ids:
            candidate = data_dir / f"{pdb_id}.h5"
            if not candidate.exists():
                raise ValueError(f"Missing H5 file for ID {pdb_id}: {candidate}")
            id_to_path[pdb_id] = candidate
    else:
        matches = safe_glob(data_dir, "*.h5")
        for path in matches:
            pdb_id = _normalise_id(path.stem)
            id_to_path[pdb_id] = path
    return dict(sorted(id_to_path.items()))


def load_h5_minimal(h5_path: Path) -> Dict[str, np.ndarray]:
    """Load the minimal set of arrays required for evaluation."""

    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as handle:
        coords = np.asarray(handle["coords"], dtype=np.float32)
        features = np.asarray(handle["features"], dtype=np.float32)
        labels = np.asarray(handle["labels"])
        chain = np.asarray(handle["meta"]["chain_id"])
        resseq = np.asarray(handle["meta"]["resseq"])
        icode = np.asarray(handle["meta"]["icode"])

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError("features must have shape (N, 8)")
    if labels.shape[0] != coords.shape[0]:
        raise ValueError("labels length must match coords")
    chain = _decode_strings(chain).reshape(coords.shape[0])
    icode = _decode_strings(icode).reshape(coords.shape[0])
    if resseq.shape[0] != coords.shape[0]:
        raise ValueError("resseq length must match coords")
    arrays.update(
        {
            "coords": coords,
            "features": features,
            "labels": labels,
            "chain": chain,
            "resseq": np.asarray(resseq, dtype=np.int32).reshape(-1),
            "icode": icode,
        }
    )
    return arrays


def _decode_strings(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.kind in {"S", "O"}:
        decoded = np.vectorize(
            lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x),
            otypes=[object],
        )(arr)
        return decoded.astype(object)
    if arr.dtype.kind == "U":
        return arr.astype(object)
    raise ValueError("Expected string array for chain/icode")


def discover_npz(run_dir: Path) -> Dict[str, Path]:
    """Discover inference NPZ payloads under ``run_dir``."""

    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    matches = safe_glob(run_dir, "*.npz")
    mapping = {_normalise_id(path.stem): path for path in matches}
    return dict(sorted(mapping.items()))


def load_npz_payload(npz_path: Path) -> Dict[str, np.ndarray | dict | str]:
    """Load an inference NPZ payload with validation."""

    required_keys = [
        "probs_atom",
        "pred_atom",
        "labels_atom",
        "density_atom",
        "coords_atom",
        "chain",
        "resseq",
        "icode",
    ]
    payload: Dict[str, np.ndarray | dict | str] = {}
    warnings: List[str] = []
    with np.load(npz_path, allow_pickle=True) as data:
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in NPZ: {npz_path}")
            payload[key] = data[key]
        if "meta" in data:
            raw_meta = data["meta"]
            if isinstance(raw_meta, np.ndarray):
                meta = raw_meta.item()
            elif isinstance(raw_meta, np.void) and getattr(raw_meta, "dtype", None) and raw_meta.dtype.names:
                meta = {name: raw_meta[name] for name in raw_meta.dtype.names}
            else:
                meta = raw_meta
            if not isinstance(meta, dict):
                raise ValueError("meta entry must be a mapping")
        else:
            meta = {}
            warnings.append("meta missing")
        payload["meta"] = meta
        for optional in ("probs_voxel", "pred_voxel", "logits_voxel", "atom2voxel"):
            if optional in data:
                payload[optional] = data[optional]
    probs_atom = np.asarray(payload["probs_atom"])
    pred_atom = np.asarray(payload["pred_atom"])
    labels_atom = np.asarray(payload["labels_atom"])
    density_atom = np.asarray(payload["density_atom"])
    coords_atom = np.asarray(payload["coords_atom"])
    if probs_atom.ndim != 2 or probs_atom.shape[1] < 2:
        raise ValueError("probs_atom must have shape (N, C>=2)")
    n_atoms = probs_atom.shape[0]
    for name, arr in (
        ("pred_atom", pred_atom),
        ("labels_atom", labels_atom),
        ("density_atom", density_atom),
        ("coords_atom", coords_atom),
    ):
        if arr.shape[0] != n_atoms:
            raise ValueError(f"{name} must align with probs_atom length")
    pred_atom = np.asarray(pred_atom).reshape(n_atoms).astype(np.int64, copy=False)
    labels_atom = np.asarray(labels_atom).reshape(n_atoms).astype(np.int64, copy=False)
    density_atom = np.asarray(density_atom).reshape(n_atoms).astype(np.float32, copy=False)
    if coords_atom.ndim != 2 or coords_atom.shape[1] != 3:
        raise ValueError("coords_atom must have shape (N, 3)")
    coords_atom = np.asarray(coords_atom, dtype=np.float32)
    chain = _decode_strings(np.asarray(payload["chain"])).reshape(n_atoms)
    resseq = np.asarray(payload["resseq"], dtype=np.int32).reshape(n_atoms)
    icode = _decode_strings(np.asarray(payload["icode"])).reshape(n_atoms)
    if chain.shape[0] != n_atoms or resseq.shape[0] != n_atoms or icode.shape[0] != n_atoms:
        raise ValueError("chain/resseq/icode must align with probs_atom")
    meta_dict = payload["meta"]
    if "pdb_id" not in meta_dict:
        meta_dict["pdb_id"] = _normalise_id(npz_path.stem)
        warnings.append("meta missing pdb_id")
    if "voxel_size" not in meta_dict:
        meta_dict["voxel_size"] = float("nan")
        warnings.append("meta missing voxel_size")
    payload.update(
        {
            "probs_atom": probs_atom,
            "pred_atom": pred_atom,
            "labels_atom": labels_atom,
            "density_atom": density_atom,
            "coords_atom": coords_atom,
            "chain": chain,
            "resseq": resseq,
            "icode": icode,
        }
    )
    if warnings:
        payload["__warning__"] = "; ".join(sorted(set(warnings)))
    return payload


def load_folds_yaml(path: Optional[Path]) -> Dict[str, List[str]]:
    """Load a folds YAML mapping and normalise identifiers."""

    if path is None:
        return {}
    from .utils import load_yaml_or_none

    content = load_yaml_or_none(path)
    if not content:
        return {}
    folds: Dict[str, List[str]] = {}
    for key, value in content.items():
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError(f"Fold list for {key} must be an iterable of identifiers")
        deduped = sorted({_normalise_id(str(v)) for v in value})
        folds[key] = deduped
    return folds


def split_by_folds(
    all_ids: List[str],
    folds: Dict[str, List[str]] | None,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Split identifiers into train/val/test according to folds or random split."""

    norm_ids = [_normalise_id(i) for i in all_ids]
    unique_ids = list(dict.fromkeys(norm_ids))
    total_set = set(unique_ids)
    if folds:
        result: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for key, values in folds.items():
            intersection = sorted(total_set.intersection(_normalise_id(v) for v in values))
            if key in result:
                result[key] = intersection
            else:
                result[key] = intersection
        remaining = total_set.difference(*map(set, result.values()))
        result.setdefault("train", [])
        result["train"] = sorted(set(result["train"]) | remaining)
        for key in ("val", "test"):
            result.setdefault(key, [])
            result[key] = sorted(result[key])
        return result
    if not 0 <= val_ratio <= 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    shuffled.sort()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(np.floor(n * val_ratio))
    n_test = int(np.floor(n * val_ratio))
    n_train = max(0, n - n_val - n_test)
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val : n_train + n_val + n_test])
    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        npz_path = tmp_dir / "1abc.npz"
        with open(npz_path, "wb") as handle:
            np.savez(
                handle,
                probs_atom=np.ones((5, 3), dtype=np.float32),
                pred_atom=np.zeros(5, dtype=np.int64),
                labels_atom=np.zeros(5, dtype=np.int64),
                density_atom=np.zeros(5, dtype=np.float32),
                coords_atom=np.zeros((5, 3), dtype=np.float32),
                chain=np.array(["A"] * 5, dtype=object),
                resseq=np.arange(5),
                icode=np.array([""] * 5, dtype=object),
                meta={"pdb_id": "1ABC", "voxel_size": 1.5},
            )
        payload = load_npz_payload(npz_path)
        assert payload["probs_atom"].shape == (5, 3)
        print("NPZ payload loaded successfully.")
    finally:
        for child in tmp_dir.iterdir():
            child.unlink()
        tmp_dir.rmdir()
