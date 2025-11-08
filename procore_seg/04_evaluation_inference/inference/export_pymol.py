from __future__ import annotations

"""Export predictions to PyMOL-friendly PDB/PML artefacts."""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

LOGGER = logging.getLogger(__name__)


def _decode_array(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype.kind == "S":
        max_len = array.dtype.itemsize
        decoded = np.char.decode(array, "utf-8")
        return decoded.astype(f"<U{max_len}")
    return array


def _read_h5_metadata(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as handle:
        coords = handle["coords"][()]
        meta_group = handle.get("meta")
        if meta_group is None:
            raise KeyError("HDF5 file missing 'meta' group")
        metadata = {}
        for key in ("chain", "resseq", "icode", "atom_name", "element"):
            if key not in meta_group:
                raise KeyError(f"HDF5 meta group missing '{key}' dataset")
            metadata[key] = _decode_array(meta_group[key][()])
        metadata["coords"] = coords.astype(np.float32)
        metadata["num_atoms"] = coords.shape[0]
    return metadata


def _load_predictions(npz_path: Path, mode: str, class_index: int, expected_atoms: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        if mode == "prob":
            probs = data["probs_atom"]
            if probs.ndim != 2:
                raise ValueError("probs_atom must have shape (N, C)")
            if class_index < 0 or class_index >= probs.shape[1]:
                raise ValueError("class_index out of bounds for probs_atom")
            values = probs[:, class_index]
        elif mode == "label":
            values = data["pred_atom"].astype(np.float32)
        else:
            raise ValueError("mode must be 'prob' or 'label'")
    if values.shape[0] != expected_atoms:
        raise RuntimeError("Prediction and HDF5 atom counts differ")
    return values.astype(np.float32)


def _format_atom_line(index: int, name: str, chain: str, resseq: int, icode: str, coords: Sequence[float], element: str, bfactor: float) -> str:
    res_name = "UNK"
    occupancy = 1.00
    return (
        f"ATOM  {index:5d} {name:>4s}  {res_name} {chain:1s}{resseq:4d}{icode:1s}   "
        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}"
    )


def _write_atomic(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        for line in lines:
            tmp.write(f"{line}\n")
        tmp.write("END\n")
        temp_name = Path(tmp.name)
    os.replace(temp_name, path)


def _write_pml(pdb_path: Path) -> None:
    pml_path = pdb_path.with_suffix(".pml")
    commands = [
        f"load {pdb_path.name}",
        "spectrum b, white_blue, minimum=0, maximum=1",
        "show cartoon",
        "util.cbaw",
    ]
    _write_atomic(pml_path, commands)


def export_pdb(h5_path: Path, npz_path: Path, out_path: Path, mode: str, class_index: int) -> None:
    metadata = _read_h5_metadata(h5_path)
    values = _load_predictions(npz_path, mode, class_index, metadata["num_atoms"])
    coords = metadata["coords"]
    chains = metadata["chain"]
    resseqs = metadata["resseq"].astype(int)
    icodes = metadata["icode"]
    names = metadata["atom_name"]
    elements = metadata["element"]

    lines = []
    for idx in range(metadata["num_atoms"]):
        line = _format_atom_line(
            index=idx + 1,
            name=str(names[idx]),
            chain=str(chains[idx])[:1],
            resseq=int(resseqs[idx]),
            icode=str(icodes[idx])[:1] if str(icodes[idx]) != "" else " ",
            coords=coords[idx],
            element=str(elements[idx])[:2].rjust(2),
            bfactor=float(values[idx]),
        )
        lines.append(line)

    _write_atomic(out_path, lines)
    _write_pml(out_path)
    LOGGER.info("Wrote %s and companion PML", out_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyMOL-coloured PDB from inference artefacts")
    parser.add_argument("--h5", required=True, type=Path, help="Processed HDF5 file")
    parser.add_argument("--npz", required=True, type=Path, help="Inference NPZ file")
    parser.add_argument("--out", required=True, type=Path, help="Output PDB path")
    parser.add_argument("--mode", type=str, default="prob", choices=["prob", "label"], help="Colouring mode")
    parser.add_argument("--class-index", type=int, default=1, help="Probability class index for mode=prob")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        export_pdb(args.h5, args.npz, args.out, args.mode, args.class_index)
    except Exception as exc:
        LOGGER.error("Export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
