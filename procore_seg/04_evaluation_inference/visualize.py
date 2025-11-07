"""Utilities to export segmentation outputs as PDB files for PyMOL."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import PDBIO, PDBParser


def write_segmentation_to_pdb(original_pdb: Path, labels: np.ndarray, output_path: Path) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(original_pdb.stem, original_pdb)
    index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == "H":
                        continue
                    if index >= len(labels):
                        break
                    atom.set_bfactor(float(labels[index]))
                    index += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("labels", type=Path, help="NumPy .npy file with predicted labels")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    labels = np.load(args.labels)
    output = args.output or args.pdb.with_suffix(".seg.pdb")
    write_segmentation_to_pdb(args.pdb, labels, output)


if __name__ == "__main__":
    main()
