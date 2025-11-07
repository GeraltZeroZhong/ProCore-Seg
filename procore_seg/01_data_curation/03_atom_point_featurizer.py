"""Convert PDB/mmCIF structures into feature rich atomic point clouds."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

FIBOS_AVAILABLE = importlib.util.find_spec("fibos") is not None
if FIBOS_AVAILABLE:  # pragma: no cover - optional dependency
    import fibos

ATOM_TYPES = ("C", "H", "O", "N", "S")


def _parse_structure(pdb_path: Path):
    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, pdb_path)


def _one_hot_element(element: str) -> np.ndarray:
    vector = np.zeros(len(ATOM_TYPES) + 1, dtype=np.float32)
    element = element.capitalize()
    if element in ATOM_TYPES:
        index = ATOM_TYPES.index(element)
        vector[index] = 1.0
    else:
        vector[-1] = 1.0
    return vector


def _compute_sasa(structure) -> Dict[Tuple[str, int], float]:
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    sasa: Dict[Tuple[str, int], float] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue
                sasa[(chain.id, residue.id[1])] = float(residue.xtra.get("EXP_SASA", 0.0))
    return sasa


def _compute_density(pdb_path: Path) -> Dict[Tuple[str, int], float]:
    if not FIBOS_AVAILABLE:  # pragma: no cover - optional dependency
        return {}
    result = fibos.osp(str(pdb_path))
    density: Dict[Tuple[str, int], float] = {}
    for chain_id, residues in result.items():
        for res_seq, value in residues.items():
            density[(chain_id, int(res_seq))] = float(value)
    return density


def featurize_structure(pdb_path: Path, label_map: Dict[Tuple[str, int], int]) -> Dict[str, np.ndarray]:
    structure = _parse_structure(pdb_path)
    sasa_map = _compute_sasa(structure)
    density_map = _compute_density(pdb_path)

    coords: List[np.ndarray] = []
    features: List[np.ndarray] = []
    labels: List[int] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " " or not is_aa(residue, standard=False):
                    continue
                chain_id = chain.id
                res_seq = residue.id[1]
                sasa = sasa_map.get((chain_id, res_seq), 0.0)
                density = density_map.get((chain_id, res_seq), 0.0)
                label = int(label_map.get((chain_id, res_seq), 0))
                for atom in residue:
                    if atom.element == "H":
                        continue
                    coords.append(atom.coord.astype(np.float32))
                    atom_features = np.concatenate(
                        [
                            _one_hot_element(atom.element),
                            np.array([sasa, density], dtype=np.float32),
                        ]
                    )
                    features.append(atom_features)
                    labels.append(label)

    return {
        "coords": np.vstack(coords) if coords else np.empty((0, 3), dtype=np.float32),
        "features": np.vstack(features) if features else np.empty((0, len(ATOM_TYPES) + 1 + 2), dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_path", type=Path)
    parser.add_argument("label_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    label_data = json.loads(args.label_json.read_text())
    residues = {
        tuple(key.split(":")): value for key, value in label_data.get("residues", {}).items()
    }
    residues_typed = {(chain, int(seq)): int(label) for (chain, seq), label in residues.items()}
    result = featurize_structure(args.pdb_path, residues_typed)

    if args.output:
        np.savez(args.output, **result)
    else:
        print({key: value.shape for key, value in result.items()})


if __name__ == "__main__":
    main()
