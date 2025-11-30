"""Atom-level point featurizer for macromolecular structures."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
from Bio.PDB.SASA import ShrakeRupley

LOGGER = logging.getLogger(__name__)

ELEMENT_ORDER: Tuple[str, ...] = ("C", "H", "O", "N", "S", "OTHER")
ELEMENT_INDEX = {name: idx for idx, name in enumerate(ELEMENT_ORDER)}
ALTLOC_EPS = 1e-6
_ALLOWED_RESIDUE_FLAGS: Set[str] = {" ", "H_MSE", "H_SEM"}


class StructureParsingError(RuntimeError):
    """Raised when the structure cannot be parsed."""


def load_labels_json(path: Path) -> Set[Tuple[str, int, str]]:
    """Load the residue label mapping produced by ``02_sifts_label_mapper.py``.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Set[Tuple[str, int, str]]
        Set of ``(chain_id, resseq, icode)`` tuples that correspond to in-domain
        residues.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the JSON payload is malformed or missing ``entries``.
    """

    if not path.exists():
        raise FileNotFoundError(f"Label map file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed input
            raise RuntimeError(f"Failed to parse label JSON: {exc}") from exc

    entries = payload.get("entries")
    if entries is None:
        raise RuntimeError("Label JSON missing required 'entries' list")
    if not isinstance(entries, Sequence):
        raise RuntimeError("Label JSON 'entries' must be a sequence of strings")

    mapping: Set[Tuple[str, int, str]] = set()
    for entry in entries:
        if not isinstance(entry, str):
            raise RuntimeError(f"Label entry must be string, received {type(entry)!r}")
        parts = entry.split("|")
        if len(parts) != 3:
            raise RuntimeError(f"Label entry '{entry}' does not match expected format")
        chain_id, resseq_str, icode = parts
        if not chain_id:
            raise RuntimeError(f"Label entry '{entry}' is missing chain identifier")
        try:
            resseq = int(resseq_str)
        except ValueError as exc:
            raise RuntimeError(
                f"Residue number '{resseq_str}' in entry '{entry}' is not an integer"
            ) from exc
        mapping.add((chain_id, resseq, icode))

    return mapping


def _detect_structure_parser(structure_path: Path):
    """Return the appropriate Biopython parser for the structure path."""

    suffix = structure_path.suffix.lower()
    suffixes = "".join(structure_path.suffixes[-2:]).lower()

    if suffix in {".pdb", ".cif"}:
        if suffix == ".pdb":
            return PDBParser(QUIET=True)
        return MMCIFParser(QUIET=True)
    if suffix == ".gz":
        if suffixes.endswith(".pdb.gz"):
            return PDBParser(QUIET=True)
        if suffixes.endswith(".cif.gz"):
            return MMCIFParser(QUIET=True)

    raise StructureParsingError(
        "Structure path must end with .pdb, .pdb.gz, .cif, or .cif.gz"
    )


def parse_structure(structure_path: Path) -> Structure:
    """Parse the provided structure file into a ``Structure`` object."""

    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    parser = _detect_structure_parser(structure_path)
    try:
        structure_id = structure_path.stem
        structure = parser.get_structure(structure_id, str(structure_path))
    except Exception as exc:  # pragma: no cover - Biopython parser internals
        raise StructureParsingError(f"Failed to parse structure: {exc}") from exc

    try:
        next(structure.get_models())
    except StopIteration as exc:
        raise StructureParsingError("Structure does not contain any models") from exc

    return structure


def normalize_element(atom: Atom) -> str:
    """Return a normalized chemical element symbol for an atom."""

    element = (atom.element or "").strip()
    if not element:
        name = atom.get_name().strip()
        letters = "".join(ch for ch in name if ch.isalpha())
        if len(letters) >= 2 and letters[:2].upper() in {
            "SE",
            "CL",
            "BR",
            "ZN",
            "MG",
            "FE",
            "MN",
            "CU",
            "CO",
            "NI",
            "CA",
            "NA",
            "ZN",
            "CD",
            "HG",
            "SR",
            "K",
        }:
            element = letters[:2]
        elif letters:
            element = letters[0]
        else:
            element = "X"
    element = element.upper()
    if len(element) > 1:
        element = element[0] + element[1:].lower()
    return element


def one_hot_element(elem: str) -> np.ndarray:
    """Return the element one-hot encoding in the fixed channel order."""

    elem_upper = elem.upper()
    if elem_upper == "SE":
        key = "OTHER"
    elif elem_upper in ELEMENT_INDEX:
        key = elem_upper
    else:
        key = "OTHER"

    vec = np.zeros(len(ELEMENT_ORDER), dtype=np.float32)
    vec[ELEMENT_INDEX[key]] = 1.0
    return vec


def _altloc_priority(altloc: str) -> int:
    if altloc == "":
        return 2
    if altloc == "A":
        return 1
    return 0


def _normalize_icode(raw: str) -> str:
    raw = raw or ""
    return raw.strip() if raw.strip() else ""


def _normalize_resseq(resseq) -> int:
    try:
        return int(resseq)
    except (TypeError, ValueError) as exc:
        raise StructureParsingError(f"Residue sequence identifier '{resseq}' is invalid") from exc


def collect_heavy_atoms_with_summary(
    structure: Structure,
) -> Tuple[List[Tuple[Atom, str, int, str]], Dict[str, int]]:
    """Return heavy atoms alongside summary statistics for the structure."""

    try:
        model = structure[0]
    except KeyError as exc:
        raise StructureParsingError("Structure does not contain model 0") from exc

    selected: List[Tuple[Atom, str, int, str]] = []
    per_chain_counts: Dict[str, int] = {}
    chains_seen: Set[str] = set()
    residues_seen: Set[Tuple[str, int, str]] = set()
    atoms_total = 0
    hydrogens_removed = 0
    waters_skipped = 0
    hetero_skipped = 0

    for chain in model:
        chain_id = str(chain.id)
        for residue in chain:
            hetflag, resseq_raw, icode_raw = residue.id
            if hetflag not in _ALLOWED_RESIDUE_FLAGS:
                resname = (residue.get_resname() or "").strip().upper()
                if hetflag == "W" or resname in {"HOH", "H2O", "WAT"}:
                    waters_skipped += 1
                else:
                    hetero_skipped += 1
                continue

            resseq = _normalize_resseq(resseq_raw)
            icode = _normalize_icode(icode_raw)
            best_atoms: Dict[str, Tuple[float, int, str, int, Atom]] = {}

            for order_index, atom in enumerate(residue.child_list):
                atoms_total += 1
                element = normalize_element(atom)
                if element.upper() == "H":
                    hydrogens_removed += 1
                    continue
                altloc_raw = atom.get_altloc()
                altloc = "" if altloc_raw in ("", " ") else str(altloc_raw)
                occ_raw = atom.get_occupancy()
                occupancy = float(occ_raw) if occ_raw is not None else -1.0
                priority = _altloc_priority(altloc)
                atom_name = atom.get_name()

                existing = best_atoms.get(atom_name)
                if existing is None:
                    best_atoms[atom_name] = (occupancy, priority, altloc, order_index, atom)
                    continue

                best_occ, best_priority, best_altloc, best_index, best_atom = existing
                replace = False
                if occupancy > best_occ + ALTLOC_EPS:
                    replace = True
                elif abs(occupancy - best_occ) <= ALTLOC_EPS:
                    if priority > best_priority:
                        replace = True
                    elif priority == best_priority:
                        if altloc < best_altloc:
                            replace = True
                        elif altloc == best_altloc and order_index < best_index:
                            replace = True

                if replace:
                    best_atoms[atom_name] = (occupancy, priority, altloc, order_index, atom)

            for entry in sorted(best_atoms.values(), key=lambda item: item[3]):
                atom = entry[4]
                selected.append((atom, chain_id, resseq, icode))
                per_chain_counts[chain_id] = per_chain_counts.get(chain_id, 0) + 1
                chains_seen.add(chain_id)
                residues_seen.add((chain_id, resseq, icode))

    LOGGER.debug(
        "Selected heavy atoms=%d (total atoms processed=%d, hydrogens removed=%d, waters skipped=%d, hetero skipped=%d)",
        len(selected),
        atoms_total,
        hydrogens_removed,
        waters_skipped,
        hetero_skipped,
    )
    for chain_id in sorted(per_chain_counts):
        LOGGER.debug("Chain %s heavy atom count=%d", chain_id, per_chain_counts[chain_id])

    summary = {
        "chains": len(chains_seen),
        "residues": len(residues_seen),
        "atoms_total": atoms_total,
        "atoms_heavy": len(selected),
        "waters_skipped": waters_skipped,
        "hydrogens_dropped": hydrogens_removed,
    }

    return selected, summary


def iter_heavy_atoms(structure: Structure) -> Iterable[Tuple[Atom, str, int, str]]:
    """Iterate over heavy atoms in the first model of the structure."""

    atoms, _ = collect_heavy_atoms_with_summary(structure)
    return atoms


def compute_residue_sasa(
    structure: Structure, probe_radius: float, n_points: int
) -> Dict[Tuple[str, int, str], float]:
    """Compute per-residue solvent accessible surface area."""

    sr = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
    sr.compute(structure, level="R")

    sasa_map: Dict[Tuple[str, int, str], float] = {}
    try:
        model = structure[0]
    except KeyError as exc:
        raise StructureParsingError("Structure does not contain model 0") from exc

    for chain in model:
        chain_id = str(chain.id)
        for residue in chain:
            hetflag, resseq_raw, icode_raw = residue.id
            if hetflag not in _ALLOWED_RESIDUE_FLAGS:
                continue
            resseq = _normalize_resseq(resseq_raw)
            icode = _normalize_icode(icode_raw)
            sasa_value = getattr(residue, "sasa", 0.0) or 0.0
            sasa_map[(chain_id, resseq, icode)] = float(sasa_value)

    return sasa_map


def compute_residue_osp(
    structure_path: Path, allow_missing: bool
) -> Tuple[Dict[Tuple[str, int, str], float], str]:
    """Compute residue packing density using fibos OSP."""

    try:
        from fibos import osp as fibos_osp
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        msg = "fibos package is required to compute packing density"
        if allow_missing:
            LOGGER.warning("%s; filling OSP with zeros", msg)
            return {}, "zero_fallback"
        raise RuntimeError(msg) from exc

    shim = _ensure_os_create_folder_shim()
    try:
        raw_result = _call_fibos_osp(fibos_osp, structure_path)
    except Exception as exc:  # pragma: no cover - external tool failure
        msg = f"fibos.osp failed for {structure_path}: {exc}"
        if allow_missing:
            LOGGER.warning("%s; filling OSP with zeros", msg)
            return {}, "zero_fallback"
        raise RuntimeError(msg) from exc
    finally:
        shim.restore()

    osp_map: Dict[Tuple[str, int, str], float] = {}
    osp_source = "fibos"

    if isinstance(raw_result, dict):
        for key, value in raw_result.items():
            if not isinstance(key, Sequence) or len(key) != 3:
                continue
            chain_id = str(key[0])
            try:
                resseq = _normalize_resseq(key[1])
            except StructureParsingError:
                continue
            icode = _normalize_icode(str(key[2]))
            try:
                osp_value = float(value)
            except (TypeError, ValueError):
                continue
            osp_map[(chain_id, resseq, icode)] = osp_value
    else:
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover - pandas not installed
            if allow_missing:
                LOGGER.warning(
                    "fibos.osp returned a non-dict result but pandas is unavailable; filling OSP with zeros"
                )
                return {}, "zero_fallback"
            raise RuntimeError(
                "fibos.osp returned a non-dict result and pandas is required to interpret it"
            )

        if isinstance(raw_result, pd.DataFrame):
            df = raw_result
        else:
            try:
                df = pd.DataFrame(raw_result)
            except Exception as exc:  # pragma: no cover - unexpected structure
                if allow_missing:
                    LOGGER.warning(
                        "fibos.osp returned unsupported result (%s); filling OSP with zeros",
                        type(raw_result),
                    )
                    return {}, "zero_fallback"
                raise RuntimeError(
                    f"fibos.osp returned unsupported result type {type(raw_result)!r}"
                ) from exc

        chain_col = _first_present_column(
            df.columns,
            ["chain", "auth_asym_id", "asym_id", "chain_id", "auth_asym"],
        )
        resseq_col = _first_present_column(
            df.columns,
            ["resseq", "auth_seq_id", "seq_id", "residue_number", "res_id"],
        )
        icode_col = _first_present_column(
            df.columns, ["icode", "ins_code", "insertion_code", "auth_ins_code"]
        )
        osp_col = _first_present_column(df.columns, ["osp", "density", "packing_density"])

        required_missing = [
            name
            for name, col in {
                "chain": chain_col,
                "resseq": resseq_col,
                "osp": osp_col,
            }.items()
            if col is None
        ]
        if required_missing:
            if allow_missing:
                LOGGER.warning(
                    "fibos.osp result missing columns %s (available=%s); filling OSP with zeros",
                    ", ".join(required_missing),
                    ", ".join(str(col) for col in df.columns),
                )
                return {}, "zero_fallback"
            raise RuntimeError(
                f"fibos.osp result missing required columns: {', '.join(required_missing)}"
            )

        for _, row in df.iterrows():
            chain_id = str(row[chain_col])
            try:
                resseq = _normalize_resseq(row[resseq_col])
            except StructureParsingError:
                continue
            icode = _normalize_icode(str(row[icode_col])) if icode_col else ""
            try:
                osp_value = float(row[osp_col])
            except (TypeError, ValueError):
                continue
            osp_map[(chain_id, resseq, icode)] = osp_value
    if not osp_map and allow_missing:
        LOGGER.warning("fibos.osp returned no data; filling OSP with zeros")
        return {}, "zero_fallback"
    if not osp_map:
        raise RuntimeError("fibos.osp returned no density data")

    return osp_map, osp_source


class _OsCreateFolderShim:
    """Context-style helper that temporarily injects ``os.create_folder``."""

    def __init__(self, original: Optional[Callable[..., object]]) -> None:
        self._original = original

    def restore(self) -> None:
        """Restore the original ``os.create_folder`` binding if it was absent."""

        if self._original is not None:
            return
        try:
            delattr(os, "create_folder")
        except Exception:
            LOGGER.debug("Unable to remove temporary os.create_folder shim")


def _ensure_os_create_folder_shim() -> _OsCreateFolderShim:
    """Install a minimal ``os.create_folder`` shim for fibos if missing."""

    original = getattr(os, "create_folder", None)
    if callable(original):
        return _OsCreateFolderShim(original)

    def _create_folder(path: str | Path, exist_ok: bool = True) -> None:
        Path(path).mkdir(parents=True, exist_ok=exist_ok)

    os.create_folder = _create_folder  # type: ignore[attr-defined]
    LOGGER.debug("Installed temporary os.create_folder shim for fibos")
    return _OsCreateFolderShim(None)


def _first_present_column(columns: Sequence[object], candidates: Sequence[str]) -> str | None:
    """Return the first column name matching any candidate (case-insensitive)."""

    lowered_to_original = {str(col).lower(): str(col) for col in columns}
    for cand in candidates:
        match = lowered_to_original.get(cand.lower())
        if match is not None:
            return match
    return None


def _call_fibos_osp(
    fibos_osp: Callable[..., object], structure_path: Path
) -> object:
    """Call ``fibos.osp`` with optional ``prot_name`` support when available.

    Some fibos versions expect a ``prot_name`` keyword and raise
    ``UnboundLocalError`` when it is absent. This helper inspects the callable
    signature and supplies a sensible default when supported, falling back to a
    positional-only invocation otherwise.

    Parameters
    ----------
    fibos_osp:
        The ``fibos.osp`` callable.
    structure_path:
        Path to the structure file passed through to fibos.

    Returns
    -------
    object
        The raw result from ``fibos.osp``.
    """

    kwargs: Dict[str, object] = {}
    try:
        if "prot_name" in inspect.signature(fibos_osp).parameters:
            kwargs["prot_name"] = structure_path.stem
    except (TypeError, ValueError):  # pragma: no cover - built-in/partial callable
        pass

    try:
        return fibos_osp(str(structure_path), **kwargs)
    except TypeError as exc:
        if kwargs:
            LOGGER.debug(
                "Retrying fibos.osp without keyword args after TypeError: %s", exc
            )
            return fibos_osp(str(structure_path))
        raise


def build_arrays(
    structure_path: Path,
    labels_path: Path,
    probe_radius: float,
    n_points: int,
    allow_missing_density: bool,
) -> Dict[str, object]:
    """Build atom-level arrays and metadata for the provided structure."""

    labels = load_labels_json(labels_path)
    structure = parse_structure(structure_path)
    atoms_info, summary = collect_heavy_atoms_with_summary(structure)
    if not atoms_info:
        raise RuntimeError("No heavy atoms were found in the structure")

    sasa_map = compute_residue_sasa(structure, probe_radius, n_points)
    osp_map, osp_source = compute_residue_osp(structure_path, allow_missing_density)

    n_atoms = len(atoms_info)
    coords = np.empty((n_atoms, 3), dtype=np.float32)
    features = np.empty((n_atoms, 8), dtype=np.float32)
    labels_array = np.empty((n_atoms, 1), dtype=np.int64)

    meta: Dict[str, List] = {
        "chain_id": [],
        "resseq": [],
        "icode": [],
        "atom_name": [],
        "element": [],
    }

    for idx, (atom, chain_id, resseq, icode) in enumerate(atoms_info):
        coords[idx, :] = atom.get_coord().astype(np.float32)
        element = normalize_element(atom)
        one_hot = one_hot_element(element)
        features[idx, :6] = one_hot
        sasa = float(sasa_map.get((chain_id, resseq, icode), 0.0))
        osp = float(osp_map.get((chain_id, resseq, icode), 0.0))
        features[idx, 6] = sasa
        features[idx, 7] = osp
        labels_array[idx, 0] = 1 if (chain_id, resseq, icode) in labels else 0

        meta["chain_id"].append(chain_id)
        meta["resseq"].append(resseq)
        meta["icode"].append(icode)
        meta["atom_name"].append(atom.get_fullname().strip())
        meta["element"].append(element)

    provenance = {
        "osp_source": osp_source,
        "sasa_probe": float(probe_radius),
        "sasa_n_points": int(n_points),
    }
    summary_payload = {
        "chains": int(summary["chains"]),
        "residues": int(summary["residues"]),
        "atoms_total": int(summary["atoms_total"]),
        "atoms_heavy": int(summary["atoms_heavy"]),
        "waters_skipped": int(summary["waters_skipped"]),
        "hydrogens_dropped": int(summary["hydrogens_dropped"]),
    }

    payload: Dict[str, object] = {
        "coords": coords,
        "features": features,
        "labels": labels_array,
        "meta": meta,
        "provenance": provenance,
        "summary": summary_payload,
    }

    LOGGER.info(
        "atoms=%d (total=%d), residues=%d, chains=%d, osp=%s, probe=%.3f, points=%d",
        summary_payload["atoms_heavy"],
        summary_payload["atoms_total"],
        summary_payload["residues"],
        summary_payload["chains"],
        osp_source,
        provenance["sasa_probe"],
        provenance["sasa_n_points"],
    )

    return payload


def save_npz(out_path: Path, payload: Dict[str, object]) -> None:
    """Persist the payload to ``.npz`` format."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = payload.get("meta")
    if not isinstance(meta, dict):  # pragma: no cover - defensive programming
        raise RuntimeError("Payload missing 'meta' dictionary")

    provenance = payload.get("provenance", {}) if isinstance(payload.get("provenance"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}

    np.savez(
        out_path,
        coords=payload["coords"],
        features=payload["features"],
        labels=payload["labels"],
        meta_chain_id=np.array(meta["chain_id"], dtype=object),
        meta_resseq=np.array(meta["resseq"], dtype=np.int32),
        meta_icode=np.array(meta["icode"], dtype=object),
        meta_atom_name=np.array(meta["atom_name"], dtype=object),
        meta_element=np.array(meta["element"], dtype=object),
        provenance_osp_source=np.array(provenance.get("osp_source", "unknown"), dtype=object),
        provenance_sasa_probe=float(provenance.get("sasa_probe", 0.0)),
        provenance_sasa_n_points=int(provenance.get("sasa_n_points", 0)),
        summary_chains=int(summary.get("chains", 0)),
        summary_residues=int(summary.get("residues", 0)),
        summary_atoms_total=int(summary.get("atoms_total", 0)),
        summary_atoms_heavy=int(summary.get("atoms_heavy", 0)),
        summary_waters_skipped=int(summary.get("waters_skipped", 0)),
        summary_hydrogens_dropped=int(summary.get("hydrogens_dropped", 0)),
    )

    LOGGER.info("Saved features to %s", out_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Featurize a structure into atom-level point clouds."
    )
    parser.add_argument("--structure", required=True, help="Path to structure file")
    parser.add_argument("--labels", required=True, help="Path to residue label JSON")
    parser.add_argument("--out", help="Optional output .npz path")
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=1.4,
        help="SASA probe radius in Angstroms",
    )
    parser.add_argument(
        "--sasa-n-points",
        type=int,
        default=960,
        help="Number of points for Shrake-Rupley algorithm",
    )
    parser.add_argument(
        "--allow-missing-density",
        action="store_true",
        help="Fill missing packing density values with zeros",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    structure_path = Path(args.structure)
    labels_path = Path(args.labels)
    out_path = Path(args.out) if args.out else None

    try:
        payload = build_arrays(
            structure_path,
            labels_path,
            args.probe_radius,
            args.sasa_n_points,
            args.allow_missing_density,
        )
    except Exception as exc:
        LOGGER.error("Featurization failed: %s", exc)
        return 1

    if out_path is not None:
        save_npz(out_path, payload)

    meta = payload["meta"]
    chain_ids = meta["chain_id"]
    residues = set(zip(chain_ids, meta["resseq"], meta["icode"]))
    LOGGER.info(
        "Featurized %s: atoms=%d residues=%d chains=%d features_dim=%d",
        structure_path.name,
        len(chain_ids),
        len(residues),
        len(set(chain_ids)),
        payload["features"].shape[1],
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
