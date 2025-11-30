"""Atom-level point featurizer for macromolecular structures."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing
import os
import shutil
import urllib.error
from contextlib import redirect_stderr
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
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


def _prepare_structure_for_fibos(
    structure_path: Path, structure: Structure
) -> tuple[Path, Callable[[], None]]:
    """Return a PDB path suitable for fibos and a cleanup callback.

    fibos expects either a 4-letter PDB identifier or a valid local PDB file
    path. This helper ensures we always provide the latter by:

    * returning the original path when it already points to a PDB file,
    * inflating ``.pdb.gz`` inputs to a temporary ``.pdb`` in the same folder,
    * exporting mmCIF inputs to a temporary PDB named after the original stem
      (e.g., ``kqlv.pdb``) using :class:`Bio.PDB.PDBIO`.

    The accompanying cleanup callable removes any temporary artefacts so callers
    can safely invoke it in a ``finally`` block.
    """

    suffixes = "".join(structure_path.suffixes).lower()

    def _noop_cleanup() -> None:
        return None

    if suffixes.endswith(".pdb"):
        sanitized, sanitize_cleanup = _sanitize_pdb_for_fibos(structure_path)

        def _cleanup() -> None:
            sanitize_cleanup()

        return sanitized, _cleanup

    parent_dir = structure_path.parent
    stemmed_pdb = parent_dir / f"{structure_path.stem}.pdb"

    if suffixes.endswith(".pdb.gz"):
        with gzip.open(structure_path, "rt") as src:
            with stemmed_pdb.open("w") as handle:
                shutil.copyfileobj(src, handle)

        def _cleanup() -> None:
            stemmed_pdb.unlink(missing_ok=True)

        sanitized, sanitize_cleanup = _sanitize_pdb_for_fibos(stemmed_pdb)

        def _composite_cleanup() -> None:
            sanitize_cleanup()
            _cleanup()

        return sanitized, _composite_cleanup

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(stemmed_pdb))

    def _cleanup() -> None:
        stemmed_pdb.unlink(missing_ok=True)

    sanitized, sanitize_cleanup = _sanitize_pdb_for_fibos(stemmed_pdb)

    def _composite_cleanup() -> None:
        sanitize_cleanup()
        _cleanup()

    return sanitized, _composite_cleanup


def _sanitize_pdb_for_fibos(structure_path: Path) -> tuple[Path, Callable[[], None]]:
    """Filter malformed atom records that fibos fails to parse.

    The legacy fibos Fortran reader expects integer residue and atom serial
    fields and well-formed floating-point coordinates. Some experimental PDB
    files contain placeholders or other non-numeric tokens that trigger a
    Fortran runtime error (``Bad value during integer read``). To shield the
    downstream occluded surface computation we create a temporary copy with
    malformed ``ATOM`` / ``HETATM`` records removed.
    """

    tmp_path = structure_path.with_suffix(structure_path.suffix + ".fibos_safe")
    removed = 0

    try:
        with structure_path.open("r") as src, tmp_path.open("w") as dst:
            for line in src:
                if not line.startswith(("ATOM", "HETATM")):
                    dst.write(line)
                    continue

                serial_field = line[6:11]
                resseq_field = line[22:26]
                try:
                    int(serial_field)
                    int(resseq_field)
                    float(line[30:38])
                    float(line[38:46])
                    float(line[46:54])
                except ValueError:
                    removed += 1
                    continue

                dst.write(line)
    except FileNotFoundError:
        return structure_path, lambda: None

    if removed == 0:
        tmp_path.unlink(missing_ok=True)
        return structure_path, lambda: None

    LOGGER.debug(
        "Filtered %d malformed atom records from %s prior to fibos", removed, structure_path
    )

    def _cleanup() -> None:
        tmp_path.unlink(missing_ok=True)

    return tmp_path, _cleanup


def compute_residue_osp(
    structure_path: Path,
    allow_missing: bool,
    default_chain_id: str | None = None,
    *,
    prot_name: str | None = None,
) -> Tuple[Dict[Tuple[str, int, str], float], str]:
    """Compute residue packing density using fibos OSP.

    Parameters
    ----------
    structure_path:
        Path to the structure file passed to fibos.
    allow_missing:
        Whether failures should fall back to zero-valued densities.
    default_chain_id:
        Fallback chain identifier to use when the fibos result omits chain
        information (e.g., when processing single-chain structures).
    prot_name:
        Preferred identifier to hand to fibos (e.g., the 4-letter PDB code)
        so SRF naming stays stable even when the structure path comes from a
        temporary mmCIF-to-PDB conversion.

    Callers should provide a real PDB filepath (see
    :func:`_prepare_structure_for_fibos`) so fibos does not attempt to invent
    paths such as ``pdb/<absolute>.ent``. When remote downloads fail (e.g., HTTP
    errors) and ``allow_missing`` is set, the OSP channel is filled with zeros
    while the pipeline continues.
    """

    try:
        import fibos  # type: ignore

        fibos_osp = fibos.osp
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        msg = "fibos package is required to compute packing density"
        if allow_missing:
            LOGGER.warning("%s; filling OSP with zeros", msg)
            return {}, "zero_fallback"
        raise RuntimeError(msg) from exc

    resolved_prot_name = prot_name or structure_path.stem
    srf_path = _build_fibos_srf_path(structure_path, prot_name=resolved_prot_name)
    shim = _ensure_os_create_folder_shim()
    try:
        resolved_srf = _ensure_fibos_srf_file(
            fibos,
            structure_path,
            srf_path,
            allow_missing=allow_missing,
            prot_name=resolved_prot_name,
        )
        if resolved_srf is None:
            return {}, "zero_fallback"
        raw_result = _call_fibos_osp(
            fibos_osp, resolved_srf, prot_name=resolved_prot_name
        )
    except Exception as exc:  # pragma: no cover - external tool failure
        msg = (
            "fibos.osp(file_path) failed for %s: %s; expected modern signature without"
            " prot_name"
        ) % (structure_path, exc)
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
            ["resseq", "auth_seq_id", "seq_id", "residue_number", "res_id", "resnum"],
        )
        icode_col = _first_present_column(
            df.columns, ["icode", "ins_code", "insertion_code", "auth_ins_code"]
        )
        osp_col = _first_present_column(df.columns, ["osp", "density", "packing_density"])

        required_missing = []
        if chain_col is None and default_chain_id is None:
            required_missing.append("chain")
        if resseq_col is None:
            required_missing.append("resseq")
        if osp_col is None:
            required_missing.append("osp")
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
            if chain_col is not None:
                chain_id = str(row[chain_col])
            else:
                chain_id = default_chain_id
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


def _build_fibos_srf_path(structure_path: Path, *, prot_name: str | None = None) -> Path:
    """Construct the expected SRF path produced by ``fibos.occluded_surface``.

    When a custom ``prot_name`` is provided (e.g., the original 4-letter PDB
    identifier), the SRF name stays stable even if the structure path comes from
    a temporary conversion.
    """

    base = prot_name or structure_path.stem
    return structure_path.parent / "fibos_files" / f"prot_{base}.srf"


def _ensure_fibos_srf_file(
    fibos_module: object,
    structure_path: Path,
    srf_path: Path,
    *,
    allow_missing: bool,
    prot_name: str,
) -> Path | None:
    """Ensure an SRF file exists for ``fibos.osp`` to consume.

    fibos 0.3+ expects ``osp(file_path)`` where ``file_path`` points to an SRF
    file generated by ``fibos.occluded_surface``. This helper attempts to
    generate the SRF when it is missing while keeping the legacy zero-fallback
    behaviour intact when generation is unavailable or fails.
    """

    if srf_path.exists():
        return srf_path

    occluded_surface = getattr(fibos_module, "occluded_surface", None)
    if not callable(occluded_surface):
        msg = (
            "fibos.osp(file_path) requires an SRF file but fibos.occluded_surface"
            f" is unavailable and no SRF was found at {srf_path}"
        )
        if allow_missing:
            LOGGER.warning("%s; filling OSP with zeros", msg)
            return None
        raise RuntimeError(msg)

    srf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _call_fibos_occluded_surface_sandboxed(
            structure_path, srf_path.parent, prot_name
        )
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        status = getattr(exc, "code", None)
        pdb_id = prot_name[:4].upper()
        msg = (
            f"fibos.occluded_surface HTTP {status or 'error'} when fetching {pdb_id}"
        )
        if allow_missing:
            LOGGER.warning("%s; filling OSP with zeros", msg)
            return None
        raise RuntimeError(msg) from exc
    except Exception as exc:  # pragma: no cover - external tool failure
        msg = f"fibos.occluded_surface failed for {structure_path}: {exc}"
        if allow_missing:
            LOGGER.warning("%s; filling OSP with zeros", msg)
            return None
        raise RuntimeError(msg) from exc

    if srf_path.exists():
        return srf_path

    if isinstance(result, (str, os.PathLike)) and Path(result).suffix.lower() == ".srf":
        alt_path = Path(result)
        if alt_path.exists():
            return alt_path

    msg = (
        "fibos.occluded_surface completed but no SRF was produced at %s;"
        " expected by osp(file_path)"
    ) % srf_path
    if allow_missing:
        LOGGER.warning("%s; filling OSP with zeros", msg)
        return None
    raise RuntimeError(msg)


def _call_fibos_occluded_surface(
    occluded_surface: Callable[..., object],
    structure_path: Path,
    out_dir: Path,
    prot_name: str,
) -> object:
    """Call ``fibos.occluded_surface`` with best-effort signature detection."""

    try:
        return occluded_surface(
            file_path=str(structure_path),
            out_folder=str(out_dir),
            prot_name=prot_name,
        )
    except TypeError:
        try:
            return occluded_surface(
                file_path=str(structure_path), out_folder=str(out_dir)
            )
        except TypeError:
            try:
                return occluded_surface(
                    str(structure_path), out_folder=str(out_dir), prot_name=prot_name
                )
            except TypeError:
                try:
                    return occluded_surface(
                        str(structure_path), out_folder=str(out_dir)
                    )
                except TypeError:
                    return occluded_surface(str(structure_path))


def _call_fibos_occluded_surface_sandboxed(
    structure_path: Path,
    out_dir: Path,
    prot_name: str,
    *,
    timeout: float | None = None,
) -> object:
    """Call ``fibos.occluded_surface`` in a child process to isolate crashes.

    fibos wraps Fortran code that can terminate the interpreter on malformed
    input. Running the computation in a dedicated process ensures that we can
    detect non-zero exit codes or other fatal errors and fall back to neutral
    OSP values without killing the caller.
    """

    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[var-annotated]

    worker = ctx.Process(
        target=_fibos_occluded_surface_worker,
        args=(structure_path, out_dir, prot_name, result_queue),
        daemon=True,
    )
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise RuntimeError(
            "fibos.occluded_surface exceeded time limit and was terminated"
        )

    exit_code = worker.exitcode
    try:
        payload = result_queue.get_nowait()
    except Exception:  # pragma: no cover - defensive guard
        payload = None

    if exit_code is None:
        raise RuntimeError("fibos.occluded_surface subprocess ended unexpectedly")
    if exit_code != 0:
        details = ""
        if isinstance(payload, dict) and payload.get("error"):
            details = f": {payload['error']}"
        elif exit_code == 2:
            details = ": fibos reader failed to parse one or more atom records"
        raise RuntimeError(
            f"fibos.occluded_surface subprocess exited with code {exit_code}{details}"
        )

    if not isinstance(payload, dict):
        raise RuntimeError("fibos.occluded_surface subprocess returned no result")
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "unknown fibos error"))

    return payload.get("result")


def _fibos_occluded_surface_worker(
    structure_path: Path,
    out_dir: Path,
    prot_name: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """Invoke fibos.occluded_surface and report success/failure through a queue."""

    try:
        import fibos  # type: ignore

        occluded_surface = getattr(fibos, "occluded_surface", None)
        if not callable(occluded_surface):
            raise RuntimeError("fibos.occluded_surface is unavailable")

        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(
            devnull
        ):
            result = _call_fibos_occluded_surface(
                occluded_surface, structure_path, out_dir, prot_name
            )
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put({"ok": False, "error": repr(exc)})


def _call_fibos_osp(
    fibos_osp: Callable[..., object], srf_path: Path, *, prot_name: str | None = None
) -> object:
    """Call ``fibos.osp`` preferring the modern ``osp(file_path)`` signature."""

    try:
        return fibos_osp(str(srf_path))
    except FileNotFoundError:
        raise
    except TypeError as exc:
        if prot_name is None:
            prot_name = srf_path.stem
        LOGGER.debug(
            "Retrying fibos.osp with prot_name compatibility shim for %s", srf_path
        )
        try:
            return fibos_osp(str(srf_path), prot_name=prot_name)
        except TypeError as exc2:
            raise exc2 from exc



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
    chain_ids = {chain_id for _, chain_id, _, _ in atoms_info}
    default_chain_id = next(iter(chain_ids)) if len(chain_ids) == 1 else None
    fibos_structure_path, cleanup_fibos = _prepare_structure_for_fibos(
        structure_path, structure
    )
    try:
        osp_map, osp_source = compute_residue_osp(
            fibos_structure_path,
            allow_missing_density,
            default_chain_id=default_chain_id,
            prot_name=structure_path.stem,
        )
    finally:
        cleanup_fibos()

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

    # Feature layout: [C, H, O, N, S, OTHER, SASA, OSP]
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
