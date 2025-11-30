"""Pipeline to build processed HDF5 datasets for a CATH superfamily.

This module coordinates the steps of fetching residue-level annotations from
SIFTS, featurizing atoms, and persisting per-entry HDF5 payloads.  The
implementation is resilient to interruptions (idempotent), parallel, and can
fall back to invoking the previous pipeline stages via their CLIs when direct
imports are not possible.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

SIFTS_MODULE_NAME = "_sifts_label_mapper"
FEATURIZER_MODULE_NAME = "_atom_point_featurizer"

DATASET_SIGNATURE = {
    "feature_order": ["C", "H", "O", "N", "S", "Other", "SASA", "OSP"],
    "label_def": "CATH_core_binary",
    "quantization_note": "voxelization done at training/inference, not stored here",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the dataset building pipeline."""

    cath_id: str
    raw_dir: Path
    out_dir: Path
    ids_file: Optional[Path]
    max_workers: int
    overwrite: bool
    timeout: int
    sasa_probe: float
    sasa_n_points: int
    allow_missing_density: bool
    log_level: str


def load_config_from_cli(argv: Sequence[str] | None = None) -> PipelineConfig:
    """Parse command-line arguments into a :class:`PipelineConfig`."""

    parser = argparse.ArgumentParser(
        description="Build processed atom-level datasets for a CATH superfamily."
    )
    parser.add_argument("--cath-id", required=True, help="Target CATH superfamily identifier")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./data/raw_pdbs"),
        help="Directory containing downloaded structure files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data/processed"),
        help="Output directory for generated HDF5 files",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Optional file containing newline-delimited PDB identifiers",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 outputs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for SIFTS GraphQL requests (seconds)",
    )
    parser.add_argument(
        "--sasa-probe",
        type=float,
        default=1.4,
        help="Probe radius for SASA computation (Angstroms)",
    )
    parser.add_argument(
        "--sasa-n-points",
        type=int,
        default=960,
        help="Number of points for Shrake-Rupley SASA algorithm",
    )
    parser.add_argument(
        "--allow-missing-density",
        action="store_true",
        help="Allow missing density values when computing OSP features",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args(argv)

    return PipelineConfig(
        cath_id=args.cath_id,
        raw_dir=args.raw_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        ids_file=args.ids_file.resolve() if args.ids_file else None,
        max_workers=max(1, args.max_workers),
        overwrite=bool(args.overwrite),
        timeout=args.timeout,
        sasa_probe=args.sasa_probe,
        sasa_n_points=args.sasa_n_points,
        allow_missing_density=bool(args.allow_missing_density),
        log_level=args.log_level.upper(),
    )


def discover_ids_and_paths(raw_dir: Path, ids_file: Path | None) -> Dict[str, Path]:
    """Discover structure paths for PDB identifiers.

    Local structures are preferred over remote downloads and the selection order
    prioritises PDB files (``.pdb``/``.pdb.gz``) before mmCIF counterparts to
    maximise compatibility with downstream tooling (e.g., fibos). If only mmCIF
    files are present they will still be used directly rather than triggering a
    fresh download.
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    preferred_suffixes = [".pdb", ".pdb.gz", ".cif", ".cif.gz"]

    discovered: Dict[str, Dict[str, Path]] = {}

    for suffix in preferred_suffixes:
        pattern = f"*{suffix}"
        for path in raw_dir.rglob(pattern):
            if not path.is_file():
                continue
            stem = path.name.split(".")[0]
            if len(stem) != 4 or not stem.isalnum():
                continue
            pdb_id = stem.upper()
            discovered.setdefault(pdb_id, {})[suffix] = path

    def select_path(pdb_id: str) -> Path | None:
        variants = discovered.get(pdb_id)
        if not variants:
            return None
        for suffix in preferred_suffixes:
            if suffix in variants:
                return variants[suffix]
        return None

    id_list: Iterable[str]
    if ids_file and ids_file.exists():
        LOGGER.info("Loading PDB identifiers from %s", ids_file)
        with ids_file.open("r", encoding="utf-8") as handle:
            id_list = [line.strip().upper() for line in handle if line.strip()]
    else:
        LOGGER.info("Scanning %s for structure files", raw_dir)
        id_list = sorted(discovered.keys())

    mapping: Dict[str, Path] = {}
    for pdb_id in id_list:
        if not pdb_id:
            continue
        path = select_path(pdb_id)
        if path is None:
            LOGGER.warning("No structure file found for %s", pdb_id)
            continue
        mapping[pdb_id.upper()] = path

    return mapping


def best_structure_path(raw_dir: Path, pdb_id: str) -> Path | None:
    """Return the best available structure path for a PDB identifier."""

    pdb_id = pdb_id.strip().lower()
    candidates = [
        raw_dir / f"{pdb_id}.pdb",
        raw_dir / f"{pdb_id}.pdb.gz",
        raw_dir / f"{pdb_id}.cif",
        raw_dir / f"{pdb_id}.cif.gz",
        raw_dir / f"{pdb_id.upper()}.pdb",
        raw_dir / f"{pdb_id.upper()}.pdb.gz",
        raw_dir / f"{pdb_id.upper()}.cif",
        raw_dir / f"{pdb_id.upper()}.cif.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # fallback: search recursively
    target = pdb_id.lower()
    for suffix in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        pattern = f"{target}{suffix}"
        for path in raw_dir.rglob(pattern):
            if path.is_file():
                return path

    return None


def _load_support_module(filename: str, module_name: str):
    import importlib.util

    module_path = Path(__file__).resolve().parent / filename
    if not module_path.exists():
        raise ImportError(f"Support module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _serialize_residue_map(residues: Set[Tuple[str, int, str]]) -> List[str]:
    return [
        f"{chain}|{resseq}|{icode}"
        for chain, resseq, icode in sorted(residues, key=lambda item: (item[0], item[1], item[2]))
    ]


def get_residue_map(
    pdb_id: str, cath_id: str, timeout: int, tmpdir: Path
) -> Set[Tuple[str, int, str]]:
    """Return the set of residues belonging to the requested CATH superfamily."""

    try:
        module = _load_support_module("02_sifts_label_mapper.py", SIFTS_MODULE_NAME)
        fetch_instance_features = getattr(module, "fetch_instance_features")
        filter_cath_features = getattr(module, "filter_cath_features")
        expand_ranges_to_map = getattr(module, "expand_ranges_to_map")
    except ImportError as exc:
        LOGGER.debug("Falling back to CLI for SIFTS labels: %s", exc)
        return _residue_map_via_cli(pdb_id, cath_id, timeout, tmpdir)

    instances = fetch_instance_features(pdb_id, timeout)
    cath_features = filter_cath_features(instances, cath_id)
    mapping: Dict[Tuple[str, int, str], int] = expand_ranges_to_map(cath_features)
    residues = {key for key, value in mapping.items() if value}
    if not residues:
        raise RuntimeError(
            f"No residues annotated with CATH superfamily {cath_id} for entry {pdb_id}"
        )
    return residues


def _residue_map_via_cli(
    pdb_id: str, cath_id: str, timeout: int, tmpdir: Path
) -> Set[Tuple[str, int, str]]:
    script_path = Path(__file__).resolve().parent / "02_sifts_label_mapper.py"
    if not script_path.exists():
        raise RuntimeError("SIFTS label mapper script not found for CLI fallback")

    with NamedTemporaryFile("w", delete=False, suffix=".json", dir=tmpdir) as handle:
        tmp_path = Path(handle.name)

    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--pdb-id",
        pdb_id,
        "--out",
        str(tmp_path),
        "--timeout",
        str(timeout),
    ]
    if cath_id:
        cmd.extend(["--cath-superfamily", cath_id])

    LOGGER.debug("Executing SIFTS CLI: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"SIFTS CLI failed for {pdb_id}: {result.stderr.strip() or result.stdout.strip()}"
        )

    try:
        with tmp_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    finally:
        tmp_path.unlink(missing_ok=True)

    entries = payload.get("entries")
    if not entries:
        raise RuntimeError(
            f"No residues annotated with CATH superfamily {cath_id} for entry {pdb_id}"
        )

    residues: Set[Tuple[str, int, str]] = set()
    for entry in entries:
        chain, resseq, icode = entry.split("|")
        residues.add((chain, int(resseq), icode))
    return residues


def featurize(
    pdb_id: str,
    structure_path: Path,
    labels_map: Set[Tuple[str, int, str]],
    sasa_probe: float,
    sasa_n_points: int,
    allow_missing_density: bool,
    tmpdir: Path,
) -> Dict[str, object]:
    """Featurize the structure into atom-level arrays."""

    labels_payload = {"entries": _serialize_residue_map(labels_map)}

    with NamedTemporaryFile("w", delete=False, suffix=".json", dir=tmpdir) as handle:
        json.dump(labels_payload, handle)
        labels_path = Path(handle.name)

    try:
        try:
            module = _load_support_module("03_atom_point_featurizer.py", FEATURIZER_MODULE_NAME)
            build_arrays = getattr(module, "build_arrays")
        except ImportError as exc:
            LOGGER.debug("Falling back to featurizer CLI for %s: %s", pdb_id, exc)
            return _featurize_via_cli(
                structure_path,
                labels_path,
                sasa_probe,
                sasa_n_points,
                allow_missing_density,
                tmpdir,
            )

        payload = build_arrays(
            structure_path,
            labels_path,
            sasa_probe,
            sasa_n_points,
            allow_missing_density,
        )
        return payload
    finally:
        labels_path.unlink(missing_ok=True)


def _featurize_via_cli(
    structure_path: Path,
    labels_path: Path,
    sasa_probe: float,
    sasa_n_points: int,
    allow_missing_density: bool,
    tmpdir: Path,
) -> Dict[str, object]:
    script_path = Path(__file__).resolve().parent / "03_atom_point_featurizer.py"
    if not script_path.exists():
        raise RuntimeError("Featurizer script not found for CLI fallback")

    with NamedTemporaryFile(delete=False, suffix=".npz", dir=tmpdir) as handle:
        npz_path = Path(handle.name)

    cmd = [
        sys.executable,
        str(script_path),
        "--structure",
        str(structure_path),
        "--labels",
        str(labels_path),
        "--out",
        str(npz_path),
        "--probe-radius",
        f"{sasa_probe}",
        "--sasa-n-points",
        str(sasa_n_points),
    ]
    if allow_missing_density:
        cmd.append("--allow-missing-density")

    LOGGER.debug("Executing featurizer CLI: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        npz_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Featurizer CLI failed for {structure_path.name}: {result.stderr.strip() or result.stdout.strip()}"
        )

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            payload = {
                "coords": data["coords"],
                "features": data["features"],
                "labels": data["labels"],
                "meta": {
                    "chain_id": data["meta_chain_id"].tolist(),
                    "resseq": data["meta_resseq"].tolist(),
                    "icode": data["meta_icode"].tolist(),
                    "atom_name": data["meta_atom_name"].tolist(),
                    "element": data["meta_element"].tolist(),
                },
            }
            if "provenance_osp_source" in data:
                payload["provenance"] = {
                    "osp_source": str(np.asarray(data["provenance_osp_source"]).item()),
                    "sasa_probe": float(np.asarray(data["provenance_sasa_probe"]).item()),
                    "sasa_n_points": int(np.asarray(data["provenance_sasa_n_points"]).item()),
                }
            if "summary_atoms_heavy" in data:
                payload["summary"] = {
                    "atoms_heavy": int(np.asarray(data["summary_atoms_heavy"]).item()),
                    "atoms_total": int(np.asarray(data["summary_atoms_total"]).item()),
                    "chains": int(np.asarray(data["summary_chains"]).item()),
                    "residues": int(np.asarray(data["summary_residues"]).item()),
                    "waters_skipped": int(np.asarray(data["summary_waters_skipped"]).item()),
                    "hydrogens_dropped": int(np.asarray(data["summary_hydrogens_dropped"]).item()),
                }
    finally:
        npz_path.unlink(missing_ok=True)

    return payload


def write_h5(
    out_path: Path,
    pdb_id: str,
    cath_id: str,
    payload: Dict[str, object],
    sasa_probe: float,
    sasa_n_points: int,
) -> None:
    """Persist the featurization payload to the specified HDF5 path."""

    import h5py

    coords = np.asarray(payload.get("coords"), dtype=np.float32)
    features = np.asarray(payload.get("features"), dtype=np.float32)
    labels = np.asarray(payload.get("labels"), dtype=np.int64)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Payload missing 'meta' dictionary")

    n_atoms = coords.shape[0]
    if coords.shape != (n_atoms, 3):
        raise ValueError("Coords array must have shape (N, 3)")
    if features.shape != (n_atoms, 8):
        raise ValueError("Features array must have shape (N, 8)")
    if labels.shape != (n_atoms, 1):
        raise ValueError("Labels array must have shape (N, 1)")

    chains = list(meta.get("chain_id", []))
    resseqs = list(meta.get("resseq", []))
    icodes = list(meta.get("icode", []))
    atom_names = list(meta.get("atom_name", []))
    elements = list(meta.get("element", []))

    if not all(len(lst) == n_atoms for lst in [chains, resseqs, icodes, atom_names, elements]):
        raise ValueError("Metadata lengths must match number of atoms")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import Bio  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency missing
        Bio = None  # type: ignore[assignment]
    try:  # pragma: no cover - optional dependency
        import fibos  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency missing
        fibos = None  # type: ignore[assignment]

    tool_versions = {
        "python": platform.python_version(),
        "biopython": getattr(Bio, "__version__", "unknown") if "Bio" in locals() and Bio else "unknown",
        "h5py": getattr(h5py, "__version__", "unknown"),
        "fibos": (
            getattr(fibos, "__version__", None) or getattr(fibos, "VERSION", "unknown")
            if "fibos" in locals() and fibos is not None
            else "unknown"
        ),
    }

    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}

    residues_set = set(zip(chains, resseqs, icodes))
    summary_residues = int(summary.get("residues", len(residues_set)))
    summary_chains = int(summary.get("chains", len(set(chains))))
    summary_atoms_heavy = int(summary.get("atoms_heavy", n_atoms))
    summary_atoms_total = int(summary.get("atoms_total", n_atoms))
    osp_source = str(provenance.get("osp_source", "unknown"))

    dtype_str = h5py.special_dtype(vlen=str)

    with NamedTemporaryFile(dir=out_path.parent, suffix=".tmp", delete=False) as tmp_handle:
        tmp_path = Path(tmp_handle.name)

    try:
        with h5py.File(tmp_path, "w") as handle:
            handle.create_dataset("coords", data=coords, dtype=np.float32)
            handle.create_dataset("features", data=features, dtype=np.float32)
            handle.create_dataset("labels", data=labels, dtype=np.int64)

            meta_group = handle.create_group("meta")
            meta_group.create_dataset("chain_id", data=np.array(chains, dtype=object), dtype=dtype_str)
            meta_group.create_dataset("resseq", data=np.asarray(resseqs, dtype=np.int32))
            meta_group.create_dataset("icode", data=np.array(icodes, dtype=object), dtype=dtype_str)
            meta_group.create_dataset("atom_name", data=np.array(atom_names, dtype=object), dtype=dtype_str)
            meta_group.create_dataset("element", data=np.array(elements, dtype=object), dtype=dtype_str)

            handle.attrs["pdb_id"] = pdb_id.upper()
            handle.attrs["cath_id"] = cath_id
            handle.attrs["generated_at"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["sasa_probe"] = float(sasa_probe)
            handle.attrs["sasa_n_points"] = int(sasa_n_points)
            handle.attrs["feature_dim"] = features.shape[1]
            handle.attrs["dataset_signature"] = json.dumps(DATASET_SIGNATURE, sort_keys=True)
            handle.attrs["tool_versions"] = json.dumps(tool_versions, sort_keys=True)
            handle.attrs["osp_source"] = osp_source
            handle.attrs["atoms_total"] = summary_atoms_total
            handle.attrs["atoms_heavy"] = summary_atoms_heavy
            handle.attrs["residues"] = summary_residues
            handle.attrs["chains"] = summary_chains

        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_one(args: Tuple) -> Dict[str, object]:
    """Worker function for processing a single PDB entry."""

    (
        config,
        pdb_id,
        structure_path,
    ) = args

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, config.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )

    out_path = config.out_dir / f"{pdb_id.upper()}.h5"
    result: Dict[str, object] = {
        "pdb_id": pdb_id.upper(),
        "status": "failed",
        "structure_path": str(structure_path) if structure_path else None,
        "h5_path": None,
        "atoms": 0,
        "residues": 0,
        "chains": 0,
        "osp_source": "unknown",
        "error": None,
    }

    if structure_path is None or not structure_path.exists():
        result["error"] = "Structure file missing"
        return result

    if out_path.exists() and not config.overwrite:
        result.update({"status": "skipped", "h5_path": str(out_path)})
        return result

    try:
        with TemporaryDirectory() as tmp_dir_str:
            tmpdir = Path(tmp_dir_str)
            residues = get_residue_map(pdb_id, config.cath_id, config.timeout, tmpdir)
            payload = featurize(
                pdb_id,
                structure_path,
                residues,
                config.sasa_probe,
                config.sasa_n_points,
                config.allow_missing_density,
                tmpdir,
            )

            coords = np.asarray(payload.get("coords"))
            if coords.size == 0:
                raise RuntimeError("Featurizer returned zero atoms")

            write_h5(out_path, pdb_id, config.cath_id, payload, config.sasa_probe, config.sasa_n_points)

        meta = payload["meta"]
        chain_ids = list(meta["chain_id"])
        residues_set = set(zip(chain_ids, meta["resseq"], meta["icode"]))
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        atoms_heavy = int(summary.get("atoms_heavy", coords.shape[0]))
        residues_count = int(summary.get("residues", len(residues_set)))
        chains_count = int(summary.get("chains", len(set(chain_ids))))
        result.update(
            {
                "status": "processed",
                "h5_path": str(out_path),
                "atoms": atoms_heavy,
                "residues": residues_count,
                "chains": chains_count,
                "osp_source": str(provenance.get("osp_source", "unknown")),
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - best-effort logging
        result["status"] = "failed"
        result["error"] = f"{exc}"
        LOGGER.debug("Processing failed for %s: %s", pdb_id, traceback.format_exc())
        return result


def write_manifest(manifest_path: Path, payload: Dict[str, object]) -> None:
    """Write the dataset manifest atomically."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", encoding="utf-8", dir=manifest_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path = Path(handle.name)

    os.replace(tmp_path, manifest_path)


def _summarize_results(results: Sequence[Dict[str, object]]) -> Dict[str, int]:
    summary = {"processed": 0, "skipped": 0, "failed": 0}
    for item in results:
        status = item.get("status")
        if status in summary:
            summary[status] += 1
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the dataset-building pipeline."""

    config = load_config_from_cli(argv)

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        id_path_map = discover_ids_and_paths(config.raw_dir, config.ids_file)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    if not id_path_map:
        LOGGER.error("No structures found to process")
        return 1

    items = sorted(id_path_map.items())
    total = len(items)
    results: List[Dict[str, object]] = []

    queue_limit = max(config.max_workers * 2, 1)

    from tqdm import tqdm

    LOGGER.info("Processing %d structures", total)
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        pending = set()
        iterator = iter(items)
        with tqdm(total=total) as progress:
            try:
                while True:
                    while len(pending) < queue_limit:
                        try:
                            pdb_id, structure_path = next(iterator)
                        except StopIteration:
                            break
                        future = executor.submit(
                            process_one,
                            (
                                config,
                                pdb_id,
                                structure_path,
                            ),
                        )
                        pending.add(future)

                    if not pending:
                        break

                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        results.append(result)
                        progress.update(1)
            except KeyboardInterrupt:  # pragma: no cover - manual interruption
                LOGGER.warning("Interrupted by user. Cancelling remaining tasks...")
                for future in pending:
                    future.cancel()
                executor.shutdown(cancel_futures=True)
                raise

        # drain any remaining results
        for future in pending:
            result = future.result()
            results.append(result)

    summary = _summarize_results(results)
    manifest_payload = {
        "cath_id": config.cath_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(config.raw_dir),
        "out_dir": str(config.out_dir),
        "sasa_probe": config.sasa_probe,
        "sasa_n_points": config.sasa_n_points,
        "allow_missing_density": config.allow_missing_density,
        "count": total,
        "summary": {"downloaded": 0, **summary},
        "entries": results,
    }

    manifest_path = config.out_dir / "dataset_manifest.json"
    write_manifest(manifest_path, manifest_payload)

    LOGGER.info(
        "Finished processing %d entries (%d processed, %d skipped, %d failed)",
        total,
        summary["processed"],
        summary["skipped"],
        summary["failed"],
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

