"""End-to-end dataset builder orchestrating the curation steps."""

from __future__ import annotations

import argparse
import json
from importlib import util
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py


def _load_module(name: str, path: Path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FETCH = _load_module("fetch_cull_pdbs", Path(__file__).with_name("01_fetch_cull_pdbs.py"))
SIFTS = _load_module(
    "sifts_label_mapper", Path(__file__).with_name("02_sifts_label_mapper.py")
)
FEATURIZER = _load_module(
    "atom_point_featurizer", Path(__file__).with_name("03_atom_point_featurizer.py")
)


def process_structure(pdb_path: Path, cath_id: str, output_dir: Path) -> Tuple[str, Path]:
    pdb_id = pdb_path.stem.split(".")[0]
    mapping = SIFTS.fetch_cath_mapping(pdb_id, cath_id)
    features = FEATURIZER.featurize_structure(
        pdb_path,
        {(chain, int(res)): int(label) for (chain, res), label in mapping.residues.items()},
    )
    output_path = output_dir / f"{pdb_id}.h5"
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("coords", data=features["coords"], compression="gzip")
        handle.create_dataset("features", data=features["features"], compression="gzip")
        handle.create_dataset("labels", data=features["labels"], compression="gzip")
    return pdb_id, output_path


def build_dataset(config_path: Path) -> List[Tuple[str, Path]]:
    config = json.loads(config_path.read_text()) if config_path.suffix == ".json" else None
    if config is None:
        import yaml

        config = yaml.safe_load(config_path.read_text())

    cath_id = config["cath_id"]
    raw_dir = Path(config["pdb_download_dir"])
    processed_dir = Path(config["processed_dataset_dir"])

    search_config = FETCH.SearchConfig(cath_id=cath_id, output_dir=raw_dir)
    search_config.ensure_output_dir()
    entry_ids = FETCH.fetch_entry_ids(search_config)
    FETCH.download_entries(entry_ids, search_config)

    cif_files = list(raw_dir.glob("*.cif")) + list(raw_dir.glob("*.cif.gz"))
    tasks = [(path, cath_id, processed_dir) for path in sorted(cif_files)]
    results: List[Tuple[str, Path]] = []
    if not tasks:
        return results

    with Pool() as pool:
        for pdb_id, output in pool.starmap(process_structure, tasks):
            results.append((pdb_id, output))
    return results


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML/JSON configuration file")
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_dataset(args.config)


if __name__ == "__main__":
    main()
