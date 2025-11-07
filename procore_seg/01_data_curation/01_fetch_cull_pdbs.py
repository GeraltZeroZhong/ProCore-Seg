"""Utilities for discovering and downloading PDB entries for a CATH superfamily.

This module talks to the RCSB Search API to identify all structures that
match a given CATH domain identifier and downloads their coordinate files as
mmCIF files.  The download logic is intentionally lightweight so the module
can be reused both as a command line entry point and as a Python library.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Iterable, List

import requests
from Bio.PDB import PDBList


SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


@dataclasses.dataclass(slots=True)
class SearchConfig:
    """Configuration describing which CATH superfamily to query."""

    cath_id: str
    output_dir: Path
    format: str = "mmCif"

    def ensure_output_dir(self) -> Path:
        """Create the output directory if it does not already exist."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def build_cath_query(cath_id: str) -> dict:
    """Build a JSON query payload for the RCSB search endpoint."""

    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_annotation.id",
                "operator": "exact_match",
                "value": cath_id,
            },
        },
        "return_type": "entry",
        "request_options": {"pager": {"start": 0, "rows": 5000}},
    }


def fetch_entry_ids(config: SearchConfig) -> List[str]:
    """Retrieve all entry identifiers associated with the requested CATH ID."""

    payload = build_cath_query(config.cath_id)
    response = requests.post(SEARCH_API_URL, data=json.dumps(payload), timeout=60)
    response.raise_for_status()
    data = response.json()
    return [result["identifier"] for result in data.get("result_set", [])]


def download_entries(entry_ids: Iterable[str], config: SearchConfig) -> List[Path]:
    """Download mmCIF files for the provided identifiers."""

    downloader = PDBList(obsolete=False)
    output_paths: List[Path] = []
    for entry_id in entry_ids:
        filename = downloader.retrieve_pdb_file(
            entry_id, pdir=str(config.output_dir), file_format=config.format
        )
        if filename is None:
            continue
        output_paths.append(Path(filename))
    return output_paths


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cath_id", type=str, help="CATH superfamily identifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_pdbs"),
        help="Directory to store downloaded mmCIF files",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = SearchConfig(cath_id=args.cath_id, output_dir=args.output_dir)
    config.ensure_output_dir()
    entry_ids = fetch_entry_ids(config)
    download_entries(entry_ids, config)


if __name__ == "__main__":
    main()
