"""Retrieve residue-level CATH assignments from the RCSB data API."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests


GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"


@dataclass(slots=True)
class ResidueIdentifier:
    """Residue-level identifier used by the SIFTS mapping."""

    asym_id: str
    seq_id: int


@dataclass(slots=True)
class LabelMappingResult:
    """Mapping result returned by :func:`fetch_cath_mapping`."""

    pdb_id: str
    cath_id: str
    residues: Dict[Tuple[str, int], int]


GRAPHQL_TEMPLATE = """
query cathMapping($entryId: String!) {
  entry(entry_id: $entryId) {
    polymer_entity_instances {
      rcsb_id
      rcsb_polymer_instance_feature {
        type
        provenance_source
        feature_positions {
          beg_seq_id
          end_seq_id
        }
      }
    }
  }
}
"""


def fetch_cath_mapping(pdb_id: str, cath_id: str | None = None) -> LabelMappingResult:
    """Fetch CATH residue annotations for a single structure."""

    payload = {"query": GRAPHQL_TEMPLATE, "variables": {"entryId": pdb_id}}
    response = requests.post(GRAPHQL_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    residues: Dict[Tuple[str, int], int] = {}

    entry = data.get("data", {}).get("entry")
    if not entry:
        return LabelMappingResult(pdb_id=pdb_id, cath_id=cath_id or "", residues=residues)

    for instance in entry.get("polymer_entity_instances", []):
        asym_id = instance.get("rcsb_id")
        for feature in instance.get("rcsb_polymer_instance_feature", []):
            if feature.get("type") != "CATH":
                continue
            if cath_id and cath_id not in json.dumps(feature):
                continue
            for position in feature.get("feature_positions", []):
                beg = int(position.get("beg_seq_id"))
                end = int(position.get("end_seq_id"))
                for seq_id in range(beg, end + 1):
                    residues[(asym_id, seq_id)] = 1
    return LabelMappingResult(pdb_id=pdb_id, cath_id=cath_id or "", residues=residues)


def save_mapping(result: LabelMappingResult, output_path: Path) -> None:
    """Persist the mapping as a JSON file."""

    payload = {
        "pdb_id": result.pdb_id,
        "cath_id": result.cath_id,
        "residues": {f"{chain}:{res}": label for (chain, res), label in result.residues.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_id", type=str)
    parser.add_argument("--cath-id", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted the mapping prints to stdout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = fetch_cath_mapping(args.pdb_id, args.cath_id)
    if args.output:
        save_mapping(result, args.output)
    else:
        print(json.dumps(result.residues, indent=2))


if __name__ == "__main__":
    main()
