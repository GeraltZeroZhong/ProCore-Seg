"""SIFTS-derived CATH label mapper for PDB entries.

This module provides a CLI that queries the RCSB GraphQL endpoint for
SIFTS-based CATH annotations and produces per-residue binary maps keyed by
author chain identifiers. The tool is deterministic and can optionally restrict
annotations to a specific CATH superfamily.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Sequence, Tuple

import requests

LOGGER = logging.getLogger(__name__)

GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"
REQUEST_ID_HEADERS = ("x-request-id", "x-trace-id", "x-request-guid")


def normalize_pdb_id(pdb_id: str) -> str:
    """Normalize and validate a PDB identifier.

    Parameters
    ----------
    pdb_id:
        Four-character PDB identifier. The value is case-insensitive.

    Returns
    -------
    str
        Uppercase PDB identifier.

    Raises
    ------
    ValueError
        If the identifier is empty or not four characters long.
    """

    if not pdb_id:
        raise ValueError("PDB ID must be provided")

    normalized = pdb_id.strip().upper()
    if len(normalized) != 4 or not normalized.isalnum():
        raise ValueError("PDB ID must be a 4-character alphanumeric string")

    return normalized


def build_graphql_query() -> str:
    """Return the GraphQL query for fetching polymer entity instances."""

    return (
        "query($entry_id: String!) {\n"
        "  entry(entry_id: $entry_id) {\n"
        "    polymer_entity_instances {\n"
        "      rcsb_polymer_entity_instance_container_identifiers {\n"
        "        auth_asym_id\n"
        "        label_asym_id\n"
        "        polymer_entity_id\n"
        "      }\n"
        "      rcsb_polymer_instance_feature {\n"
        "        type\n"
        "        name\n"
        "        description\n"
        "        provenance_source\n"
        "        feature_id\n"
        "        feature_positions {\n"
        "          beg_seq_id\n"
        "          end_seq_id\n"
        "        }\n"
        "        related_database_citations {\n"
        "          database_name\n"
        "          accession\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def fetch_instance_features(pdb_id: str, timeout: int) -> List[Dict[str, object]]:
    """Fetch polymer instance features for a PDB entry from RCSB GraphQL."""

    normalized_id = normalize_pdb_id(pdb_id)
    query = build_graphql_query()
    payload = {"query": query, "variables": {"entry_id": normalized_id}}

    LOGGER.debug("GraphQL query payload: %s", json.dumps(payload, sort_keys=True))

    try:
        response = requests.post(GRAPHQL_ENDPOINT, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise RuntimeError(f"GraphQL request failed: {exc}") from exc

    request_id = None
    for header in REQUEST_ID_HEADERS:
        if header in response.headers:
            request_id = response.headers[header]
            break
    if request_id:
        LOGGER.debug("GraphQL request id: %s", request_id)

    if response.status_code != requests.codes.ok:
        raise RuntimeError(
            f"GraphQL request failed with status {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("GraphQL response was not valid JSON") from exc

    if "errors" in payload and payload["errors"]:
        first_error = payload["errors"][0]
        message = first_error.get("message", "Unknown GraphQL error")
        raise RuntimeError(f"GraphQL response contained errors: {message}")

    data = payload.get("data")
    if not data or not isinstance(data, dict):
        raise RuntimeError("GraphQL response missing 'data' field")

    entry = data.get("entry")
    if not entry:
        raise RuntimeError(f"Entry '{normalized_id}' was not found in RCSB data")

    instances = entry.get("polymer_entity_instances")
    if not instances:
        raise RuntimeError(
            f"Entry '{normalized_id}' does not contain polymer entity instances"
        )

    normalized_instances: List[Dict[str, object]] = []

    for raw_instance in instances:
        identifiers = raw_instance.get(
            "rcsb_polymer_entity_instance_container_identifiers", {}
        )
        auth_asym_id = identifiers.get("auth_asym_id")
        if not auth_asym_id:
            LOGGER.debug(
                "Skipping instance without auth_asym_id: %s", json.dumps(raw_instance)
            )
            continue

        raw_features = raw_instance.get("rcsb_polymer_instance_feature") or []
        normalized_features: List[Dict[str, object]] = []

        for feature in raw_features:
            feature_positions = feature.get("feature_positions") or []
            positions: List[Tuple[int, int]] = []
            for pos in feature_positions:
                try:
                    beg = int(pos["beg_seq_id"])
                    end = int(pos["end_seq_id"])
                except (KeyError, TypeError, ValueError):
                    LOGGER.debug("Skipping malformed feature position: %s", pos)
                    continue
                if beg > end:
                    beg, end = end, beg
                positions.append((beg, end))

            if not positions:
                continue

            normalized_features.append(
                {
                    "type": feature.get("type"),
                    "name": feature.get("name"),
                    "description": feature.get("description"),
                    "positions": positions,
                    "related_database_citations": feature.get(
                        "related_database_citations"
                    )
                    or [],
                }
            )

        normalized_instances.append(
            {
                "auth_asym_id": str(auth_asym_id),
                "features": normalized_features,
            }
        )

        LOGGER.debug(
            "Instance %s contains %d usable features",
            auth_asym_id,
            len(normalized_features),
        )

    return normalized_instances


def _feature_matches_superfamily(feature: Dict[str, object], cath_superfamily: str) -> bool:
    """Check if a feature matches the target CATH superfamily."""

    target = cath_superfamily.strip()
    if not target:
        return True

    name = feature.get("name")
    if isinstance(name, str) and target in name:
        return True

    description = feature.get("description")
    if isinstance(description, str) and target in description:
        return True

    citations = feature.get("related_database_citations") or []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        db_name = citation.get("database_name")
        accession = citation.get("accession")
        if (
            isinstance(db_name, str)
            and "CATH" in db_name.upper()
            and isinstance(accession, str)
            and accession.strip() == target
        ):
            return True

    return False


def filter_cath_features(
    instances: Sequence[Dict[str, object]], cath_superfamily: Optional[str]
) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Filter features to retain only CATH annotations and optional superfamily."""

    cath_ranges: Dict[str, List[Tuple[int, int]]] = {}
    for instance in instances:
        auth_asym_id = instance.get("auth_asym_id")
        if not isinstance(auth_asym_id, str):
            continue

        features = instance.get("features") or []
        for feature in features:
            if feature.get("type") != "CATH":
                continue

            if cath_superfamily and not _feature_matches_superfamily(
                feature, cath_superfamily
            ):
                continue

            positions = feature.get("positions") or []
            valid_positions = [pos for pos in positions if isinstance(pos, tuple)]
            if not valid_positions:
                continue

            cath_ranges.setdefault(auth_asym_id, []).extend(valid_positions)

    result: List[Tuple[str, List[Tuple[int, int]]]] = []
    for auth_asym_id, ranges in cath_ranges.items():
        sorted_ranges = sorted(ranges, key=lambda item: (item[0], item[1]))
        result.append((auth_asym_id, sorted_ranges))

    result.sort(key=lambda item: item[0])
    return result


def expand_ranges_to_map(
    ranged: Sequence[Tuple[str, Sequence[Tuple[int, int]]]]
) -> Dict[Tuple[str, int, str], int]:
    """Expand inclusive ranges to a per-residue binary map."""

    mapping: Dict[Tuple[str, int, str], int] = {}
    for auth_asym_id, ranges in ranged:
        for beg, end in ranges:
            for seq_id in range(int(beg), int(end) + 1):
                mapping[(auth_asym_id, seq_id, "")] = 1

    return mapping


def serialize_map(mapping: Dict[Tuple[str, int, str], int]) -> List[str]:
    """Serialize the per-residue map keys to compact string representation."""

    entries: List[str] = []
    for chain, seq_id, ins_code in sorted(mapping.keys(), key=lambda item: (item[0], item[1], item[2])):
        entries.append(f"{chain}|{seq_id}|{ins_code}")
    return entries


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON payload atomically to the target path."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2, sort_keys=True)
        tmp_file.flush()
        tmp_name = tmp_file.name

    Path(tmp_name).replace(path)


def build_payload(
    pdb_id: str, cath_superfamily: Optional[str], mapping: Dict[Tuple[str, int, str], int]
) -> Dict[str, object]:
    """Construct a JSON-serializable payload for the given mapping."""

    entries = serialize_map(mapping)
    generated_at = datetime.now(timezone.utc).isoformat()

    payload: Dict[str, object] = {
        "pdb_id": normalize_pdb_id(pdb_id),
        "cath_superfamily": cath_superfamily,
        "generated_at": generated_at,
        "key_format": "auth_asym_id|auth_seq_id|auth_ins_code",
        "entries": entries,
    }
    return payload


def _setup_logging(log_level: str) -> None:
    """Configure logging according to the requested level."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Generate per-residue CATH labels from SIFTS annotations.")
    parser.add_argument("--pdb-id", required=True, help="4-character PDB identifier")
    parser.add_argument("--out", help="Output JSON file path")
    parser.add_argument("--cath-superfamily", help="CATH superfamily identifier to filter on")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for GraphQL requests (seconds)")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    _setup_logging(args.log_level)

    try:
        normalized_id = normalize_pdb_id(args.pdb_id)
        instances = fetch_instance_features(normalized_id, args.timeout)
        cath_features = filter_cath_features(instances, args.cath_superfamily)
        mapping = expand_ranges_to_map(cath_features)

        if not mapping:
            if args.cath_superfamily:
                raise RuntimeError(
                    f"No residues annotated with CATH superfamily {args.cath_superfamily} for entry {normalized_id}"
                )
            raise RuntimeError(f"No CATH-annotated residues found for entry {normalized_id}")

        chains = sorted({chain for chain, _, _ in mapping.keys()})
        LOGGER.info(
            "Annotated %d residues across chains %s%s",
            len(mapping),
            ",".join(chains) if chains else "-",
            f" using superfamily {args.cath_superfamily}" if args.cath_superfamily else "",
        )

        if args.out:
            payload = build_payload(normalized_id, args.cath_superfamily, mapping)
            write_json(Path(args.out), payload)
        else:
            entries = serialize_map(mapping)
            print(json.dumps(entries, separators=(",", ":")))

        return 0
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
