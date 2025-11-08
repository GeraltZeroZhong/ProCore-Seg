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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re
import string

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
        "          beg_ins_code\n"
        "          end_seq_id\n"
        "          end_ins_code\n"
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


def _coerce_seq_id(value: object) -> Optional[str]:
    """Return a stripped string representation of a sequence identifier."""

    if value is None:
        return None
    seq = str(value).strip()
    return seq or None


def _extract_seq_number(seq_id: Optional[str]) -> Optional[int]:
    """Extract the integer portion of a sequence identifier if present."""

    if not seq_id:
        return None

    match = re.match(r"^[+-]?\d+", seq_id)
    if not match:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def _extract_ins_code(seq_id: Optional[str], ins_code: Optional[object]) -> str:
    """Determine the insertion code for a residue."""

    if isinstance(ins_code, str) and ins_code.strip():
        return ins_code.strip()

    if not seq_id:
        return ""

    match = re.match(r"^[+-]?\d+(?P<code>[A-Za-z]*)$", seq_id)
    if match:
        return match.group("code") or ""

    return ""


_INS_CODE_ORDER = [""] + list(string.ascii_uppercase)


def _expand_ins_codes(start: str, end: str) -> List[str]:
    """Return the inclusive range of insertion codes between start and end."""

    start = start or ""
    end = end or ""

    if start == end:
        return [start]

    if start in _INS_CODE_ORDER and end in _INS_CODE_ORDER:
        start_idx = _INS_CODE_ORDER.index(start)
        end_idx = _INS_CODE_ORDER.index(end)
        step = 1 if start_idx <= end_idx else -1
        return [
            _INS_CODE_ORDER[idx]
            for idx in range(start_idx, end_idx + step, step)
        ]

    # Fallback for uncommon multi-character or non-standard codes.
    if start < end:
        return [start, end]
    return [start, end]


def _iter_residue_positions(
    beg_seq_id: object,
    beg_ins_code: object,
    end_seq_id: object,
    end_ins_code: object,
) -> Iterable[Tuple[str, str]]:
    """Yield residue identifiers spanning a feature range inclusively."""

    beg_seq = _coerce_seq_id(beg_seq_id)
    end_seq = _coerce_seq_id(end_seq_id)
    if beg_seq is None or end_seq is None:
        LOGGER.debug(
            "Skipping feature range with missing sequence identifiers: %s -> %s",
            beg_seq_id,
            end_seq_id,
        )
        return

    beg_num = _extract_seq_number(beg_seq)
    end_num = _extract_seq_number(end_seq)
    beg_code = _extract_ins_code(beg_seq, beg_ins_code)
    end_code = _extract_ins_code(end_seq, end_ins_code)

    if beg_num is None or end_num is None:
        # Unable to determine numeric span; yield the boundary residues only.
        yield (beg_seq, beg_code)
        if beg_seq != end_seq or beg_code != end_code:
            yield (end_seq, end_code)
        return

    if beg_num > end_num:
        beg_num, end_num = end_num, beg_num
        beg_code, end_code = end_code, beg_code

    if beg_num == end_num:
        for code in _expand_ins_codes(beg_code, end_code):
            yield (str(beg_num), code)
        return

    # Start residue
    yield (str(beg_num), beg_code)

    # Interior residues without insertion codes
    for seq_num in range(beg_num + 1, end_num):
        yield (str(seq_num), "")

    # End residue base position (without insertion code)
    if end_code:
        yield (str(end_num), "")

    # End residue including insertion code when present
    yield (str(end_num), end_code)


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
            positions: List[Tuple[str, str, str, str]] = []
            for pos in feature_positions:
                beg_seq = _coerce_seq_id(pos.get("beg_seq_id"))
                end_seq = _coerce_seq_id(pos.get("end_seq_id"))
                if beg_seq is None or end_seq is None:
                    LOGGER.debug("Skipping malformed feature position: %s", pos)
                    continue
                beg_ins = _extract_ins_code(beg_seq, pos.get("beg_ins_code"))
                end_ins = _extract_ins_code(end_seq, pos.get("end_ins_code"))
                positions.append((beg_seq, beg_ins, end_seq, end_ins))

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
) -> List[Tuple[str, List[Tuple[str, str, str, str]]]]:
    """Filter features to retain only CATH annotations and optional superfamily."""

    cath_ranges: Dict[str, List[Tuple[str, str, str, str]]] = {}
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
            valid_positions = [
                pos
                for pos in positions
                if isinstance(pos, tuple)
                and len(pos) == 4
                and pos[0]
                and pos[2]
            ]
            if not valid_positions:
                continue

            cath_ranges.setdefault(auth_asym_id, []).extend(valid_positions)

    def _range_sort_key(span: Tuple[str, str, str, str]) -> Tuple[object, ...]:
        beg_seq, beg_ins, end_seq, end_ins = span
        beg_num = _extract_seq_number(beg_seq)
        end_num = _extract_seq_number(end_seq)
        return (
            beg_num if beg_num is not None else 10**9,
            beg_ins or "",
            end_num if end_num is not None else 10**9,
            end_ins or "",
            beg_seq,
            end_seq,
        )

    result: List[Tuple[str, List[Tuple[str, str, str, str]]]] = []
    for auth_asym_id, ranges in cath_ranges.items():
        sorted_ranges = sorted(ranges, key=_range_sort_key)
        result.append((auth_asym_id, sorted_ranges))

    result.sort(key=lambda item: item[0])
    return result


def expand_ranges_to_map(
    ranged: Sequence[Tuple[str, Sequence[Tuple[str, str, str, str]]]]
) -> Dict[Tuple[str, str, str], int]:
    """Expand inclusive ranges to a per-residue binary map."""

    mapping: Dict[Tuple[str, str, str], int] = {}
    for auth_asym_id, ranges in ranged:
        for beg_seq, beg_ins, end_seq, end_ins in ranges:
            for seq_id, ins_code in _iter_residue_positions(
                beg_seq, beg_ins, end_seq, end_ins
            ):
                mapping[(auth_asym_id, seq_id, ins_code)] = 1

    return mapping


def serialize_map(mapping: Dict[Tuple[str, str, str], int]) -> List[str]:
    """Serialize the per-residue map keys to compact string representation."""

    entries: List[str] = []
    def _map_sort_key(item: Tuple[str, str, str]) -> Tuple[object, ...]:
        chain, seq_id, ins_code = item
        seq_num = _extract_seq_number(seq_id)
        return (
            chain,
            0 if seq_num is not None else 1,
            seq_num if seq_num is not None else seq_id,
            ins_code or "",
        )

    for chain, seq_id, ins_code in sorted(mapping.keys(), key=_map_sort_key):
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
    pdb_id: str, cath_superfamily: Optional[str], mapping: Dict[Tuple[str, str, str], int]
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
