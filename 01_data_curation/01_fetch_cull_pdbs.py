"""Fetch mmCIF files for a CATH superfamily from the RCSB PDB.

This script queries the RCSB PDB Search API for all entry identifiers that
match a provided CATH superfamily identifier, optionally excluding obsolete
entries. The matching mmCIF files are downloaded in parallel using Biopython
and stored within the specified output directory. A manifest describing the
result of each attempted download as well as a plaintext list of PDB entry IDs
are emitted to facilitate reproducibility.

Usage (CLI arguments override YAML config values):

    python 01_fetch_cull_pdbs.py \
      --config ./configs/cath_1.10.490.10.yaml \
      --cath-id 1.10.490.10 \
      --out-dir ./data/raw_pdbs \
      --max-workers 8 \
      --allow-obsolete false \
      --timeout 20 \
      --retries 3 \
      --sleep 0.2 \
      --log-level INFO
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep as blocking_sleep
from typing import Iterator, Optional, Sequence, Tuple

import requests
import yaml
from Bio.PDB import PDBList
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

RCSB_REQUEST_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
HOLDINGS_BASE = "https://data.rcsb.org/rest/v1/holdings"
DEFAULT_OUT_DIR = Path("./data/raw_pdbs")

SEARCH_BASES_DEFAULT = [
    "https://search.rcsb.org/rcsbsearch/v2/query",
    "https://search-east.rcsb.org/rcsbsearch/v2/query",
    "https://search-west.rcsb.org/rcsbsearch/v2/query",
]


@dataclass(frozen=True)
class FetchConfig:
    """Configuration for fetching mmCIF files for a CATH superfamily."""

    cath_id: str
    out_dir: Path
    max_workers: int
    allow_obsolete: bool
    timeout: int
    retries: int
    sleep: float
    log_level: str
    dry_run: bool
    self_test: bool
    search_base: Optional[str]


def parse_bool(value: Optional[str]) -> Optional[bool]:
    """Parse a truthy/falsy string into a boolean value."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Could not interpret boolean value from '{value}'.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Download mmCIF files for a specified CATH superfamily.",
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--cath-id", type=str, help="CATH superfamily identifier.")
    parser.add_argument(
        "--out-dir",
        type=str,
        help=f"Output directory for mmCIF files (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Number of worker threads to use for mmCIF downloads.",
    )
    parser.add_argument(
        "--allow-obsolete",
        type=str,
        help="Allow downloads of obsolete entries (true/false).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="HTTP timeout in seconds for RCSB API requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        help="Number of retries for mmCIF downloads.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        help="Base sleep duration between download attempts in seconds.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first RCSB payload without querying remote services.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Print sample payloads for diagnostics and exit.",
    )
    parser.add_argument(
        "--search-base",
        type=str,
        default="",
        help=(
            "Override the RCSB Search API base URL (default uses primary and mirror hosts)."
        ),
    )
    return parser.parse_args(argv)


def load_config(argv: Optional[Sequence[str]] = None) -> FetchConfig:
    """Load configuration by merging YAML config and CLI arguments."""

    args = parse_args(argv)
    config_file_data: dict = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config_file_data = yaml.safe_load(handle) or {}
        if not isinstance(config_file_data, dict):
            raise ValueError("Config file must contain a mapping of configuration values.")

    def from_config(key: str) -> Optional[object]:
        value = config_file_data.get(key)
        return value

    cath_id = args.cath_id or from_config("cath_id")
    if not cath_id:
        if args.self_test:
            cath_id = "1.10.490.10"
        else:
            raise ValueError("CATH ID must be provided via --cath-id or the configuration file.")

    out_dir_str = args.out_dir or from_config("out_dir")
    out_dir = Path(out_dir_str) if out_dir_str else DEFAULT_OUT_DIR

    max_workers = args.max_workers or from_config("max_workers") or 8
    allow_obsolete = parse_bool(args.allow_obsolete) if args.allow_obsolete is not None else None
    if allow_obsolete is None:
        config_allow_obsolete = from_config("allow_obsolete")
        if config_allow_obsolete is not None:
            if isinstance(config_allow_obsolete, bool):
                allow_obsolete = config_allow_obsolete
            else:
                allow_obsolete = parse_bool(str(config_allow_obsolete))
        else:
            allow_obsolete = False

    timeout = args.timeout or from_config("timeout") or 20
    retries = args.retries or from_config("retries") or 3
    sleep_seconds = args.sleep or from_config("sleep") or 0.2
    log_level = args.log_level or from_config("log_level") or "INFO"
    dry_run = bool(args.dry_run)
    if not dry_run:
        config_dry_run = from_config("dry_run")
        if config_dry_run is not None:
            if isinstance(config_dry_run, bool):
                dry_run = config_dry_run
            else:
                dry_run = bool(parse_bool(str(config_dry_run)))
    self_test = bool(args.self_test)
    if not self_test:
        config_self_test = from_config("self_test")
        if config_self_test is not None:
            if isinstance(config_self_test, bool):
                self_test = config_self_test
            else:
                self_test = bool(parse_bool(str(config_self_test)))

    search_base = (args.search_base or "").strip()
    if not search_base:
        config_search_base = from_config("search_base")
        if config_search_base is not None:
            search_base = str(config_search_base).strip()
    if not search_base:
        search_base = None

    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    timeout = int(timeout)
    if timeout <= 0:
        raise ValueError("timeout must be positive.")

    retries = int(retries)
    if retries < 0:
        raise ValueError("retries cannot be negative.")

    sleep_seconds = float(sleep_seconds)
    if sleep_seconds < 0:
        raise ValueError("sleep must be non-negative.")

    log_level_str = str(log_level).upper()
    if not hasattr(logging, log_level_str):
        raise ValueError(f"Unsupported log level: {log_level}")

    return FetchConfig(
        cath_id=str(cath_id),
        out_dir=out_dir,
        max_workers=max_workers,
        allow_obsolete=bool(allow_obsolete),
        timeout=timeout,
        retries=retries,
        sleep=sleep_seconds,
        log_level=log_level_str,
        dry_run=dry_run,
        self_test=self_test,
        search_base=search_base,
    )


def build_cath_search_payload(cath_id: str, service: str = "text") -> dict:
    """Primary Search API payload for a CATH superfamily."""

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": service,
                    "parameters": {
                        "attribute": "rcsb_polymer_instance_annotation.type",
                        "operator": "exact_match",
                        "value": "CATH",
                    },
                },
                {
                    "type": "terminal",
                    "service": service,
                    "parameters": {
                        "attribute": "rcsb_polymer_instance_annotation.annotation_lineage.id",
                        "operator": "exact_match",
                        "value": cath_id,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_content_type": ["experimental"],
        },
    }


def build_cath_search_backups(cath_id: str, service: str = "text") -> list[dict]:
    """Fallback payloads that target annotation IDs directly."""

    def payload(attribute: str, value: str) -> dict:
        return {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": service,
                        "parameters": {
                            "attribute": "rcsb_polymer_instance_annotation.type",
                            "operator": "exact_match",
                            "value": "CATH",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": service,
                        "parameters": {
                            "attribute": attribute,
                            "operator": "exact_match",
                            "value": value,
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": 10000},
                "results_content_type": ["experimental"],
            },
        }

    return [
        payload("rcsb_polymer_instance_annotation.annotation_id", f"CATH:{cath_id}"),
        payload("rcsb_polymer_instance_annotation.annotation_id", cath_id),
    ]


def build_rcsb_cath_queries(cath_id: str, service: str = "text") -> list[dict]:
    """Construct Search API payloads targeting CATH annotations."""

    payloads = [build_cath_search_payload(cath_id, service=service)]
    payloads.extend(build_cath_search_backups(cath_id, service=service))
    return payloads


def _search_with_paging(
    session: requests.Session, url: str, payload: dict, timeout: int
) -> list[str]:
    """Submit the payload and collect all entry identifiers with pagination."""

    headers = RCSB_REQUEST_HEADERS
    working_payload = json.loads(json.dumps(payload))
    request_options = working_payload.setdefault("request_options", {})
    paginate = request_options.setdefault("paginate", {"start": 0, "rows": 10000})
    rows = int(paginate.get("rows", 10000))
    start = 0
    identifiers: list[str] = []

    while True:
        paginate["start"] = start
        LOGGER.debug(
            "POST %s payload:\n%s",
            url,
            json.dumps(working_payload, indent=2, sort_keys=True),
        )
        response = session.post(url, headers=headers, json=working_payload, timeout=timeout)
        if response.status_code >= 400:
            snippet = (response.text or "")[:2000]
            raise requests.HTTPError(
                f"HTTP {response.status_code} at {url}; body[:2000]= {snippet}", response=response
            )

        data = response.json()
        result_set = data.get("result_set") or []
        identifiers.extend(
            item.get("identifier")
            for item in result_set
            if isinstance(item, dict) and item.get("identifier")
        )

        total_count = data.get("total_count")
        if not result_set:
            break
        start += rows
        if isinstance(total_count, int) and start >= total_count:
            break

    return [
        identifier.upper()
        for identifier in identifiers
        if isinstance(identifier, str) and len(identifier) == 4
    ]


def _chunked(seq: Sequence[str], n: int) -> Iterator[list[str]]:
    """Yield successive n-sized chunks from the sequence."""

    it = iter(seq)
    while True:
        block = list(itertools.islice(it, n))
        if not block:
            return
        yield block


def fetch_holdings_status_map(
    entry_ids: Sequence[str],
    timeout: int = 30,
    session: Optional[requests.Session] = None,
    use_removed_cache: bool = True,
    chunk_size: int = 300,
) -> dict[str, str]:
    """Return a mapping from entry ID to holdings status (CURRENT/OBSOLETE/etc.)."""

    if not entry_ids:
        return {}

    uppercase_ids = [entry.upper() for entry in entry_ids if entry]
    if not uppercase_ids:
        return {}

    own_session = session is None
    sess = session or requests.Session()
    logger = logging.getLogger(__name__)
    statuses: dict[str, str] = {}

    try:
        try:
            for chunk in _chunked(uppercase_ids, chunk_size):
                params = {"ids": ",".join(chunk)}
                response = sess.get(
                    f"{HOLDINGS_BASE}/status", params=params, timeout=timeout
                )
                response.raise_for_status()
                for record in response.json() or []:
                    pdb_id = str(record.get("pdb_id", "")).upper()
                    status = str(record.get("status", "")).upper() or "UNKNOWN"
                    if pdb_id:
                        statuses[pdb_id] = status
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to retrieve holdings status; falling back to removed cache only: %s",
                exc,
            )
            statuses = {}

        missing = {entry for entry in uppercase_ids if entry not in statuses}
        if missing and use_removed_cache:
            try:
                response = sess.get(
                    f"{HOLDINGS_BASE}/removed/entry_ids", timeout=timeout
                )
                response.raise_for_status()
                removed = {str(item).upper() for item in (response.json() or [])}
                for entry in missing:
                    if entry in removed:
                        statuses.setdefault(entry, "OBSOLETE")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Removed list fetch failed while classifying entries: %s", exc
                )

        for entry in uppercase_ids:
            statuses.setdefault(entry, "UNKNOWN")
        return statuses
    finally:
        if own_session:
            sess.close()


def filter_nonobsolete_entries_rest(
    entry_ids: list[str],
    timeout: int = 30,
    session: Optional[requests.Session] = None,
    use_removed_cache: bool = True,
    chunk_size: int = 300,
) -> list[str]:
    """Filter out obsolete entries using the RCSB Holdings REST API."""

    if not entry_ids:
        return []

    own_session = session is None
    sess = session or requests.Session()

    try:
        statuses = fetch_holdings_status_map(
            entry_ids,
            timeout=timeout,
            session=sess,
            use_removed_cache=use_removed_cache,
            chunk_size=chunk_size,
        )
    finally:
        if own_session:
            sess.close()

    kept = sorted(
        {
            entry.upper()
            for entry in entry_ids
            if statuses.get(entry.upper(), "UNKNOWN") != "OBSOLETE"
        }
    )

    if not kept:
        raise RuntimeError(
            "All candidate entries are obsolete or holdings/status returned empty."
        )

    return kept


def fetch_entry_ids_for_cath(
    cath_id: str,
    timeout: int,
    allow_obsolete: bool,
    search_base: Optional[str] = None,
) -> list[str]:
    """Fetch entry IDs for a CATH superfamily via the RCSB Search API."""

    session = requests.Session()
    base_url = (search_base or "").strip() or SEARCH_BASES_DEFAULT[0]
    errors: list[str] = []

    try:
        candidate_ids: list[str] = []
        for payload in build_rcsb_cath_queries(cath_id):
            try:
                candidate_ids = _search_with_paging(session, base_url, payload, timeout)
                if candidate_ids:
                    LOGGER.info(
                        "Search returned %d candidate entries for CATH %s.",
                        len(candidate_ids),
                        cath_id,
                    )
                    break
            except requests.HTTPError as exc:  # noqa: BLE001
                compact_payload = json.dumps(payload, separators=(",", ":"))
                LOGGER.warning("Search API error for %s: %s", base_url, exc)
                errors.append(f"{exc} | PAYLOAD={compact_payload}")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Search API exception for %s: %s", base_url, exc)
                errors.append(str(exc))

        if not candidate_ids:
            if errors:
                raise RuntimeError(
                    "Failed to fetch entry IDs for CATH ID "
                    f"{cath_id}. Attempts={len(errors)}. Last error: {errors[-1]}"
                )
            raise RuntimeError(
                f"No entries found for CATH ID {cath_id}. Try without obsolete filter or verify CATH ID."
            )

        unique_ids = sorted(set(candidate_ids))
        if allow_obsolete:
            return unique_ids

        filtered_ids = filter_nonobsolete_entries_rest(
            unique_ids,
            timeout=timeout,
            session=session,
        )
        if not filtered_ids:
            raise RuntimeError(
                "All candidate entries are obsolete or failed Data API filtering for "
                f"CATH ID {cath_id}."
            )
        LOGGER.info(
            "Kept %d non-obsolete entries after Data API filtering for CATH %s.",
            len(filtered_ids),
            cath_id,
        )
        return filtered_ids
    finally:
        session.close()


def ensure_out_dir(path: Path) -> None:
    """Ensure the output directory exists and is writable."""

    path.mkdir(parents=True, exist_ok=True)
    test_file = None
    try:
        test_file = NamedTemporaryFile(dir=path, delete=False)
        test_file.write(b"")
        test_file.flush()
    finally:
        if test_file is not None:
            test_file.close()
            os.unlink(test_file.name)


def existing_mmcif_path(out_dir: Path, entry_id: str) -> Optional[Path]:
    """Return the path to an already-downloaded mmCIF file if present."""

    entry_id_lower = entry_id.lower()
    candidates = [
        out_dir / f"{entry_id_lower}.cif",
        out_dir / f"{entry_id_lower}.cif.gz",
        out_dir / f"pdb{entry_id_lower}.cif",
        out_dir / f"pdb{entry_id_lower}.cif.gz",
        out_dir / "mmCIF" / f"{entry_id_lower}.cif",
        out_dir / "mmCIF" / f"{entry_id_lower}.cif.gz",
        out_dir / "mmCIF" / f"pdb{entry_id_lower}.cif",
        out_dir / "mmCIF" / f"pdb{entry_id_lower}.cif.gz",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    # Fallback: search for files matching the entry ID pattern within depth 2
    for candidate in out_dir.glob(f"**/*{entry_id_lower}*.cif*"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    return None


def http_fallback_mmcif(
    entry_id: str,
    out_dir: Path,
    overwrite_existing: bool,
    timeout: int,
) -> Optional[Path]:
    """Download mmCIF via the RCSB file service as a fallback path."""

    entry_upper = entry_id.upper()
    entry_lower = entry_id.lower()
    urls = [
        f"https://files.rcsb.org/download/{entry_upper}.cif.gz",
        f"https://files.rcsb.org/download/{entry_upper}.cif",
    ]

    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                if response.status_code != 200:
                    LOGGER.debug(
                        "HTTP fallback for %s returned %s at %s",
                        entry_upper,
                        response.status_code,
                        url,
                    )
                    continue

                suffix = ".cif.gz" if url.endswith(".gz") else ".cif"
                destination = out_dir / f"{entry_lower}{suffix}"
                if destination.exists() and not overwrite_existing:
                    return destination

                with NamedTemporaryFile(dir=out_dir, delete=False) as handle:
                    for chunk in response.iter_content(1 << 16):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    temp_path = Path(handle.name)

                os.replace(temp_path, destination)
                return destination
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "HTTP fallback attempt for %s at %s failed: %s", entry_upper, url, exc
            )

    return None


def download_mmcif(
    entry_id: str,
    out_dir: Path,
    retries: int,
    sleep: float,
    is_obsolete: bool,
    timeout: int,
    overwrite_existing: bool = False,
) -> Tuple[str, str, Optional[str]]:
    """Download a single mmCIF file with retry logic and HTTP fallback."""

    if not overwrite_existing:
        existing = existing_mmcif_path(out_dir, entry_id)
        if existing is not None:
            return entry_id, "skipped", str(existing)

    attempt = 0
    last_error: Optional[str] = None

    while attempt <= retries:
        if attempt > 0 and sleep > 0:
            blocking_sleep(sleep * (2 ** (attempt - 1)))

        try:
            pdb_list = PDBList()
            entry_code = entry_id.lower()
            path_str = pdb_list.retrieve_pdb_file(
                pdb_code=entry_code,
                file_format="mmCif",
                pdir=str(out_dir),
                overwrite=overwrite_existing,
                obsolete=is_obsolete,
            )
            if not path_str:
                raise RuntimeError("Biopython did not return a file path.")

            located = existing_mmcif_path(out_dir, entry_id)
            candidate_path = Path(path_str)
            if located is not None:
                candidate_path = located
            if not candidate_path.exists() or candidate_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Expected mmCIF file for {entry_id} not found after download."
                )

            return entry_id, "downloaded", str(candidate_path)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            LOGGER.warning(
                "Biopython download attempt %d for %s failed: %s",
                attempt + 1,
                entry_id,
                last_error,
            )
            fallback_path = http_fallback_mmcif(
                entry_id,
                out_dir,
                overwrite_existing=overwrite_existing,
                timeout=timeout,
            )
            if fallback_path is not None:
                if fallback_path.stat().st_size == 0:
                    fallback_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                    LOGGER.warning(
                        "HTTP fallback produced empty file for %s; continuing retries.",
                        entry_id,
                    )
                else:
                    LOGGER.info(
                        "Retrieved %s via HTTP fallback (%s)", entry_id, fallback_path.name
                    )
                    return entry_id, "downloaded", str(fallback_path)

        attempt += 1

    if is_obsolete:
        LOGGER.warning(
            "Obsolete entry %s could not be retrieved; coordinates may be unavailable on all mirrors.",
            entry_id,
        )

    return entry_id, "failed", last_error


def write_manifest(out_dir: Path, cath_id: str, results: list[dict]) -> None:
    """Write a JSON manifest summarizing download outcomes."""

    manifest = {
        "cath_id": cath_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "entries": results,
    }
    target = out_dir / "manifest.json"
    with NamedTemporaryFile("w", encoding="utf-8", dir=out_dir, delete=False) as handle:
        json.dump(manifest, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, target)


def write_id_list(out_dir: Path, entry_ids: Sequence[str]) -> None:
    """Write the plaintext ID list to the output directory."""

    sorted_ids = sorted(set(entry_ids))
    target = out_dir / "pdb_ids.txt"
    content = "\n".join(sorted_ids) + "\n"
    with NamedTemporaryFile("w", encoding="utf-8", dir=out_dir, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, target)


def _run_self_test() -> None:
    """Emit diagnostic payloads and ensure obsolete filters are absent."""

    test_cath_id = "1.10.490.10"
    print(f"Self-test payloads for {test_cath_id}:")
    queries = build_rcsb_cath_queries(test_cath_id)
    for index, payload in enumerate(queries, start=1):
        print(f"\nPayload {index}:")
        print(json.dumps(payload, indent=2, sort_keys=True))
        nodes = payload.get("query", {}).get("nodes", [])
        for node in nodes:
            params = node.get("parameters", {})
            attribute = params.get("attribute")
            if attribute == "rcsb_accession_info.is_obsolete":
                raise AssertionError("Search payload should not include obsolete filter nodes.")
    print("\nSelf-test completed successfully.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI script."""

    try:
        config = load_config(argv)
    except Exception as exc:  # noqa: BLE001
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        LOGGER.exception("Failed to load configuration: %s", exc)
        return 1

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config.self_test:
        _run_self_test()
        return 0

    LOGGER.info("Starting mmCIF fetch for CATH ID %s", config.cath_id)

    if config.dry_run:
        payloads = build_rcsb_cath_queries(config.cath_id)
        if not payloads:
            LOGGER.warning("Dry run generated no payloads for %s.", config.cath_id)
        for index, payload in enumerate(payloads, start=1):
            LOGGER.info(
                "Dry run payload %d:\n%s",
                index,
                json.dumps(payload, indent=2, sort_keys=True),
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        ensure_out_dir(config.out_dir)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to prepare output directory '%s': %s", config.out_dir, exc)
        return 1

    try:
        entry_ids = fetch_entry_ids_for_cath(
            config.cath_id,
            timeout=config.timeout,
            allow_obsolete=config.allow_obsolete,
            search_base=config.search_base,
        )
        LOGGER.info("Identified %d entry IDs for CATH ID %s", len(entry_ids), config.cath_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to fetch entry IDs: %s", exc)
        return 1

    obsolete_flags: dict[str, bool] = {entry_id: False for entry_id in entry_ids}
    if config.allow_obsolete and entry_ids:
        try:
            with requests.Session() as status_session:
                status_map = fetch_holdings_status_map(
                    entry_ids,
                    timeout=config.timeout,
                    session=status_session,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to classify entry statuses via holdings API: %s", exc)
            return 1

        unknown_entries: list[str] = []
        for entry_id in entry_ids:
            status = status_map.get(entry_id.upper(), "UNKNOWN")
            if status == "UNKNOWN":
                unknown_entries.append(entry_id)
            obsolete_flags[entry_id] = status == "OBSOLETE"

        current_count = sum(1 for flag in obsolete_flags.values() if not flag)
        obsolete_count = sum(1 for flag in obsolete_flags.values() if flag)
        LOGGER.info(
            "Classified entry statuses: current=%d, obsolete=%d",
            current_count,
            obsolete_count,
        )
        if unknown_entries:
            preview = ", ".join(sorted(unknown_entries)[:5])
            if len(unknown_entries) > 5:
                preview = f"{preview}, …"
            LOGGER.warning(
                "Holdings status unavailable for %d entries; treating as current: %s",
                len(unknown_entries),
                preview,
            )
    else:
        LOGGER.info(
            "Classified entry statuses: current=%d, obsolete=%d (obsolete downloads disabled)",
            len(entry_ids),
            0,
        )

    results: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_id: dict[Future[Tuple[str, str, Optional[str]]], str] = {}
            for entry_id in entry_ids:
                future = executor.submit(
                    download_mmcif,
                    entry_id,
                    config.out_dir,
                    config.retries,
                    config.sleep,
                    obsolete_flags.get(entry_id, False),
                    config.timeout,
                )
                future_to_id[future] = entry_id

            for future in tqdm(
                as_completed(future_to_id),
                total=len(future_to_id),
                desc="Downloading mmCIF",
                unit="entry",
            ):
                entry_id, status, info = future.result()
                if status == "failed":
                    LOGGER.error("Failed to download %s: %s", entry_id, info)
                    results.append(
                        {
                            "entry_id": entry_id,
                            "status": status,
                            "path": None,
                            "error": info,
                        }
                    )
                else:
                    LOGGER.info("%s %s", status.capitalize(), entry_id)
                    results.append(
                        {
                            "entry_id": entry_id,
                            "status": status,
                            "path": info,
                            "error": None,
                        }
                    )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unexpected error during downloads: %s", exc)
        return 1

    try:
        write_manifest(config.out_dir, config.cath_id, results)
        write_id_list(config.out_dir, [result["entry_id"] for result in results])
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to write output metadata: %s", exc)
        return 1

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        LOGGER.warning("Completed with %d failed downloads.", len(failed))
    else:
        LOGGER.info("Completed successfully with all downloads processed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
