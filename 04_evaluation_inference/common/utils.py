from __future__ import annotations

"""General-purpose utilities shared across evaluation and inference code."""

import csv
import json
import logging
import os
import platform
import random
import tempfile
import time
from functools import wraps
from hashlib import sha1 as _sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic behaviour across supported libraries."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    try:  # Torch is optional; mirror NumPy's RNG where possible.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - backend specific
            pass
    except ImportError:  # pragma: no cover - optional
        pass


def env_snapshot() -> Dict[str, str]:
    """Capture a lightweight snapshot of the execution environment."""

    snapshot: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__  # type: ignore[attr-defined]
    except ImportError:
        snapshot["torch"] = "not-installed"
    try:
        import MinkowskiEngine as me  # type: ignore

        snapshot["minkowski"] = getattr(me, "__version__", "unknown")
    except ImportError:
        snapshot["minkowski"] = "not-installed"
    return snapshot


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""

    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_bytes(data: bytes, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    tmp_fd: Optional[int] = None
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent)) as tmp_file:
            tmp_fd = tmp_file.fileno()
            tmp_path = Path(tmp_file.name)
            tmp_file.write(data)
        os.replace(str(tmp_path), str(out_path))
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_text(text: str, out_path: Path) -> None:
    """Write text to disk atomically using UTF-8 encoding."""

    _atomic_write_bytes(text.encode("utf-8"), out_path)


def atomic_write_json(obj: Dict[str, Any], out_path: Path) -> None:
    """Write JSON to disk atomically with deterministic formatting."""

    data = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(data, out_path)


def load_yaml_or_none(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a YAML file into a dictionary if available."""

    if path is None:
        return None
    if not path.exists():
        return None
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML files but is not installed")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return None
    if not isinstance(loaded, dict):
        raise ValueError("YAML content must be a mapping")
    return loaded


def safe_glob(dir_path: Path, pattern: str) -> List[Path]:
    """Return a deterministically ordered list of paths matching a glob pattern."""

    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {dir_path}")
    matches = sorted(dir_path.glob(pattern), key=lambda p: p.name.upper())
    return [p for p in matches if p.exists()]


def read_ids_file(path: Path) -> List[str]:
    """Read newline-separated identifiers, normalise, deduplicate, and sort."""

    if not path.exists():
        raise ValueError(f"IDs file does not exist: {path}")
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                ids.append(token.upper())
    unique_ids = sorted(set(ids))
    return unique_ids


def chunked(iterable: Sequence[Any] | Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield successive chunks of the provided iterable."""

    if size <= 0:
        raise ValueError("size must be positive")
    if isinstance(iterable, Sequence):
        for start in range(0, len(iterable), size):
            yield list(iterable[start : start + size])
        return
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def timed(func: Callable[..., Any]) -> Callable[..., Tuple[Any, float]]:
    """Decorator that measures execution time of the wrapped function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return wrapper


def sha1_of_file(path: Path, block_size: int = 1 << 20) -> str:
    """Compute the SHA1 hash of a file using streaming reads."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    hasher = _sha1()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def log_init(level: str = "INFO") -> None:
    """Initialise basic logging configuration if not already configured."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def json_load(path: Path) -> Dict[str, Any]:
    """Load a JSON file encoded as UTF-8 and return a mapping."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def csv_write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a CSV file with deterministic ordering."""

    if not rows:
        atomic_write_text("", path)
        return
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in sorted(row.keys()):
            if key not in seen:
                seen.add(key)
                columns.append(key)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), newline="", encoding="utf-8") as tmp_file:
        writer = csv.DictWriter(tmp_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
        tmp_path = Path(tmp_file.name)
    os.replace(str(tmp_path), str(path))


def csv_read_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries with string values."""

    if not path.exists():
        raise ValueError(f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        content = handle.read()
        if not content:
            return []
        handle.seek(0)
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


if __name__ == "__main__":
    log_init()
    temp_dir = Path(tempfile.mkdtemp())
    try:
        set_seed(42)
        LOGGER.info("Env snapshot: %s", env_snapshot())
        txt_path = temp_dir / "example.txt"
        json_path = temp_dir / "example.json"
        csv_path = temp_dir / "example.csv"
        atomic_write_text("hello", txt_path)
        atomic_write_json({"a": 1, "b": 2}, json_path)
        csv_write_rows(csv_path, [{"col1": 1, "col2": "x"}, {"col2": "y", "col3": 3}])
        LOGGER.info("Read text: %s", txt_path.read_text(encoding="utf-8"))
        LOGGER.info("Read json: %s", json_load(json_path))
        LOGGER.info("Read csv: %s", csv_read_rows(csv_path))
    finally:
        for child in temp_dir.iterdir():
            child.unlink()
        temp_dir.rmdir()
