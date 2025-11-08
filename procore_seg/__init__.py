"""ProCore-Seg namespace package for CLI utilities."""

from __future__ import annotations

from pathlib import Path

__all__ = ["VERSION"]

VERSION = "0.1.0"

# Extend package search path to include repository root for legacy module layout.
_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
if str(_repo_root) not in __path__:
    __path__.append(str(_repo_root))
