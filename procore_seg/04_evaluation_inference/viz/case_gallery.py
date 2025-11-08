"""Generate Markdown galleries for evaluation case studies."""
from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def read_per_protein_table(path: Path) -> List[Dict[str, str | float | int]]:
    """Read the per-protein CSV table into a list of dictionaries."""

    rows: List[Dict[str, str | float | int]] = []
    if pd is not None:
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            rows.append({key: record[key] for key in record})
        return rows

    import csv

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, str | float | int] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    if value.isdigit():
                        parsed[key] = int(value)
                    else:
                        parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def select_top_bottom(rows: List[Dict[str, str | float | int]], metric: str, k: int) -> Tuple[List[Dict[str, str | float | int]], List[Dict[str, str | float | int]]]:
    """Select the top-K and bottom-K entries by the specified metric."""

    def _metric_value(row: Dict[str, str | float | int]) -> float:
        value = row.get(metric, float("nan"))
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    filtered: List[Dict[str, str | float | int]] = []
    for row in rows:
        score = _metric_value(row)
        if not math.isnan(score):
            filtered.append(row)
    if not filtered:
        raise ValueError(f"Metric '{metric}' not found in table.")

    sorted_rows = sorted(filtered, key=_metric_value)
    bottom = sorted_rows[:k]
    top = list(reversed(sorted_rows[-k:]))
    return top, bottom


def _find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    return start


def maybe_render_thumbnail(
    pdb_id: str,
    npz_dir: Path,
    h5_dir: Path,
    thumb_dir: Path,
    pymol_bin: str,
    dpi: int,
) -> Optional[Path]:
    """Render a PyMOL thumbnail if possible."""

    if shutil.which(pymol_bin) is None:
        LOGGER.warning("PyMOL executable '%s' not found; skipping thumbnails.", pymol_bin)
        return None

    npz_path = npz_dir / f"{pdb_id}.npz"
    if not npz_path.exists():
        LOGGER.warning("NPZ file missing for %s; cannot render thumbnail.", pdb_id)
        return None

    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{pdb_id}.png"

    repo_root = _find_repo_root(Path(__file__).resolve())
    export_script = repo_root / "04_evaluation_inference" / "inference" / "export_pymol.py"
    seg_pdb = npz_dir / f"{pdb_id}.seg.pdb"

    if not seg_pdb.exists() and export_script.exists():
        python_bin = shutil.which("python") or shutil.which("python3") or "python"
        cmd = [
            python_bin,
            os.fspath(export_script),
            "--npz",
            os.fspath(npz_path),
            "--out",
            os.fspath(seg_pdb),
        ]
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            LOGGER.warning("Failed to export PDB for %s: %s", pdb_id, result.stderr.strip())
            return None

    if not seg_pdb.exists():
        LOGGER.warning("Segmented PDB missing for %s; skipping thumbnail.", pdb_id)
        return None

    pymol_script = f"""
load {seg_pdb.as_posix()}
hide everything
show cartoon
spectrum b, white_blue, minimum=0, maximum=1
set ray_trace_mode, 1
png {thumb_path.as_posix()}, dpi={dpi}
quit
"""
    proc = subprocess.run(
        [pymol_bin, "-cq"],
        input=pymol_script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        LOGGER.warning("PyMOL rendering failed for %s: %s", pdb_id, proc.stderr.strip())
        return None
    return thumb_path


def render_markdown(
    top_rows: List[Dict[str, str | float | int]],
    bottom_rows: List[Dict[str, str | float | int]],
    npz_dir: Path,
    h5_dir: Path,
    thumb_dir: Optional[Path],
    metric: str,
) -> str:
    """Render the Markdown gallery string."""

    def _format_row(row: Dict[str, str | float | int]) -> str:
        pdb_id = str(row.get("pdb_id", "unknown"))
        metric_val = row.get(metric, "n/a")
        metric_str = f"{float(metric_val):.4f}" if isinstance(metric_val, (int, float)) else str(metric_val)
        atoms = row.get("num_atoms", "")
        voxels = row.get("num_voxels", "")
        npz_link = (npz_dir / f"{pdb_id}.npz").as_posix()
        h5_link = (h5_dir / f"{pdb_id}.h5").as_posix()
        thumb_cell = ""
        thumb_key = row.get("thumbnail_path")
        if thumb_key and thumb_dir is not None:
            thumb_path = thumb_dir / thumb_key  # type: ignore[arg-type]
            if thumb_path.exists():
                thumb_cell = f"![{pdb_id}]({thumb_path.as_posix()})"
        return (
            f"| {pdb_id} | {metric_str} | {atoms} | {voxels} | "
            f"[npz]({npz_link}) | [h5]({h5_link}) | {thumb_cell} |"
        )

    sections = []
    header = f"| PDB ID | {metric} | Atoms | Voxels | NPZ | H5 | Thumbnail |"
    divider = "| --- | ---: | ---: | ---: | --- | --- | --- |"

    for title, entries in (("Top-K", top_rows), ("Bottom-K", bottom_rows)):
        sections.append(f"## {title}")
        sections.append("")
        sections.append(header)
        sections.append(divider)
        for row in entries:
            sections.append(_format_row(row))
        sections.append("")

    return "\n".join(sections).strip() + "\n"


def atomic_write_text(text: str, out_md: Path) -> None:
    """Write text to disk atomically."""

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=out_md.parent, suffix=out_md.suffix or ".md") as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, out_md)
    LOGGER.info("Gallery written to %s", out_md)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the gallery generator."""

    parser = argparse.ArgumentParser(description="Create Markdown galleries for evaluation cases.")
    parser.add_argument("--per-protein", type=Path, required=True, help="Path to per-protein CSV table.")
    parser.add_argument("--npz-dir", type=Path, required=True, help="Directory containing NPZ inference files.")
    parser.add_argument("--h5-dir", type=Path, required=True, help="Directory containing H5 ground-truth files.")
    parser.add_argument("--out-md", type=Path, required=True, help="Destination Markdown file.")
    parser.add_argument("--k", type=int, default=10, help="Number of cases to include per section.")
    parser.add_argument("--metric", default="iou", help="Metric column to sort by.")
    parser.add_argument("--render-thumbnails", default="false", help="Render PyMOL thumbnails (true/false).")
    parser.add_argument("--pymol-bin", default="pymol", help="PyMOL executable name.")
    parser.add_argument("--thumb-dir", type=Path, default=Path("thumbs"), help="Directory for thumbnail images.")
    parser.add_argument("--dpi", type=int, default=180, help="Thumbnail DPI when rendering.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    rows = read_per_protein_table(args.per_protein)
    top_rows, bottom_rows = select_top_bottom(rows, args.metric, args.k)

    render = _parse_bool(str(args.render_thumbnails))
    thumb_dir = args.thumb_dir if render else None

    if render and thumb_dir is not None:
        for row in top_rows + bottom_rows:
            pdb_id = str(row.get("pdb_id", ""))
            if not pdb_id:
                continue
            thumb_path = maybe_render_thumbnail(pdb_id, args.npz_dir, args.h5_dir, thumb_dir, args.pymol_bin, args.dpi)
            if thumb_path is not None:
                row["thumbnail_path"] = thumb_path.name

    markdown = render_markdown(top_rows, bottom_rows, args.npz_dir, args.h5_dir, thumb_dir, args.metric)
    atomic_write_text(markdown, args.out_md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
