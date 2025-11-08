"""Render publication-ready scatter overlays of volumetric predictions."""
from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

LOGGER = logging.getLogger(__name__)

_CURRENT_RANGE: Tuple[float, float] = (0.0, 1.0)
_CURRENT_LABEL: str = ""


def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load NPZ inference data."""

    with np.load(npz_path) as data:
        coords = np.asarray(data["coords_atom"], dtype=float)
        if "probs_atom" in data:
            payload = np.asarray(data["probs_atom"], dtype=float)
        elif "pred_atom" in data:
            payload = np.asarray(data["pred_atom"], dtype=float)
        else:
            raise KeyError("NPZ missing 'probs_atom' or 'pred_atom'.")
        pdb_id = data.get("pdb_id", npz_path.stem)
        if isinstance(pdb_id, np.ndarray):
            pdb_id = pdb_id.item()
    return coords, payload, str(pdb_id)


def standardize_scalar(values: np.ndarray, mode: str) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Normalize scalar fields while retaining original range."""

    values = np.asarray(values, dtype=float)
    if mode == "label":
        clipped = np.clip(values, 0.0, 1.0)
        return clipped, (0.0, 1.0)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.zeros_like(values), (0.0, 1.0)
    vmin = float(valid.min())
    vmax = float(valid.max())
    if math.isclose(vmin, vmax):
        normed = np.zeros_like(values)
    else:
        normed = (values - vmin) / (vmax - vmin)
    return normed, (vmin, vmax)


def _savefig_atomic(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix or ".png"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False, dir=out_path.parent) as handle:
        fig.savefig(handle.name, dpi=dpi, bbox_inches="tight")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, out_path)
    LOGGER.info("Saved overlay figure: %s", out_path)


def plot_projection(
    coords: np.ndarray,
    scalars: np.ndarray,
    view: str,
    out_path: Path,
    mode: str,
    alpha: float,
    dpi: int,
) -> None:
    """Plot a specific projection of the scatter overlay."""

    global _CURRENT_RANGE
    coords = np.asarray(coords, dtype=float)
    scalars = np.asarray(scalars, dtype=float).reshape(-1)
    if coords.shape[0] != scalars.shape[0]:
        raise ValueError("Coordinate and scalar arrays must have matching lengths.")

    if view == "iso":
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=45)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    cmap = plt.cm.viridis
    norm_bounds = (0.0, 1.0)
    if mode == "label":
        cmap = ListedColormap(["#d0d0d0", "#1f77b4"])
        norm_bounds = (0, 1)

    if view == "xy":
        x_idx, y_idx = 0, 1
    elif view == "xz":
        x_idx, y_idx = 0, 2
    elif view == "yz":
        x_idx, y_idx = 1, 2
    elif view == "iso":
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=scalars,
            cmap=cmap,
            vmin=norm_bounds[0],
            vmax=norm_bounds[1],
            s=4,
            alpha=alpha,
            linewidths=0.0,
        )
    else:
        raise ValueError(f"Unsupported view '{view}'.")

    if view != "iso":
        x = coords[:, x_idx]
        y = coords[:, y_idx]
        scatter = ax.scatter(
            x,
            y,
            c=scalars,
            cmap=cmap,
            vmin=norm_bounds[0],
            vmax=norm_bounds[1],
            s=4,
            alpha=alpha,
            linewidths=0.0,
        )
        ax.set_aspect("equal", adjustable="box")

    if view == "iso":
        ranges = coords.max(axis=0) - coords.min(axis=0)
        max_range = max(ranges.max(), 1e-6)
        centers = coords.mean(axis=0)
        for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centers):
            setter(center - max_range / 2, center + max_range / 2)
        ax.set_axis_off()
    else:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if math.isclose(x_min, x_max):
            span = max(abs(x_min), 1.0) * 0.01
            x_min -= span
            x_max += span
        if math.isclose(y_min, y_max):
            span = max(abs(y_min), 1.0) * 0.01
            y_min -= span
            y_max += span
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_axis_off()

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    if mode == "label":
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Background", "Positive"])
        colorbar.set_label("Class", rotation=270, labelpad=12)
    else:
        colorbar.set_label("Value", rotation=270, labelpad=12)
    vmin, vmax = _CURRENT_RANGE
    ax.text(
        0.02,
        0.95,
        f"{view.upper()} – {_CURRENT_LABEL}\nmin={vmin:.3f} max={vmax:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.6, "pad": 2},
    )

    _savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


def _subsample(coords: np.ndarray, scalars: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    if step <= 1:
        return coords, scalars
    coords = coords[::step]
    scalars = scalars[::step]
    return coords, scalars


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for volume overlays."""

    parser = argparse.ArgumentParser(description="Render 3D volume overlays of atom predictions.")
    parser.add_argument("--npz", type=Path, required=True, help="Input NPZ file with inference outputs.")
    parser.add_argument("--out-prefix", type=Path, required=True, help="Output prefix for generated images.")
    parser.add_argument("--mode", choices=["prob", "uncertainty", "label"], default="prob", help="Scalar field to visualize.")
    parser.add_argument("--class-index", type=int, default=0, help="Class index for probability mode.")
    parser.add_argument("--subsample", type=int, default=0, help="Subsampling step; 0 disables subsampling.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Marker transparency.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    coords, _, pdb_id = load_npz(args.npz)

    with np.load(args.npz) as data:
        probs = np.asarray(data["probs_atom"], dtype=float) if "probs_atom" in data else None
        pred = np.asarray(data["pred_atom"], dtype=float) if "pred_atom" in data else None

    if args.mode == "prob":
        if probs is None:
            raise ValueError("Probability mode requires 'probs_atom' in NPZ file.")
        class_index = max(0, min(args.class_index, probs.shape[1] - 1))
        scalars = probs[:, class_index]
    elif args.mode == "uncertainty":
        if probs is None:
            raise ValueError("Uncertainty mode requires 'probs_atom'.")
        scalars = 1.0 - probs.max(axis=1)
    else:  # label
        if pred is None:
            raise ValueError("Label mode requires 'pred_atom'.")
        scalars = np.clip(pred, 0.0, 1.0)

    if args.subsample > 0:
        coords, scalars = _subsample(coords, scalars, args.subsample)

    max_points = 250_000
    if coords.shape[0] > max_points:
        step = math.ceil(coords.shape[0] / max_points)
        coords, scalars = _subsample(coords, scalars, step)

    normed, value_range = standardize_scalar(scalars, args.mode)
    global _CURRENT_RANGE, _CURRENT_LABEL
    _CURRENT_RANGE = value_range
    _CURRENT_LABEL = pdb_id

    for view in ("xy", "xz", "yz", "iso"):
        out_path = args.out_prefix.parent / f"{args.out_prefix.name}_{view}.png"
        plot_projection(coords, normed, view, out_path, args.mode, args.alpha, args.dpi)

    LOGGER.info("Generated overlays for %s", pdb_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
