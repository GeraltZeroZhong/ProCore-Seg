"""Visualization utilities for evaluation plots."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # Optional dependency.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional.
    pd = None  # type: ignore

try:  # Optional dependency.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional.
    yaml = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StyleConfig:
    """Container for plotting style values."""

    font_size: float
    line_width: float
    alpha_fill: float
    figure_size: Tuple[float, float]
    colors: Dict[str, str]


_DEFAULT_STYLE = StyleConfig(
    font_size=11.0,
    line_width=2.0,
    alpha_fill=0.2,
    figure_size=(6.0, 4.0),
    colors={
        "curve": "#1f77b4",
        "baseline": "#7f7f7f",
        "positive": "#1f77b4",
        "negative": "#d62728",
        "quantile": "#2ca02c",
        "hist": "#1f77b4",
        "pareto": "#ff7f0e",
    },
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_table(per_protein_path: Path) -> Dict[str, np.ndarray]:
    """Load the per-protein CSV table.

    Parameters
    ----------
    per_protein_path:
        Path to the CSV file containing per-protein metrics.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from column name to numpy array of column values.
    """

    if pd is not None:
        LOGGER.debug("Loading per-protein table via pandas: %s", per_protein_path)
        frame = pd.read_csv(per_protein_path)
        return {col: frame[col].to_numpy() for col in frame.columns}

    LOGGER.debug("Loading per-protein table via csv module: %s", per_protein_path)
    import csv

    with per_protein_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns: Dict[str, List[float]] = {}
        for row in reader:
            for key, value in row.items():
                if value is None:
                    continue
                try:
                    parsed: float = float(value)
                except ValueError:
                    parsed = float("nan")
                columns.setdefault(key, []).append(parsed)
    return {key: np.asarray(values) for key, values in columns.items()}


def load_style(style_yaml: Optional[Path]) -> Dict[str, float | str | int]:
    """Load plotting style information, falling back to defaults.

    Parameters
    ----------
    style_yaml:
        Optional path to a YAML configuration file describing plotting
        aesthetics.

    Returns
    -------
    Dict[str, float | str | int]
        A dictionary compatible with :class:`StyleConfig`.
    """

    style = _DEFAULT_STYLE
    if style_yaml is not None and style_yaml.exists():
        if yaml is None:
            LOGGER.warning("YAML style file provided but PyYAML is not available.")
        else:
            LOGGER.info("Loading style configuration: %s", style_yaml)
            with style_yaml.open("r", encoding="utf-8") as handle:
                raw_cfg = yaml.safe_load(handle) or {}
            colors = dict(style.colors)
            colors.update(raw_cfg.get("colors", {}))
            figure_size = tuple(raw_cfg.get("figure_size", style.figure_size))
            style = StyleConfig(
                font_size=float(raw_cfg.get("font_size", style.font_size)),
                line_width=float(raw_cfg.get("line_width", style.line_width)),
                alpha_fill=float(raw_cfg.get("alpha_fill", style.alpha_fill)),
                figure_size=(float(figure_size[0]), float(figure_size[1])),
                colors=colors,
            )
    return {
        "font_size": style.font_size,
        "line_width": style.line_width,
        "alpha_fill": style.alpha_fill,
        "figure_size": style.figure_size,
        "colors": style.colors,
    }


def savefig_atomic(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    """Persist a Matplotlib figure using an atomic write."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix or ".png"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False, dir=out_path.parent) as handle:
        tmp_path = Path(handle.name)
        fig.savefig(handle.name, dpi=dpi, bbox_inches="tight")
    os.replace(tmp_path, out_path)
    LOGGER.info("Saved figure: %s", out_path)


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------

def pr_curve_from_scores(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve data from scores."""

    y_true = np.asarray(y_true).astype(bool)
    prob_pos = np.asarray(prob_pos)
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)
    positives = y_true.sum()
    for idx, thr in enumerate(thresholds):
        predicted = prob_pos >= thr
        tp = np.logical_and(predicted, y_true).sum()
        fp = np.logical_and(predicted, ~y_true).sum()
        fn = positives - tp
        precision[idx] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall[idx] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return thresholds, precision, recall


def _setup_axes(style: Dict, projection: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=style["figure_size"])
    if projection is None:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection=projection)
    ax.tick_params(labelsize=style["font_size"] * 0.9)
    return fig, ax


def plot_pr_curve(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    out_path: Path,
    label: str,
    style: Dict,
    dpi: int,
) -> None:
    """Plot a precision-recall curve and save the figure."""

    thresholds, precision, recall = pr_curve_from_scores(y_true, prob_pos)
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    pr_auc = np.trapz(precision_sorted, recall_sorted)
    positive_rate = float(y_true.mean())

    fig, ax = _setup_axes(style)
    ax.plot(
        recall,
        precision,
        color=style["colors"].get("curve", "#1f77b4"),
        linewidth=style["line_width"],
        label=f"{label} (AUC={pr_auc:.3f})",
    )
    ax.set_xlabel("Recall", fontsize=style["font_size"])
    ax.set_ylabel("Precision", fontsize=style["font_size"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Precision-Recall Curve – {label}", fontsize=style["font_size"] * 1.1)
    ax.legend(fontsize=style["font_size"] * 0.9, loc="lower left")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.text(
        0.99,
        0.02,
        f"Pos. rate: {positive_rate:.3f}",
        ha="right",
        va="bottom",
        fontsize=style["font_size"] * 0.85,
        transform=ax.transAxes,
    )
    savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


def plot_reliability_diagram(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    n_bins: int,
    out_path: Path,
    label: str,
    style: Dict,
    dpi: int,
) -> None:
    """Create a reliability diagram with ECE annotation."""

    y_true = np.asarray(y_true).astype(bool)
    prob_pos = np.asarray(prob_pos)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob_pos, bins, right=True)
    accuracy = []
    confidence = []
    weights = []
    total = len(prob_pos)
    for idx in range(1, len(bins)):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        conf = prob_pos[mask].mean()
        acc = y_true[mask].mean()
        weight = mask.sum() / total
        confidence.append(conf)
        accuracy.append(acc)
        weights.append(weight)
    ece = float(np.sum(np.abs(np.asarray(accuracy) - np.asarray(confidence)) * np.asarray(weights)))

    fig, ax = _setup_axes(style)
    ax.bar(
        confidence,
        accuracy,
        width=1.0 / n_bins,
        color=style["colors"].get("curve", "#1f77b4"),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], color=style["colors"].get("baseline", "#7f7f7f"), linestyle="--", linewidth=1.0)
    ax.set_xlabel("Confidence", fontsize=style["font_size"])
    ax.set_ylabel("Accuracy", fontsize=style["font_size"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Reliability Diagram – {label}", fontsize=style["font_size"] * 1.1)
    ax.text(
        0.95,
        0.05,
        f"ECE = {ece:.3f}",
        ha="right",
        va="bottom",
        fontsize=style["font_size"] * 0.85,
        transform=ax.transAxes,
    )
    savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


def _compute_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def plot_density_stratified_iou(
    density: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Tuple[float, float, float] = (0.25, 0.5, 0.75),
    out_path: Path = Path("density_strata_iou.png"),
    label: str = "",
    style: Dict = load_style(None),
    dpi: int = 180,
) -> None:
    """Plot IoU stratified by density quantiles."""

    density = np.asarray(density, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if density.ndim != 1:
        density = density.reshape(-1)
    if y_true.shape[0] != density.shape[0]:
        y_true = np.broadcast_to(y_true, density.shape)
    if y_pred.shape[0] != density.shape[0]:
        y_pred = np.broadcast_to(y_pred, density.shape)

    def _is_binary(arr: np.ndarray) -> bool:
        arr = arr.astype(float, copy=False)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return False
        unique = np.unique(arr)
        return np.all(np.isin(unique, [0.0, 1.0]))

    qs = np.quantile(density, quantiles)
    edges = np.concatenate(([-np.inf], qs, [np.inf]))
    centers: List[float] = []
    ious: List[float] = []

    for idx in range(len(edges) - 1):
        mask = (density > edges[idx]) & (density <= edges[idx + 1])
        if not np.any(mask):
            continue
        centers.append(float(density[mask].mean()))
        segment_true = y_true[mask]
        segment_pred = y_pred[mask]
        if _is_binary(segment_true) and _is_binary(segment_pred):
            iou_val = _compute_iou(segment_true.astype(bool), segment_pred.astype(bool))
        else:
            iou_val = float(np.nanmean(segment_pred))
        ious.append(iou_val)

    fig, ax = _setup_axes(style)
    ax.plot(
        centers,
        ious,
        marker="o",
        color=style["colors"].get("curve", "#1f77b4"),
        linewidth=style["line_width"],
    )
    ax.scatter(centers, ious, color=style["colors"].get("positive", "#1f77b4"), s=40)
    for q in qs:
        ax.axvline(q, color=style["colors"].get("quantile", "#2ca02c"), linestyle="--", linewidth=1.0)
    ax.set_xlabel("Density", fontsize=style["font_size"])
    ax.set_ylabel("IoU", fontsize=style["font_size"])
    ax.set_title(f"Density-Stratified IoU – {label}", fontsize=style["font_size"] * 1.1)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


def plot_radar_ablations(
    metrics_by_setting: Dict[str, Dict[str, float]],
    out_path: Path,
    style: Dict,
    dpi: int,
) -> None:
    """Plot a radar chart comparing ablation settings."""

    if not metrics_by_setting:
        LOGGER.warning("No ablation metrics provided; skipping radar plot.")
        return

    metrics = sorted({metric for d in metrics_by_setting.values() for metric in d.keys()})
    values = np.array([[metrics_by_setting[s].get(m, np.nan) for m in metrics] for s in metrics_by_setting])
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    span = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    normalized = (values - mins) / span

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=style["figure_size"])
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for idx, (setting, scores) in enumerate(zip(metrics_by_setting.keys(), normalized)):
        values_loop = np.concatenate([scores, scores[:1]])
        ax.plot(
            angles,
            values_loop,
            label=setting,
            linewidth=style["line_width"],
        )
        ax.fill(angles, values_loop, alpha=style["alpha_fill"], label="_nolegend_")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=style["font_size"] * 0.9)
    ax.set_yticklabels([])
    ax.set_title("Ablation Radar", fontsize=style["font_size"] * 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=style["font_size"] * 0.8)
    savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


def plot_pareto_iou_vs_throughput(
    iou: np.ndarray,
    throughput: np.ndarray,
    labels: List[str],
    out_path: Path,
    style: Dict,
    dpi: int,
) -> None:
    """Plot IoU versus throughput with Pareto frontier."""

    iou = np.asarray(iou)
    throughput = np.asarray(throughput)
    if iou.size == 0:
        LOGGER.warning("Empty arrays provided to Pareto plot; skipping.")
        return

    fig, ax = _setup_axes(style)
    ax.scatter(
        throughput,
        iou,
        color=style["colors"].get("curve", "#1f77b4"),
        s=40,
        alpha=0.9,
    )

    order = np.argsort(-throughput)
    best = -np.inf
    frontier_indices: List[int] = []
    for idx in order:
        if iou[idx] >= best:
            frontier_indices.append(idx)
            best = iou[idx]
    frontier_indices = sorted(frontier_indices, key=lambda x: throughput[x])

    ax.plot(
        throughput[frontier_indices],
        iou[frontier_indices],
        color=style["colors"].get("pareto", "#ff7f0e"),
        linewidth=style["line_width"],
        label="Pareto frontier",
    )

    # Label up to five top-performing points.
    top_k = list(islice(np.argsort(-iou), min(5, len(labels))))
    offsets = np.linspace(-0.02, 0.02, len(top_k))
    for offset, idx in zip(offsets, top_k):
        ax.text(
            throughput[idx],
            iou[idx] + offset,
            labels[idx],
            fontsize=style["font_size"] * 0.8,
            ha="center",
        )

    ax.set_xlabel("Throughput", fontsize=style["font_size"])
    ax.set_ylabel("IoU", fontsize=style["font_size"])
    ax.set_title("IoU vs. Throughput", fontsize=style["font_size"] * 1.1)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=style["font_size"] * 0.85)
    savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


def plot_histograms(
    values: Dict[str, np.ndarray],
    bins: int,
    out_path: Path,
    style: Dict,
    dpi: int,
) -> None:
    """Plot histograms for multiple metrics."""

    if not values:
        LOGGER.warning("No histogram values provided; skipping.")
        return

    names = list(values.keys())
    total = len(names)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(style["figure_size"][0] * cols, style["figure_size"][1] * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for idx, name in enumerate(names):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        data = np.asarray(values[name])
        ax.hist(
            data[~np.isnan(data)],
            bins=bins,
            color=style["colors"].get("hist", "#1f77b4"),
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_title(name, fontsize=style["font_size"])
        ax.tick_params(labelsize=style["font_size"] * 0.8)
    # Hide unused axes.
    for idx in range(total, rows * cols):
        row, col = divmod(idx, cols)
        fig.delaxes(axes[row, col])

    fig.suptitle("Metric Distributions", fontsize=style["font_size"] * 1.2)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig_atomic(fig, out_path, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _prepare_default_probabilities(table: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    values = table.get("iou")
    if values is None or values.size == 0:
        LOGGER.warning("IoU column missing; synthesising placeholder data for PR/reliability plots.")
        values = np.linspace(0.1, 0.9, 50)
    values = np.nan_to_num(values, nan=float(np.nanmean(values)))
    norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
    y_true = norm >= np.median(norm)
    return y_true.astype(int), norm


def _parse_formats(formats: str) -> List[str]:
    output = []
    for fmt in formats.split(","):
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        if fmt not in {"png", "svg"}:
            LOGGER.warning("Unsupported format '%s'; falling back to png.", fmt)
            output.append("png")
        else:
            output.append(fmt)
    return output or ["png"]


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the plotting CLI."""

    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--per-protein", type=Path, required=True, help="Path to per-protein CSV table.")
    parser.add_argument("--aggregate", type=Path, default=None, help="Path to aggregate JSON file.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory where figures will be stored.")
    parser.add_argument("--style", type=Path, default=None, help="Optional YAML style file.")
    parser.add_argument("--formats", default="png", help="Comma-separated list of formats (png, svg).")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--label", default="model", help="Label for figure titles.")
    parser.add_argument("--reliability-bins", type=int, default=10, help="Number of bins for reliability diagrams.")
    parser.add_argument("--hist-bins", type=int, default=20, help="Bins for histogram plots.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    table = load_table(args.per_protein)
    style = load_style(args.style)
    formats = _parse_formats(args.formats)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate: Dict[str, object] = {}
    if args.aggregate is not None and args.aggregate.exists():
        LOGGER.info("Loading aggregate statistics from %s", args.aggregate)
        with args.aggregate.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)

    # Prepare probability arrays.
    y_true, prob_pos = _prepare_default_probabilities(table)
    if "y_true" in aggregate and "prob_pos" in aggregate:
        agg_y = np.asarray(aggregate["y_true"], dtype=float)
        agg_prob = np.asarray(aggregate["prob_pos"], dtype=float)
        if agg_y.shape == agg_prob.shape:
            y_true = agg_y.astype(int)
            prob_pos = agg_prob

    for fmt in formats:
        plot_pr_curve(
            y_true,
            prob_pos,
            out_dir / f"pr_curve.{fmt}",
            args.label,
            style,
            args.dpi,
        )
        plot_reliability_diagram(
            y_true,
            prob_pos,
            args.reliability_bins,
            out_dir / f"reliability.{fmt}",
            args.label,
            style,
            args.dpi,
        )

        density = table.get("num_atoms")
        voxels = table.get("num_voxels")
        if density is not None and voxels is not None:
            density_values = density / np.maximum(voxels, 1)
        else:
            density_values = np.linspace(0.1, 1.0, len(prob_pos))
        iou_values = table.get("iou", prob_pos)
        plot_density_stratified_iou(
            density_values,
            prob_pos,
            iou_values,
            out_path=out_dir / f"density_strata_iou.{fmt}",
            label=args.label,
            style=style,
            dpi=args.dpi,
        )

        if isinstance(aggregate.get("ablations"), dict):
            metrics_by_setting = {
                str(key): {str(metric): float(value) for metric, value in metrics.items()}
                for key, metrics in aggregate["ablations"].items()
            }
        else:
            metrics_by_setting = {
                args.label: {
                    "IoU": float(np.nanmean(table.get("iou", np.array([np.nan])))),
                    "F1": float(np.nanmean(table.get("f1", np.array([np.nan])))),
                    "PR AUC": float(np.nanmean(table.get("pr_auc", np.array([np.nan])))),
                }
            }
        plot_radar_ablations(
            metrics_by_setting,
            out_dir / f"radar_ablations.{fmt}",
            style,
            args.dpi,
        )

        throughput = table.get("atoms_per_second")
        if throughput is None:
            throughput = np.maximum(table.get("num_atoms", np.arange(len(prob_pos)) + 1), 1)
        labels = aggregate.get("labels") if isinstance(aggregate.get("labels"), list) else []
        if not labels:
            labels = [f"case_{idx}" for idx in range(len(prob_pos))]
        plot_pareto_iou_vs_throughput(
            table.get("iou", prob_pos),
            throughput,
            labels,
            out_dir / f"pareto_iou_vs_throughput.{fmt}",
            style,
            args.dpi,
        )

        histogram_values = {name: table[name] for name in ["iou", "ece", "brier"] if name in table}
        if not histogram_values:
            histogram_values = {"IoU": table.get("iou", prob_pos)}
        plot_histograms(
            histogram_values,
            args.hist_bins,
            out_dir / f"histograms.{fmt}",
            style,
            args.dpi,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
