"""Markdown report generation for evaluation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def load_aggregate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Aggregate file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_per_protein(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        rows: list[dict[str, Any]] = []
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
        return rows


def _format_ci(mean: float | None, ci: list[float] | None) -> str:
    if mean is None or ci is None or len(ci) != 2:
        return "N/A"
    return f"{mean:.4f} (95% CI: {ci[0]:.4f}–{ci[1]:.4f})"


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_bottom(rows: list[dict[str, Any]], metric: str, k: int = 5) -> tuple[list[str], list[str]]:
    scored = []
    for row in rows:
        value = _maybe_float(row.get(metric))
        if value is None:
            continue
        scored.append((value, str(row.get("pdb_id", "unknown"))))
    if not scored:
        return [], []
    scored.sort(key=lambda item: item[0])
    bottom = [f"{pdb} ({score:.4f})" for score, pdb in scored[:k]]
    top = [f"{pdb} ({score:.4f})" for score, pdb in reversed(scored[-k:])]
    return top, bottom


def _figure_references(fig_dir: Path | None, include_boundary: bool, include_calibration: bool) -> list[str]:
    if fig_dir is None or not fig_dir.exists():
        return []
    refs: list[str] = []
    figures = sorted(fig_dir.glob("*"))
    for figure in figures:
        name = figure.name.lower()
        if "boundary" in name and not include_boundary:
            continue
        if "calibration" in name and not include_calibration:
            continue
        refs.append(figure.as_posix())
    return refs


def render_markdown(
    title: str,
    agg: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    fig_dir: Path | None,
    include_boundary: bool,
    include_calibration: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    generated_at = agg.get("generated_at", "")
    n_entries = agg.get("n", 0)
    env = agg.get("env", {})
    config = agg.get("config", {})
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    lines.append("## Summary")
    lines.append("")
    lines.append(f"*Generated:* {generated_at}")
    lines.append(f"*Entries:* {n_entries}")
    lines.append(
        f"*Environment:* Python {env.get('python', 'unknown')}, NumPy {env.get('numpy', 'unknown')}"
    )
    lines.append(f"*Config hash:* `{config_hash}`")
    if config:
        preview = json.dumps(config, sort_keys=True)[:200]
        lines.append(f"*Config excerpt:* `{preview}`")
    lines.append("")

    means = agg.get("means", {})
    ci = agg.get("ci95", {})

    lines.append("## Main Metrics")
    lines.append("")
    main_metrics = [
        ("IoU", _format_ci(means.get("iou"), ci.get("iou"))),
        ("Core IoU (P50)", _format_ci(means.get("core_iou_p50"), ci.get("core_iou_p50"))),
        ("F1", _format_ci(means.get("f1"), ci.get("f1"))),
        ("PR AUC", _format_ci(means.get("pr_auc"), ci.get("pr_auc"))),
    ]
    lines.append("| Metric | Score |")
    lines.append("| --- | --- |")
    for metric_name, formatted in main_metrics:
        lines.append(f"| {metric_name} | {formatted} |")
    lines.append("")

    if include_boundary:
        lines.append("## Boundary Quality")
        lines.append("")
        for radius in (2, 4, 6):
            f1_key = f"boundary_f1@{radius}"
            iou_key = f"boundary_iou@{radius}"
            lines.append(
                f"- **F1@{radius}Å:** {_format_ci(means.get(f1_key), ci.get(f1_key))}; "
                f"**IoU@{radius}Å:** {_format_ci(means.get(iou_key), ci.get(iou_key))}"
            )
        lines.append("")

    if include_calibration:
        lines.append("## Calibration")
        lines.append("")
        lines.append(f"- **ECE:** {_format_ci(means.get('ece'), ci.get('ece'))}")
        lines.append(f"- **MCE:** {_format_ci(means.get('mce'), ci.get('mce'))}")
        lines.append(f"- **Brier score:** {_format_ci(means.get('brier'), ci.get('brier'))}")
        lines.append(f"- **NLL:** {_format_ci(means.get('nll'), ci.get('nll'))}")
        lines.append("")

    figures = _figure_references(fig_dir, include_boundary, include_calibration)
    if figures:
        lines.append("## Figures")
        lines.append("")
        for figure in figures:
            lines.append(f"- {figure}")
        lines.append("")

    if rows:
        lines.append("## Per-Protein Highlights")
        lines.append("")
        top, bottom = _top_bottom(rows, "iou")
        if top:
            lines.append("**Top IoU:**")
            for entry in top:
                lines.append(f"- {entry}")
        if bottom:
            lines.append("")
            lines.append("**Challenging Cases (lowest IoU):**")
            for entry in bottom:
                lines.append(f"- {entry}")
        lines.append("")
        lines.append("Full table: see `per_protein.csv`.")

    return "\n".join(lines).strip() + "\n"


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--run-dir", type=Path, required=True, help="Evaluation results directory")
    parser.add_argument("--out", type=Path, required=True, help="Output markdown path")
    parser.add_argument("--title", type=str, required=True, help="Report title")
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--include-boundary", type=str, default="true")
    parser.add_argument("--include-calibration", type=str, default="true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    aggregate_path = args.run_dir / "aggregate.json"
    per_protein_csv = args.run_dir / "per_protein.csv"
    if not per_protein_csv.exists():
        alternative = args.run_dir / "per_protein.tsv"
        per_protein_csv = alternative if alternative.exists() else per_protein_csv

    agg = load_aggregate(aggregate_path)
    rows = load_per_protein(per_protein_csv)
    include_boundary = args.include_boundary.lower() == "true"
    include_calibration = args.include_calibration.lower() == "true"

    markdown = render_markdown(
        args.title,
        agg,
        rows,
        args.fig_dir,
        include_boundary,
        include_calibration,
    )

    atomic_write_text(markdown, args.out)
    LOGGER.info("Report written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())