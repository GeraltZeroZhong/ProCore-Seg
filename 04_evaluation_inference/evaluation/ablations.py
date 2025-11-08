"""Ablation utilities for segmentation evaluation."""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .evaluate_dataset import (
    DEFAULT_POSITIVE_CLASS,
    aggregate_metrics,
    compute_per_protein_metrics,
    load_npz,
    write_csv_or_tsv,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)


def enumerate_settings(posthoc_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not posthoc_cfg:
        return [{}]

    keys = sorted(posthoc_cfg.keys())
    value_lists: list[list[Any]] = []
    for key in keys:
        value = posthoc_cfg[key]
        if isinstance(value, (list, tuple)):
            value_lists.append(list(value))
        else:
            value_lists.append([value])

    combinations: list[dict[str, Any]] = []
    for product in itertools.product(*value_lists):
        combo = {key: value for key, value in zip(keys, product)}
        combinations.append(combo)
    return combinations or [{}]


def recompute_probs_atom(npz_payload: dict[str, Any], temperature: float) -> np.ndarray:
    if abs(temperature - 1.0) < 1e-6:
        return np.asarray(npz_payload["probs_atom"], dtype=np.float64)

    logits_key = None
    if "logits_atom" in npz_payload:
        logits_key = "logits_atom"
    elif "logits_voxel" in npz_payload and "atom2voxel" in npz_payload:
        logits_key = "logits_voxel"
    if logits_key is None:
        raise ValueError("Temperature scaling requested but logits are not available in NPZ payload")

    logits = np.asarray(npz_payload[logits_key], dtype=np.float64)
    if logits_key == "logits_voxel":
        indices = np.asarray(npz_payload["atom2voxel"], dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("atom2voxel must be a 1-D array of indices")
        logits = logits[indices]

    logits = logits / float(temperature)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _core_percentile_for_strategy(strategy: str) -> float:
    mapping = {
        "median": 50.0,
        "p25": 25.0,
        "p50": 50.0,
        "p75": 75.0,
    }
    if strategy not in mapping:
        raise ValueError(f"Unknown core strategy: {strategy}")
    return mapping[strategy]


def _core_percentiles_from_metrics(
    metric_names: list[str], strategy: str
) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for name in metric_names:
        if not name.startswith("core_iou"):
            continue
        suffix = name.split("_")[-1]
        if suffix.startswith("p"):
            try:
                mapping[name] = float(suffix[1:])
                continue
            except ValueError:
                pass
        mapping[name] = _core_percentile_for_strategy(strategy)
    return mapping


def eval_run_with_settings(
    run_name: str,
    run_dir: Path,
    settings: list[dict[str, Any]],
    core_metrics: list[str],
    metrics_of_interest: list[str],
    base_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    npz_files = sorted(run_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found for run {run_name} at {run_dir}")

    per_protein_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []

    for setting in settings:
        temperature = float(setting.get("temperature", 1.0))
        core_strategy = setting.get("core_strategy", "median")
        percentile_mapping = _core_percentiles_from_metrics(core_metrics, str(core_strategy))
        setting_name_parts = [f"temperature={temperature:g}", f"core={core_strategy}"]
        for key, value in setting.items():
            if key in {"temperature", "core_strategy"}:
                continue
            setting_name_parts.append(f"{key}={value}")
        setting_name = "|".join(setting_name_parts)

        rows_for_setting: list[dict[str, Any]] = []
        for npz_path in npz_files:
            payload = load_npz(npz_path)
            probs = recompute_probs_atom(payload, temperature)
            payload = dict(payload)
            payload["probs_atom"] = probs
            payload["pred_atom"] = np.argmax(probs, axis=1).astype(np.int32)
            metrics = compute_per_protein_metrics(
                payload,
                ece_bins=base_kwargs.get("ece_bins", 15),
                positive_class=base_kwargs.get("positive_class", DEFAULT_POSITIVE_CLASS),
                core_percentiles=percentile_mapping if percentile_mapping else None,
            )
            metrics["pdb_id"] = payload.get("meta", {}).get("pdb_id", npz_path.stem)
            metrics["run"] = run_name
            metrics["setting"] = setting_name
            rows_for_setting.append(metrics)

        per_protein_rows.extend(rows_for_setting)
        agg = aggregate_metrics(rows_for_setting, base_kwargs.get("n_bootstrap", 1000), base_kwargs.get("alpha", 0.05), base_kwargs.get("seed", 7))
        for metric, mean in agg["means"].items():
            if metric not in metrics_of_interest:
                continue
            aggregates.append(
                {
                    "run": run_name,
                    "setting": setting_name,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": agg["ci95"].get(metric, [float("nan"), float("nan")])[0],
                    "ci_high": agg["ci95"].get(metric, [float("nan"), float("nan")])[1],
                }
            )

    return per_protein_rows, aggregates


def _direction_for_metric(metric: str) -> int:
    lower_is_better = {"ece", "mce", "nll", "brier"}
    return -1 if metric.lower() in lower_is_better else 1


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < sorted_values.size:
        j = i + 1
        while j < sorted_values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _wilcoxon_signed_rank(diff: np.ndarray) -> tuple[float, int]:
    mask = diff != 0
    diff = diff[mask]
    n = diff.size
    if n == 0:
        return 0.0, 0
    ranks = _rankdata(np.abs(diff))
    w_pos = float(np.sum(ranks[diff > 0]))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0:
        return 0.0, n
    z = (w_pos - mean_w) / np.sqrt(var_w)
    # Two sided normal approximation
    p = 2 * min(_normal_cdf(z), 1 - _normal_cdf(z))
    return float(p), n


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / np.sqrt(2.0)))


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x = x.size
    n_y = y.size
    if n_x == 0 or n_y == 0:
        return 0.0
    signs = np.sign(x[:, None] - y[None, :])
    return float(np.sum(signs) / (n_x * n_y))


def pairwise_compare(
    per_protein_long: list[dict[str, Any]],
    metric: str,
    best_settings: dict[str, str],
    baseline: str | None = None,
) -> dict[str, Any]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    for row in per_protein_long:
        if row["metric"] != metric:
            continue
        key = (row["run"], row["setting"])
        values.setdefault(key, {})[row["pdb_id"]] = float(row["value"])

    runs = list(best_settings.keys())
    if baseline and baseline not in runs:
        raise ValueError(f"Baseline {baseline} not present in runs")

    comparisons: dict[str, Any] = {}

    def _compare(pair_a: str, pair_b: str) -> None:
        key_a = (pair_a, best_settings[pair_a])
        key_b = (pair_b, best_settings[pair_b])
        data_a = values.get(key_a, {})
        data_b = values.get(key_b, {})
        pdb_ids = sorted(set(data_a).intersection(data_b))
        if not pdb_ids:
            comparisons[f"{pair_a}__vs__{pair_b}"] = {
                "p_value": float("nan"),
                "n": 0,
                "cohen_dz": float("nan"),
                "cliffs_delta": float("nan"),
            }
            return
        values_a = np.array([data_a[pdb] for pdb in pdb_ids], dtype=np.float64)
        values_b = np.array([data_b[pdb] for pdb in pdb_ids], dtype=np.float64)
        diff = values_a - values_b
        p_value, n_used = _wilcoxon_signed_rank(diff.copy())
        dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if n_used > 1 else float("nan")
        delta = _cliffs_delta(values_a, values_b)
        comparisons[f"{pair_a}__vs__{pair_b}"] = {
            "p_value": p_value,
            "n": n_used,
            "cohen_dz": dz,
            "cliffs_delta": delta,
        }

    if baseline:
        for run in runs:
            if run == baseline:
                continue
            _compare(baseline, run)
    else:
        for i, run_a in enumerate(runs):
            for run_b in runs[i + 1 :]:
                _compare(run_a, run_b)

    return comparisons


def _build_long_table(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        for metric in metrics:
            if metric not in row:
                continue
            entries.append(
                {
                    "run": row["run"],
                    "setting": row["setting"],
                    "pdb_id": row["pdb_id"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation ablations")
    parser.add_argument("--grid", type=Path, required=True, help="Grid configuration YAML")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline", type=str, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with args.grid.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    base_cfg = cfg.get("base", {})
    metrics = list(base_cfg.get("metrics", []))
    if not metrics:
        raise ValueError("No metrics listed in grid base configuration")
    ece_bins = int(base_cfg.get("ece_bins", 15))
    positive_class = int(base_cfg.get("positive_class", DEFAULT_POSITIVE_CLASS))

    runs_cfg = cfg.get("runs", [])
    if not runs_cfg:
        raise ValueError("No runs defined in grid configuration")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    aggregated_rows: list[dict[str, Any]] = []

    base_kwargs = {
        "ece_bins": ece_bins,
        "positive_class": positive_class,
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
    }

    for run in runs_cfg:
        run_name = run.get("name")
        run_dir = Path(run.get("run_dir"))
        posthoc = run.get("posthoc", {})
        settings = enumerate_settings(posthoc)
        core_metrics = [m for m in metrics if m.startswith("core_iou")]
        rows, agg = eval_run_with_settings(
            run_name,
            run_dir,
            settings,
            core_metrics,
            metrics,
            base_kwargs,
        )
        all_rows.extend(rows)
        aggregated_rows.extend(agg)

    per_protein_long = _build_long_table(all_rows, metrics)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv_or_tsv(all_rows, tables_dir / "ablations_per_protein.csv")

    if aggregated_rows:
        write_csv_or_tsv(aggregated_rows, tables_dir / "ablations_aggregate.csv")

    # Determine best settings per run for each metric
    best_settings_per_metric: dict[str, dict[str, str]] = {}
    for metric in metrics:
        direction = _direction_for_metric(metric)
        best_for_metric: dict[str, tuple[str, float]] = {}
        for row in aggregated_rows:
            if row["metric"] != metric:
                continue
            run_name = row["run"]
            score = row["mean"] * direction
            if run_name not in best_for_metric or score > best_for_metric[run_name][1]:
                best_for_metric[run_name] = (row["setting"], score)
        best_settings_per_metric[metric] = {run: setting for run, (setting, _) in best_for_metric.items()}

    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        best_settings = best_settings_per_metric.get(metric, {})
        comparisons = pairwise_compare(per_protein_long, metric, best_settings, args.baseline)
        write_json_atomic(comparisons, stats_dir / f"{metric}_pairwise_tests.json")

    LOGGER.info("Ablation analysis completed. Outputs stored in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())