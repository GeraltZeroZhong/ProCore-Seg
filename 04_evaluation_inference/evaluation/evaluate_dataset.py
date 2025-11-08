"""Dataset level evaluation utilities.

This module aggregates per-structure inference artefacts produced by
``run_inference.py`` into per-protein metrics and a dataset wide summary.

The implementation intentionally relies only on the Python standard library
and :mod:`numpy` so that it can execute in CPU only environments without any
optional dependencies installed.  When ``pandas`` is available it is utilised
for convenient table generation, otherwise a standards compliant CSV writer is
used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


LOGGER = logging.getLogger(__name__)


try:  # Optional dependency – used only when present.
    import pandas as _pd  # type: ignore

    HAS_PANDAS = True
except Exception:  # pragma: no cover - defensive guard, pandas is optional.
    HAS_PANDAS = False


DEFAULT_POSITIVE_CLASS = 1
FLOAT_EPS = 1e-12


@dataclass(frozen=True)
class BootstrapResult:
    """Container describing bootstrap aggregations for a metric."""

    mean: float
    ci_low: float
    ci_high: float


def load_npz(path: Path) -> dict[str, Any]:
    """Load an inference artefact stored as ``.npz``.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.

    Returns
    -------
    dict[str, Any]
        Mapping containing the arrays and metadata stored inside the archive.

    Raises
    ------
    ValueError
        If the file does not contain the mandatory keys required for
        evaluation.
    """

    if not path.exists():
        raise ValueError(f"NPZ file does not exist: {path}")

    LOGGER.debug("Loading inference artefact: %s", path)
    payload = np.load(path, allow_pickle=True)

    required = {
        "probs_atom",
        "pred_atom",
        "labels_atom",
        "density_atom",
        "coords_atom",
        "chain",
        "resseq",
        "icode",
        "meta",
    }
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"NPZ file {path} is missing required keys: {', '.join(sorted(missing))}"
        )

    data = {key: payload[key] for key in payload.files}

    # Ensure arrays are numpy ndarrays (np.load may lazily load object arrays).
    for key in payload.files:
        if isinstance(data[key], np.ndarray):
            continue
        data[key] = np.asarray(data[key])

    meta_obj = data.get("meta")
    if isinstance(meta_obj, np.ndarray) and meta_obj.dtype == object:
        # np.save with allow_pickle stores dicts as zero-d arrays of objects.
        meta_obj = meta_obj.item()
        data["meta"] = meta_obj
    if not isinstance(meta_obj, Mapping):
        raise ValueError(f"NPZ file {path} contains invalid meta field: {type(meta_obj)!r}")

    return data


def _validate_shapes(npz_payload: Mapping[str, Any]) -> None:
    """Validate basic shape contracts for the loaded arrays."""

    probs = np.asarray(npz_payload["probs_atom"])
    preds = np.asarray(npz_payload["pred_atom"])
    labels = np.asarray(npz_payload["labels_atom"])
    density = np.asarray(npz_payload["density_atom"])
    coords = np.asarray(npz_payload["coords_atom"])
    chain = np.asarray(npz_payload["chain"])
    resseq = np.asarray(npz_payload["resseq"])
    icode = np.asarray(npz_payload["icode"])

    if probs.ndim != 2:
        raise ValueError("probs_atom must be a 2-D array")
    n_atoms, n_classes = probs.shape
    if n_classes < 2:
        raise ValueError("probs_atom must contain at least two classes")

    for name, array in {
        "pred_atom": preds,
        "labels_atom": labels,
        "density_atom": density,
        "coords_atom": coords,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
    }.items():
        if array.shape[0] != n_atoms:
            raise ValueError(
                f"{name} length ({array.shape[0]}) does not match probs_atom ({n_atoms})"
            )

    if coords.shape[1:] != (3,):
        raise ValueError("coords_atom must have shape (N, 3)")


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    sum_exp = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(sum_exp, FLOAT_EPS, None)


def _binary_metrics(
    labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, positive_class: int
) -> dict[str, float]:
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D")

    labels_bin = (labels == positive_class).astype(np.int32)
    preds_bin = (preds == positive_class).astype(np.int32)

    total = labels_bin.size
    correct = int((preds_bin == labels_bin).sum())
    accuracy = correct / total if total else 0.0

    tp = int(((preds_bin == 1) & (labels_bin == 1)).sum())
    fp = int(((preds_bin == 1) & (labels_bin == 0)).sum())
    fn = int(((preds_bin == 0) & (labels_bin == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    prob_pos = np.clip(probs[:, positive_class], FLOAT_EPS, 1.0 - FLOAT_EPS)
    pr_auc = _pr_auc(prob_pos, labels_bin)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _pr_auc(prob_pos: np.ndarray, labels: np.ndarray) -> float:
    if not labels.size:
        return 0.0
    order = np.argsort(-prob_pos)
    sorted_probs = prob_pos[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    total_pos = tp[-1]
    total_neg = fp[-1]
    if total_pos == 0:
        return 0.0
    precision = tp / np.maximum(tp + fp, FLOAT_EPS)
    recall = tp / total_pos
    # Append starting point (recall=0, precision=1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapz(precision, recall))


def _compute_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_true, mask_pred).sum(dtype=np.float64)
    union = np.logical_or(mask_true, mask_pred).sum(dtype=np.float64)
    return float(intersection / union) if union > 0 else 0.0


def _core_iou(
    labels: np.ndarray,
    preds: np.ndarray,
    density: np.ndarray,
    percentile: float,
    positive_class: int,
) -> float:
    if labels.size == 0:
        return 0.0
    if percentile <= 0 or percentile >= 100:
        threshold = np.percentile(density, percentile)
    else:
        threshold = np.percentile(density, percentile)
    mask = density >= threshold
    if not np.any(mask):
        return 0.0
    labels_bin = labels == positive_class
    preds_bin = preds == positive_class
    return _compute_iou(labels_bin[mask], preds_bin[mask])


def _residue_keys(
    chains: np.ndarray, resseq: np.ndarray, icodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    chains = chains.astype(str)
    resseq = resseq.astype(str)
    icodes = icodes.astype(str)
    keys = np.core.defchararray.add(chains, resseq)
    keys = np.core.defchararray.add(keys, icodes)
    return np.unique(keys, return_inverse=True)


def _aggregate_mean(values: np.ndarray, inverse: np.ndarray, n_residues: int) -> np.ndarray:
    if values.ndim == 1:
        result = np.zeros(n_residues, dtype=np.float64)
    else:
        result = np.zeros((n_residues, values.shape[1]), dtype=np.float64)
    counts = np.zeros(n_residues, dtype=np.float64)
    for idx, resid_idx in enumerate(inverse):
        result[resid_idx] += values[idx]
        counts[resid_idx] += 1.0
    if values.ndim == 1:
        return result / np.maximum(counts, 1.0)
    counts = np.maximum(counts, 1.0)[:, None]
    return result / counts


def _aggregate_majority(
    labels: np.ndarray, inverse: np.ndarray, n_residues: int
) -> np.ndarray:
    votes = np.zeros(n_residues, dtype=np.int32)
    for idx, resid_idx in enumerate(inverse):
        votes[resid_idx] += 1 if labels[idx] else -1
    return (votes >= 0).astype(np.int32)


def _boundary_metrics(
    coords: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    chains: np.ndarray,
    resseq: np.ndarray,
    icodes: np.ndarray,
    tolerances: Iterable[float],
) -> dict[str, float]:
    unique_keys, inverse = _residue_keys(chains, resseq, icodes)
    n_res = unique_keys.size
    residue_coords = _aggregate_mean(coords, inverse, n_res)
    residue_prob = _aggregate_mean(preds.astype(np.float64), inverse, n_res)
    residue_labels = _aggregate_majority(labels, inverse, n_res)
    residue_preds = (residue_prob >= 0.5).astype(np.int32)

    metrics: dict[str, float] = {}

    for tol in tolerances:
        true_mask = _boundary_mask(residue_coords, residue_labels, tol)
        pred_mask = _boundary_mask(residue_coords, residue_preds, tol)
        key_f1 = f"boundary_f1@{int(tol)}"
        key_iou = f"boundary_iou@{int(tol)}"
        metrics[key_f1] = _f1_from_masks(true_mask, pred_mask)
        metrics[key_iou] = _compute_iou(true_mask, pred_mask)
    return metrics


def _boundary_mask(coords: np.ndarray, labels: np.ndarray, tolerance: float) -> np.ndarray:
    n = labels.size
    mask = np.zeros(n, dtype=bool)
    if tolerance <= 0:
        return mask
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if not pos_idx.size or not neg_idx.size:
        return mask

    tol_sq = tolerance * tolerance
    for idx in pos_idx:
        diffs = coords[neg_idx] - coords[idx]
        d2 = np.sum(diffs * diffs, axis=1)
        if np.any(d2 <= tol_sq):
            mask[idx] = True
    for idx in neg_idx:
        diffs = coords[pos_idx] - coords[idx]
        d2 = np.sum(diffs * diffs, axis=1)
        if np.any(d2 <= tol_sq):
            mask[idx] = True
    return mask


def _f1_from_masks(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    tp = np.logical_and(true_mask, pred_mask).sum(dtype=np.float64)
    fp = np.logical_and(~true_mask, pred_mask).sum(dtype=np.float64)
    fn = np.logical_and(true_mask, ~pred_mask).sum(dtype=np.float64)
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _calibration_metrics(prob_pos: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, float]:
    if n_bins <= 1:
        raise ValueError("ece_bins must be greater than 1")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob_pos, bins, right=True)
    ece = 0.0
    mce = 0.0
    total = labels.size
    brier = np.mean((prob_pos - labels) ** 2) if total else 0.0
    nll = float(-np.mean(labels * np.log(prob_pos + FLOAT_EPS) + (1 - labels) * np.log(1 - prob_pos + FLOAT_EPS))) if total else 0.0
    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = np.mean(prob_pos[mask])
        acc = np.mean(labels[mask])
        gap = abs(conf - acc)
        weight = mask.sum() / total if total else 0.0
        ece += gap * weight
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce), "brier": float(brier), "nll": nll}


def compute_per_protein_metrics(
    npz_payload: Mapping[str, Any],
    ece_bins: int = 15,
    positive_class: int = DEFAULT_POSITIVE_CLASS,
    core_percentiles: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    payload = {key: np.asarray(value) for key, value in npz_payload.items()}
    _validate_shapes(payload)

    probs = payload["probs_atom"].astype(np.float64)
    preds = payload["pred_atom"].astype(np.int32)
    labels = payload["labels_atom"].astype(np.int32)
    density = payload["density_atom"].astype(np.float64)
    coords = payload["coords_atom"].astype(np.float64)
    chains = payload["chain"].astype(str)
    resseq = payload["resseq"].astype(str)
    icodes = payload["icode"].astype(str)

    metrics: dict[str, Any] = {}

    bin_metrics = _binary_metrics(labels, probs, preds, positive_class)
    metrics.update(bin_metrics)

    prob_pos = np.clip(probs[:, positive_class], FLOAT_EPS, 1.0 - FLOAT_EPS)
    calib = _calibration_metrics(prob_pos, (labels == positive_class).astype(int), ece_bins)
    metrics.update(calib)

    metrics["iou"] = _compute_iou(labels == positive_class, preds == positive_class)

    percentiles = core_percentiles or {
        "core_iou_p25": 25.0,
        "core_iou_p50": 50.0,
        "core_iou_p75": 75.0,
    }
    for name, percentile in percentiles.items():
        metrics[name] = _core_iou(labels, preds, density, percentile, positive_class)

    boundary = _boundary_metrics(
        coords,
        (labels == positive_class).astype(int),
        (preds == positive_class).astype(int),
        chains,
        resseq,
        icodes,
        tolerances=[2.0, 4.0, 6.0],
    )
    metrics.update(boundary)

    metrics["num_atoms"] = int(labels.size)
    metrics["num_positive"] = int((labels == positive_class).sum())
    meta = npz_payload.get("meta", {})
    metrics["pdb_id"] = str(meta.get("pdb_id", meta.get("id", "unknown")))
    metrics["num_voxels"] = int(meta.get("num_voxels", 0))
    if metrics["num_atoms"] > 0:
        metrics["sparsity"] = (
            float(metrics["num_voxels"]) / float(metrics["num_atoms"])
            if metrics["num_atoms"]
            else 0.0
        )
    else:
        metrics["sparsity"] = 0.0

    return metrics


def _bootstrap_statistics(
    data: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator
) -> BootstrapResult:
    if data.size == 0:
        return BootstrapResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    mean = float(np.mean(data))
    if n_boot <= 0:
        return BootstrapResult(mean=mean, ci_low=mean, ci_high=mean)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = data.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(data[idx]))
    lower = float(np.percentile(boot_means, alpha / 2 * 100))
    upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    return BootstrapResult(mean=mean, ci_low=lower, ci_high=upper)


def aggregate_metrics(
    rows: list[dict[str, Any]],
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows provided for aggregation")

    metrics: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in {"pdb_id"}:
                continue
            if isinstance(value, (int, float, np.floating)):
                metrics.setdefault(key, []).append(float(value))

    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {
        "run_dir": "",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n": len(rows),
        "means": {},
        "ci95": {},
        "stats": {},
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "config": {},
    }

    for key, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        boot = _bootstrap_statistics(arr, n_boot, alpha, rng)
        summary["means"][key] = boot.mean
        summary["ci95"][key] = [boot.ci_low, boot.ci_high]

    return summary


def write_csv_or_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty rows")

    columns = list(rows[0].keys())
    for row in rows:
        for key in columns:
            row.setdefault(key, "")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PANDAS:
        df = _pd.DataFrame(rows, columns=columns)  # type: ignore[name-defined]
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        return

    delimiter = "\t" if out_path.suffix.lower() == ".tsv" else ","
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, out_path)


def write_json_atomic(obj: Mapping[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, out_path)


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _verify_with_h5(pdb_id: str, expected: int, h5_dir: Path) -> None:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("h5py is required for H5 verification") from exc

    h5_path = h5_dir / f"{pdb_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file missing for {pdb_id}: {h5_path}")
    with h5py.File(h5_path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"H5 file for {pdb_id} missing 'coords' dataset")
        coords_len = handle["coords"].shape[0]
    if coords_len != expected:
        raise ValueError(
            f"Atom count mismatch for {pdb_id}: NPZ={expected}, H5 coords={coords_len}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate dataset level metrics")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory with NPZ artefacts")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics")
    parser.add_argument("--config", type=Path, help="Evaluation config YAML", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verify-h5", type=str, default="false")
    parser.add_argument("--h5-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--ece-bins", type=int, default=15)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = _load_yaml_config(args.config)
    positive_class = config.get("positive_class", DEFAULT_POSITIVE_CLASS)
    core_percentiles = config.get("core_percentiles", None)

    run_dir = args.run_dir
    if not run_dir.exists():
        LOGGER.error("Run directory does not exist: %s", run_dir)
        return 1

    npz_files = sorted(run_dir.glob("*.npz"))
    if not npz_files:
        LOGGER.error("No NPZ files found in %s", run_dir)
        return 1

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    start_time = time.time()

    for npz_file in npz_files:
        try:
            payload = load_npz(npz_file)
            metrics = compute_per_protein_metrics(
                payload,
                ece_bins=args.ece_bins,
                positive_class=positive_class,
                core_percentiles=core_percentiles,
            )
            metrics["pdb_id"] = payload.get("meta", {}).get("pdb_id", npz_file.stem)
            if args.verify_h5.lower() == "true":
                _verify_with_h5(str(metrics["pdb_id"]), metrics["num_atoms"], args.h5_dir)
            rows.append(metrics)
        except Exception as exc:
            LOGGER.warning("Skipping %s due to error: %s", npz_file.name, exc)
            skipped.append(npz_file.name)

    if not rows:
        LOGGER.error("No valid NPZ artefacts processed")
        return 1

    duration = time.time() - start_time
    LOGGER.info("Processed %d entries in %.2f seconds", len(rows), duration)
    if skipped:
        LOGGER.warning("Skipped %d files: %s", len(skipped), ", ".join(skipped))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_protein.csv"
    write_csv_or_tsv(rows, csv_path)

    summary = aggregate_metrics(rows, args.n_bootstrap, args.alpha, args.seed)
    summary["run_dir"] = str(run_dir)
    summary["stats"] = {"ece_bins": args.ece_bins}
    summary["config"] = config
    json_path = out_dir / "aggregate.json"
    write_json_atomic(summary, json_path)

    LOGGER.info("Wrote per-protein metrics to %s", csv_path)
    LOGGER.info("Wrote aggregate summary to %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())