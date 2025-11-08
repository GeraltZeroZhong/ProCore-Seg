"""Baseline evaluation adapters."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .evaluate_dataset import (
    DEFAULT_POSITIVE_CLASS,
    aggregate_metrics,
    compute_per_protein_metrics,
    write_csv_or_tsv,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)


def list_pdb_ids(h5_dir: Path) -> list[str]:
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory does not exist: {h5_dir}")
    return sorted(path.stem for path in h5_dir.glob("*.h5"))


def _load_h5_atom_data(pdb_id: str, h5_dir: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("h5py is required for baseline evaluation") from exc

    path = h5_dir / f"{pdb_id}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Processed H5 file missing for {pdb_id}: {path}")

    def _get_dataset(handle: Any, candidates: list[str], default: Any | None = None) -> np.ndarray:
        for candidate in candidates:
            if candidate in handle:
                data = handle[candidate][()]
                return np.array(data)
        if default is not None:
            return np.array(default)
        raise KeyError(f"None of the datasets {candidates} found in {path}")

    with h5py.File(path, "r") as handle:
        coords = _get_dataset(handle, ["coords", "coords_atom", "atoms/coords"])
        chain = _get_dataset(handle, ["chain", "atom_chain", "chains"], default=np.array([]))
        resseq = _get_dataset(handle, ["resseq", "atom_resseq", "res_ids"], default=np.array([]))
        icode = _get_dataset(handle, ["icode", "atom_icode"], default=np.zeros(coords.shape[0], dtype="U1"))
        labels = _get_dataset(handle, ["labels_atom", "atom_labels", "labels"])
        density = _get_dataset(handle, ["density_atom", "density"], default=np.ones(coords.shape[0], dtype=np.float32))
        num_voxels = handle.attrs.get("num_voxels", 0)

    chain = np.asarray(chain)
    if chain.dtype.kind in {"S", "O"}:
        chain = np.char.decode(chain.astype("S"), "utf-8")

    resseq = np.asarray(resseq)
    if resseq.dtype.kind in {"S", "O"}:
        resseq = np.char.decode(resseq.astype("S"), "utf-8")

    icode = np.asarray(icode)
    if icode.size == 0:
        icode = np.full(coords.shape[0], "", dtype="U1")
    elif icode.dtype.kind in {"S", "O"}:
        icode = np.char.decode(icode.astype("S"), "utf-8")

    return {
        "coords": np.asarray(coords, dtype=np.float32),
        "chain": chain.astype(str),
        "resseq": resseq.astype(str),
        "icode": icode.astype(str),
        "labels": np.asarray(labels, dtype=np.int32),
        "density": np.asarray(density, dtype=np.float32),
        "num_voxels": int(num_voxels),
    }


def _read_csv_probabilities(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"CSV baseline file missing: {path}")
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
        if "prob_pos" not in df.columns:
            raise KeyError(f"CSV file {path} must contain 'prob_pos' column")
        return df["prob_pos"].to_numpy(dtype=np.float64)
    except Exception:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            probs = [float(row["prob_pos"]) for row in reader]
        return np.asarray(probs, dtype=np.float64)


def _read_json_labels(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"JSON baseline file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): int(value) for key, value in payload.items()}


def _resolve_pattern(pattern: str, pdb_id: str) -> str:
    try:
        return pattern.format(pdb=pdb_id)
    except KeyError as exc:
        raise ValueError(f"Pattern {pattern!r} missing placeholder: {exc}") from exc


def build_pseudo_npz_from_adapter(
    adapter_name: str,
    inputs_cfg: dict[str, Any],
    pdb_id: str,
    h5_dir: Path,
) -> dict[str, Any]:
    atom_data = _load_h5_atom_data(pdb_id, h5_dir)
    n_atoms = atom_data["labels"].shape[0]
    base_payload = {
        "labels_atom": atom_data["labels"],
        "coords_atom": atom_data["coords"],
        "chain": atom_data["chain"],
        "resseq": atom_data["resseq"],
        "icode": atom_data["icode"],
        "density_atom": atom_data["density"],
        "meta": {
            "pdb_id": pdb_id,
            "num_atoms": int(n_atoms),
            "num_voxels": atom_data.get("num_voxels", 0),
        },
    }

    adapter_name = adapter_name.lower()
    inputs_dir = Path(inputs_cfg.get("dir", "."))
    pattern = inputs_cfg.get("pattern")
    if not pattern:
        raise ValueError(f"Adapter {adapter_name} requires 'pattern' in inputs configuration")

    if adapter_name == "csv_probs_per_atom":
        file_path = inputs_dir / _resolve_pattern(pattern, pdb_id)
        probs = _read_csv_probabilities(file_path)
        if probs.shape[0] != n_atoms:
            raise ValueError(
                f"Probability length mismatch for {pdb_id}: baseline={probs.shape[0]}, atoms={n_atoms}"
            )
        probs = np.clip(probs, 0.0, 1.0)
        probs_atom = np.stack([1.0 - probs, probs], axis=1)
        pred_atom = (probs >= 0.5).astype(np.int32)
    elif adapter_name == "labels_per_residue_json":
        file_path = inputs_dir / _resolve_pattern(pattern, pdb_id)
        residue_labels = _read_json_labels(file_path)
        preds = np.zeros(n_atoms, dtype=np.int32)
        probs = np.zeros(n_atoms, dtype=np.float64)
        keys = [f"{c}|{r}|{i}" for c, r, i in zip(base_payload["chain"], base_payload["resseq"], base_payload["icode"])]
        for idx, key in enumerate(keys):
            label = residue_labels.get(key, 0)
            preds[idx] = int(label)
            probs[idx] = float(label)
        probs_atom = np.stack([1.0 - probs, probs], axis=1)
        pred_atom = preds
    elif adapter_name == "labels_per_atom_npy":
        label_path = inputs_dir / _resolve_pattern(pattern, pdb_id)
        if not label_path.exists():
            raise FileNotFoundError(f"Label array missing for {pdb_id}: {label_path}")
        labels = np.load(label_path)
        if labels.shape[0] != n_atoms:
            raise ValueError(
                f"Label length mismatch for {pdb_id}: baseline={labels.shape[0]}, atoms={n_atoms}"
            )
        labels = labels.astype(np.int32)
        prob_pattern = inputs_cfg.get("prob_pattern")
        probs = None
        if prob_pattern:
            prob_path = inputs_dir / _resolve_pattern(prob_pattern, pdb_id)
            if prob_path.exists():
                probs = np.load(prob_path).astype(np.float64)
        if probs is None:
            probs = labels.astype(np.float64)
        else:
            if probs.shape[0] != n_atoms:
                raise ValueError(
                    f"Probability length mismatch for {pdb_id}: baseline={probs.shape[0]}, atoms={n_atoms}"
                )
        probs = np.clip(probs, 0.0, 1.0)
        probs_atom = np.stack([1.0 - probs, probs], axis=1)
        pred_atom = labels
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    payload = dict(base_payload)
    payload["probs_atom"] = probs_atom.astype(np.float64)
    payload["pred_atom"] = pred_atom.astype(np.int32)
    return payload


def evaluate_baseline(
    name: str,
    adapter_cfg: dict[str, Any],
    out_dir: Path,
    h5_dir: Path,
    n_bootstrap: int,
    alpha: float,
    seed: int,
    ece_bins: int,
    positive_class: int,
) -> dict[str, Any]:
    per_rows: list[dict[str, Any]] = []
    pdb_ids = list_pdb_ids(h5_dir)
    if not pdb_ids:
        raise ValueError(f"No H5 files found in {h5_dir}")

    adapter_name = adapter_cfg.get("adapter")
    inputs_cfg = adapter_cfg.get("inputs", {})
    if not adapter_name:
        raise ValueError("Adapter configuration must include 'adapter'")

    for pdb_id in pdb_ids:
        try:
            payload = build_pseudo_npz_from_adapter(adapter_name, inputs_cfg, pdb_id, h5_dir)
            metrics = compute_per_protein_metrics(
                payload,
                ece_bins=ece_bins,
                positive_class=positive_class,
            )
            metrics["pdb_id"] = pdb_id
            per_rows.append(metrics)
        except Exception as exc:
            LOGGER.warning("Skipping baseline %s entry %s due to error: %s", name, pdb_id, exc)

    if not per_rows:
        raise ValueError(f"No valid entries produced for baseline {name}")

    baseline_dir = out_dir / name
    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_csv_or_tsv(per_rows, baseline_dir / "per_protein.csv")

    summary = aggregate_metrics(per_rows, n_bootstrap, alpha, seed)
    summary["baseline"] = name
    summary["stats"] = {"ece_bins": ece_bins}
    write_json_atomic(summary, baseline_dir / "aggregate.json")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate external baselines")
    parser.add_argument("--adapters", type=Path, required=True, help="Adapters configuration YAML")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--h5-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--positive-class", type=int, default=DEFAULT_POSITIVE_CLASS)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with args.adapters.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    baselines_cfg = cfg.get("baselines", [])
    if not baselines_cfg:
        raise ValueError("Adapters configuration must define at least one baseline")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    key_metrics = ["iou", "core_iou_p50", "f1", "pr_auc"]

    for baseline in baselines_cfg:
        name = baseline.get("name")
        if not name:
            raise ValueError("Each baseline entry must include a name")
        summary = evaluate_baseline(
            name,
            baseline,
            out_dir,
            args.h5_dir,
            args.n_bootstrap,
            args.alpha,
            args.seed,
            args.ece_bins,
            args.positive_class,
        )
        summaries.append(summary)
        row = {"baseline": name}
        for metric in key_metrics:
            row[f"{metric}_mean"] = summary["means"].get(metric, float("nan"))
            ci = summary["ci95"].get(metric, [float("nan"), float("nan")])
            row[f"{metric}_ci_low"] = ci[0]
            row[f"{metric}_ci_high"] = ci[1]
        comparison_rows.append(row)

    write_csv_or_tsv(comparison_rows, out_dir / "comparison.csv")

    LOGGER.info("Baseline evaluation complete. Outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())