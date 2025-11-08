from __future__ import annotations

"""Shared type signatures for evaluation and inference utilities."""

from typing import Any, Dict, List, Optional, Tuple, TypedDict

FEATURE_ORDER: List[str] = ["C", "H", "O", "N", "S", "Other", "SASA", "OSP"]
DENSITY_FEATURE_INDEX: int = 7  # OSP


class InferenceMeta(TypedDict, total=False):
    """Metadata describing a single inference run for a protein."""

    pdb_id: str
    cath_id: str
    voxel_size: float
    num_voxels: int
    num_atoms: int
    timestamp: str
    checkpoint_voxel_size: float
    temperature: float
    run_id: str


class InferenceNPZPayload(TypedDict, total=False):
    """Structure expected when loading inference npz payloads."""

    logits_voxel: Any
    probs_voxel: Any
    pred_voxel: Any
    atom2voxel: Any
    probs_atom: Any
    pred_atom: Any
    labels_atom: Any
    density_atom: Any
    coords_atom: Any
    chain: Any
    resseq: Any
    icode: Any
    meta: InferenceMeta


PerProteinMetricsRow = TypedDict(
    "PerProteinMetricsRow",
    {
        "pdb_id": str,
        "acc": float,
        "precision": float,
        "recall": float,
        "f1": float,
        "pr_auc": float,
        "iou": float,
        "core_iou_p25": float,
        "core_iou_p50": float,
        "core_iou_p75": float,
        "boundary_f1@2": float,
        "boundary_f1@4": float,
        "boundary_f1@6": float,
        "boundary_iou@2": float,
        "boundary_iou@4": float,
        "boundary_iou@6": float,
        "ece": float,
        "mce": float,
        "brier": float,
        "nll": float,
        "num_atoms": int,
        "num_voxels": int,
        "sparsity": float,
    },
    total=False,
)
"""Schema for per-protein metrics table rows."""


class AggregateSummary(TypedDict, total=False):
    """Aggregate metrics summary schema."""

    run_dir: str
    generated_at: str
    n: int
    means: Dict[str, float]
    ci95: Dict[str, Tuple[float, float]]
    stats: Dict[str, float | int | str]
    env: Dict[str, str]
    config: Dict[str, Any]


if __name__ == "__main__":
    example_meta: InferenceMeta = {
        "pdb_id": "1ABC",
        "voxel_size": 1.5,
        "num_atoms": 42,
    }
    print("FEATURE_ORDER:", FEATURE_ORDER)
    print("DENSITY_FEATURE_INDEX:", DENSITY_FEATURE_INDEX)
    print("Example meta:", example_meta)
